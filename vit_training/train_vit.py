import argparse
import yaml
import os
from functools import partial
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
from models.BP_ViT import BPVit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import vit_b_16
from torchvision.transforms import v2 as T2
from training_utils.trainer import Trainer
from training_utils.data_loader_factory import get_cifar10_loaders, get_cifar100_loaders, get_imagenet_loaders, get_tiny_imagenet_loaders, get_imagenet100_loaders
import torch.nn as nn
import torch.optim as optim
from timm.models.vision_transformer import VisionTransformer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def get_param_groups(named_params, weight_decay, no_wd_bias_norm=False):
    """
    Split parameters into two groups:
      - decay group:    all weight matrices (weight decay applied)
      - no-decay group: biases + norm layer weights/biases (no weight decay)

    If no_wd_bias_norm=False, returns a single flat list (all params get weight_decay).
    """
    if not no_wd_bias_norm:
        return [p for _, p in named_params]

    decay, no_decay = [], []
    no_decay_names = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        # Exclude biases, 1-D params (LayerNorm weight/bias, etc.),
        # and ViT-specific embeddings (cls_token, pos_embed)
        if param.ndim == 1 or name.endswith('.bias') or 'cls_token' in name or 'pos_embed' in name:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)

    print(f"  param groups — decay: {len(decay)}, no-decay: {len(no_decay)} {no_decay_names[:5]}{'...' if len(no_decay_names)>5 else ''}")
    return [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

def replace_linear(module, new_linear_cls, **kwargs):
    for name, child in module.named_children():
        if 'patch_embed' in name:
            continue  # Special case for ViT classifier head
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            curr_kwargs = kwargs.copy()

            if 'rank' in curr_kwargs and 'qkv' in name:
                curr_kwargs['rank'] = kwargs['rank'] * 3  # Triple rank for QKV layers
           
            new_linear = new_linear_cls(in_features, out_features, **curr_kwargs, bias=bias)
            new_linear.weight.data = child.weight.data.clone()
            if bias:
                new_linear.bias.data = child.bias.data.clone()
            
            new_linear.init_svd_approx() if hasattr(new_linear, 'init_svd_approx') else None
            setattr(module, name, new_linear)
        else:
            replace_linear(child, new_linear_cls, **kwargs)


def replace_linear_by_layer(model, layer_rank_map, default_rank, **kwargs):
    """
    Replace linear layers in each ViT block (model.blocks[i]) using a per-layer rank map.

    layer_rank_map: dict of {layer_index (int) -> rank (int)}
        - rank > 0  : use LDFA_Linear with that rank (QKV layers get rank * 3)
        - rank == -1: keep as BP_Linear (standard backprop, no low-rank feedback)
    default_rank: fallback rank for layers not listed in layer_rank_map.

    Layers outside model.blocks (e.g. head) are always left as BP_Linear.
    """
    for i, block in enumerate(model.blocks):
        rank = layer_rank_map.get(i, default_rank)
        if rank == -1:
            replace_linear(block, BP_Linear)
        else:
            replace_linear(block, LDFA_Linear, rank=rank, **kwargs)


def reinitialize_pq_layers(trainer, r=None):
    """Reinitialize P and Q matrices for all rAFA layers in the model and clear qp_optimizer state"""
    model = getattr(trainer.model, '_orig_mod', trainer.model)
    for module in model.modules():
        if hasattr(module, 'init_svd_approx'):
            module.init_svd_approx()
    # Clear qp_optimizer state (assume it's the second optimizer in the list)
    # if len(trainer.optimizers) > 1:
    #     trainer.optimizers[0].state.clear()
    #     trainer.optimizers[1].state.clear()


def accuracy_metric(outputs, targets):
    # If outputs is a dict (e.g., {'logits': ...}), extract logits
    if isinstance(outputs, dict) and 'logits' in outputs:
        outputs = outputs['logits']
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    acc = correct / total if total > 0 else 0.0

    if int(os.environ.get('RANK', '0')) == 0:
        print(f'acc: {acc}')
    return acc, 'accuracy'


def topk_accuracy_metric(outputs, targets, k=5):
    # If outputs is a dict (e.g., {'logits': ...}), extract logits
    if isinstance(outputs, dict) and 'logits' in outputs:
        outputs = outputs['logits']
    
    # Get top-5 predictions
    _, predicted_topk = torch.topk(outputs, k=k, dim=1)
    
    # Expand targets to compare with top-5 predictions
    targets_expanded = targets.unsqueeze(1).expand_as(predicted_topk)
    
    # Check if true label is in top-5 predictions
    correct = (predicted_topk == targets_expanded).any(dim=1).sum().item()
    total = targets.size(0)
    topk_acc = correct / total if total > 0 else 0.0
    
    if int(os.environ.get('RANK', '0')) == 0:
        print(f'top{k}_acc: {topk_acc}')
    return topk_acc, f'top{k}_accuracy'

def main():

    parser = argparse.ArgumentParser(description='Train ViT with LDFA or BP')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Multi-GPU setup
    # gpu_ids can be a list [0, 1, ...] or a single int / absent (single-GPU default)
    gpu_ids_raw = config.get('gpu_ids', None)
    if gpu_ids_raw is None:
        gpu_ids = [0]
    elif isinstance(gpu_ids_raw, int):
        gpu_ids = [gpu_ids_raw]
    else:
        gpu_ids = list(gpu_ids_raw)

    world_size = len(gpu_ids)

    if world_size > 1:
        # Launch one process per GPU via mp.spawn
        mp.spawn(train_worker, args=(world_size, gpu_ids, config), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, gpu_ids, config)


def train_worker(rank, world_size, gpu_ids, config):
    """Training function executed by each DDP process (or directly for single-GPU)."""
    is_ddp = world_size > 1
    local_gpu = gpu_ids[rank]
    device = f'cuda:{local_gpu}'

    if is_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(config.get('master_port', os.environ.get('MASTER_PORT', '12355')))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_gpu)

    dataset = config.get('dataset', 'cifar10')
    batch_size = config.get('batch_size', 128)
    num_epochs = config.get('num_epochs', 150)
    lr = float(config.get('learning_rate', 3e-4))
    weight_decay = float(config.get('weight_decay', 1e-4))
    no_wd_bias_norm = config.get('no_wd_bias_norm', False)
    eta_min = float(config.get('eta_min', 1e-6))
    qp_eta_min = float(config.get('qp_eta_min', 1e-6))
    warmup_epochs = config.get('warmup_epochs', 10)

    # Mixup / CutMix / Label smoothing
    mixup_alpha    = float(config.get('mixup_alpha', 0.0))
    cutmix_alpha   = float(config.get('cutmix_alpha', 0.0))
    label_smoothing = float(config.get('label_smoothing', 0.0))

    use_ldfa_linear = config.get('use_ldfa_linear', True)
    ldfa_rank = config.get('ldfa_rank', 32)
    ldfa_layer_ranks_raw = config.get('ldfa_layer_ranks', None)
    # YAML keys are strings; convert to int->int dict
    ldfa_layer_ranks = {int(k): int(v) for k, v in ldfa_layer_ranks_raw.items()} if ldfa_layer_ranks_raw else None
    qp_lr = float(config.get('qp_lr', lr))
    qp_weight_decay = float(config.get('qp_weight_decay', weight_decay))
    model_name = config.get('model_name', 'vit_b_16')
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    num_subset_classes = config.get('num_subset_classes', None)  # For CIFAR-100 subset
    log_name = config.get('log_name', 'default_run')
    grad_clip = config.get('grad_clip', None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    use_ema = config.get('use_ema', False)
    ema_decay = float(config.get('ema_decay', 0.9999)) if use_ema else None
    num_workers = config.get('num_workers', 16)

    #load model parameters
    patch_size = config.get('patch_size')
    in_chans = config.get('in_chans')
    embed_dim = config.get('embed_dim')
    depth = config.get('depth')
    num_heads = config.get('num_heads')
    mlp_ratio = config.get('mlp_ratio')
    qkv_bias = config.get('qkv_bias')
    drop_rate = config.get('drop_rate')
    attn_drop_rate = config.get('attn_drop_rate')
    drop_path_rate = config.get('drop_path_rate')

    # Update num_classes if using CIFAR-100 subset BEFORE creating data loaders and model
    if dataset.lower() == 'cifar100' and num_subset_classes is not None and num_subset_classes < 100:
        print(f"Overriding num_classes from {num_classes} to {num_subset_classes} for CIFAR-100 subset")
        num_classes = num_subset_classes
    
    # Data
    if dataset.lower() == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, root='./data', num_workers=12)
    elif dataset.lower() == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size, root='./data', num_subset_classes=num_subset_classes)
    elif dataset.lower() == 'tiny_imagenet':
        tiny_imagenet_dir = config.get('data_dir', './data/tiny-imagenet-200')
        train_loader, test_loader = get_tiny_imagenet_loaders(data_dir=tiny_imagenet_dir, batch_size=batch_size)
    elif dataset.lower() == 'imagenet100':
        imagenet_dir = config.get('data_dir', '/home/maherhanut/Documents/data/imagenet')
        class_list_file = config.get('class_list_file', './data/imagenet100_classes.txt')
        train_loader, test_loader = get_imagenet100_loaders(
            data_dir=imagenet_dir, batch_size=batch_size, class_list_file=class_list_file)
    elif dataset.lower() == 'imagenet':
        imagenet_dir = config.get('imagenet_dir', '/home/maherhanut/Documents/data/imagenet')
        train_loader, test_loader = get_imagenet_loaders(data_dir=imagenet_dir, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # --- DDP: replace train_loader with DistributedSampler ---
    # Each GPU sees batch_size samples per step → effective batch = world_size * batch_size
    # (equivalent to single-GPU training with world_size * batch_size)
    train_sampler = None
    eval_train_loader = train_loader  # full train set for logging (rank 0 only)
    if is_ddp:
        # Divide workers evenly across processes to keep total CPU/IO load the same as single-GPU
        workers_per_proc = max(1, train_loader.num_workers // world_size)
        train_sampler = DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=workers_per_proc,
            pin_memory=True,
            prefetch_factor=4,        # prefetch more to keep GPU fed despite fewer workers
            persistent_workers=True,
        )
        # Rank 0 uses a separate full (non-distributed) loader for train metric logging
        if rank == 0:
            eval_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers_per_proc,
                pin_memory=True,
            )
        else:
            eval_train_loader = None  # non-rank-0 processes don't log

    # --- Mixup / CutMix (applied as a batch-level transform on GPU) ---
    mixup_cutmix_transforms = []
    if mixup_alpha > 0.0:
        mixup_cutmix_transforms.append(T2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0.0:
        mixup_cutmix_transforms.append(T2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if mixup_cutmix_transforms:
        mixup_cutmix_fn = T2.RandomChoice(mixup_cutmix_transforms)
    else:
        mixup_cutmix_fn = None

    model = VisionTransformer(img_size=image_size,
                              patch_size=patch_size,
                              in_chans=in_chans,
                              num_classes=num_classes,
                              embed_dim=embed_dim,
                              depth=depth,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              drop_rate=drop_rate,
                              attn_drop_rate=attn_drop_rate,
                              drop_path_rate=drop_path_rate)
    
    if use_ldfa_linear:
        if ldfa_layer_ranks is not None:
            replace_linear_by_layer(model, ldfa_layer_ranks, default_rank=ldfa_rank)
        else:
            replace_linear(model, LDFA_Linear, rank=ldfa_rank)
    else:
        replace_linear(model, BP_Linear)
    model = model.to(device)
    model = torch.compile(model)   # compile before DDP to avoid capturing all-reduce in the graph
    if is_ddp:
        model = DDP(model, device_ids=[local_gpu], output_device=local_gpu)

    print('*******', use_ldfa_linear, "###########")

    # Loss — soft targets when Mixup/CutMix active, label smoothing always
    base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if mixup_cutmix_fn is not None:
        # During training targets are soft (B, C) from Mixup/CutMix;
        # during validation targets are hard (B,) integer indices — handle both.
        def soft_ce(outputs, targets):
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                # Hard labels (validation) — delegate to standard CE
                return base_criterion(outputs, targets)
            # Soft labels (training with Mixup/CutMix)
            loss = -(targets * log_probs).sum(dim=-1).mean()
            if label_smoothing > 0:
                smooth_loss = -log_probs.mean(dim=-1).mean()
                loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss
            return loss
        loss_fns = [(soft_ce, 1.0)]
    else:
        loss_fns = [(base_criterion, 1.0)]

    # Metrics
    metrics = [accuracy_metric,
               lambda x, y: topk_accuracy_metric(x, y, k=5),
               lambda x, y: topk_accuracy_metric(x, y, k=2)
               ]

    # Set a name for this run (from config)
    checkpoint_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}'
    log_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}/logs'


    if use_ldfa_linear:
        qp_params = []
        model_named_params = []
        # Use the underlying module's named_parameters (bypasses DDP/compile wrappers)
        base_model = getattr(getattr(model, '_orig_mod', model), 'module', getattr(model, '_orig_mod', model))
        for name, param in model.named_parameters():
            if 'P' in name or 'Q' in name:
                qp_params.append(param)
            else:
                model_named_params.append((name, param))

        warmup_steps = warmup_epochs * len(train_loader)

        model_pg = get_param_groups(model_named_params, weight_decay, no_wd_bias_norm)
        model_optimizer = optim.AdamW(model_pg, lr=lr, weight_decay=weight_decay)
        qp_optimizer = optim.AdamW(qp_params, lr=qp_lr, weight_decay=qp_weight_decay)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            model_optimizer, start_factor=1/warmup_epochs, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optimizer, T_max=(num_epochs - warmup_epochs) * len(train_loader), eta_min=eta_min)
        model_scheduler = torch.optim.lr_scheduler.SequentialLR(
            model_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

        qp_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            qp_optimizer, start_factor=1/warmup_epochs, end_factor=1.0, total_iters=warmup_steps)
        qp_main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            qp_optimizer, T_max=(num_epochs - warmup_epochs) * len(train_loader), eta_min=qp_eta_min)
        qp_scheduler = torch.optim.lr_scheduler.SequentialLR(
            qp_optimizer, schedulers=[qp_warmup_scheduler, qp_main_scheduler], milestones=[warmup_steps])

        optimizers = [model_optimizer, qp_optimizer]
        schedulers = [model_scheduler, qp_scheduler]
        modify_funcs = [lambda trainer: reinitialize_pq_layers(trainer, 0.5)]
        modification_rate = len(train_loader) // 2

    else:
        warmup_steps = warmup_epochs * len(train_loader)

        model_pg = get_param_groups(model.named_parameters(), weight_decay, no_wd_bias_norm)
        optimizer = optim.AdamW(model_pg, lr=lr, weight_decay=weight_decay)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1/warmup_epochs, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(num_epochs - warmup_epochs) * len(train_loader), eta_min=eta_min)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

        schedulers = [scheduler]
        optimizers = [optimizer]
        modify_funcs = None
        modification_rate = None




    trainer = Trainer(
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics=metrics,
        num_epochs=num_epochs,
        loss_fns=loss_fns,
        model_modify_fns=modify_funcs,
        model_modify_iters=modification_rate,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        mixup_cutmix_fn=mixup_cutmix_fn,
        grad_clip=grad_clip,
        ema_decay=ema_decay,
        rank=rank,
        train_sampler=train_sampler,
        eval_train_loader=eval_train_loader,
    )
    trainer.train()

    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
