import argparse
import yaml
import os
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
from models.BP_ViT import BPVit
import torch
from torchvision.models import vit_b_16
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

    dataset = config.get('dataset', 'cifar10')
    batch_size = config.get('batch_size', 128)
    num_epochs = config.get('num_epochs', 150)
    lr = config.get('learning_rate', 3e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    no_wd_bias_norm = config.get('no_wd_bias_norm', False)
    eta_min = config.get('eta_min', 1e-6)
    qp_eta_min = config.get('qp_eta_min', 1e-6)

    use_ldfa_linear = config.get('use_ldfa_linear', True)
    ldfa_rank = config.get('ldfa_rank', 32)
    qp_lr = config.get('qp_lr', lr)
    qp_weight_decay = config.get('qp_weight_decay', weight_decay)
    model_name = config.get('model_name', 'vit_b_16')
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    num_subset_classes = config.get('num_subset_classes', None)  # For CIFAR-100 subset
    log_name = config.get('log_name', 'default_run')

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

    device = 'cuda' #'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        # You may want to set the path in your config as 'imagenet_dir'
        imagenet_dir = config.get('imagenet_dir', '/home/maherhanut/Documents/data/imagenet')
        train_loader, test_loader = get_imagenet_loaders(data_dir=imagenet_dir, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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
        replace_linear(model, LDFA_Linear, rank=ldfa_rank)
    else:
        replace_linear(model, BP_Linear)
    model = model.to(device)
    model = torch.compile(model)

    print('*******', use_ldfa_linear, "###########")

    # Loss and optimizer
    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]

    # Metrics
    metrics = [accuracy_metric,
               lambda x, y: topk_accuracy_metric(x, y, k=5),
               lambda x, y: topk_accuracy_metric(x, y, k=2)
               ]

    # Set a name for this run (from config)
    checkpoint_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}'
    log_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}/logs'


    if use_ldfa_linear:
        # modifiable_modules = [module for module in model.modules() if hasattr(module, 'init_svd_approx')]
        qp_params = []
        model_named_params = []
            
        for name, param in model.named_parameters():
            if 'P' in name or 'Q' in name:
                qp_params.append(param)
            else:
                model_named_params.append((name, param))

        model_pg = get_param_groups(model_named_params, weight_decay, no_wd_bias_norm)
        model_optimizer = optim.AdamW(model_pg, lr=lr, weight_decay=weight_decay)
        qp_optimizer = optim.AdamW(qp_params, lr=qp_lr, weight_decay=qp_weight_decay)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(model_optimizer, start_factor=1/25, end_factor=1.0, total_iters=10 * len(train_loader))
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max = (num_epochs - 10) * len(train_loader), eta_min=eta_min)
        model_scheduler = torch.optim.lr_scheduler.SequentialLR(
            model_optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[10 * len(train_loader)]
        )


        qp_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(qp_optimizer, start_factor=1/10, end_factor=1.0, total_iters=10 * len(train_loader))
        qp_main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(qp_optimizer, T_max = (num_epochs - 10) * len(train_loader), eta_min=qp_eta_min)

        qp_scheduler = torch.optim.lr_scheduler.SequentialLR(
            qp_optimizer,
            schedulers=[qp_warmup_scheduler, qp_main_scheduler],
            milestones=[10 * len(train_loader)]
        )

        optimizers = [model_optimizer, qp_optimizer]
        schedulers = [model_scheduler, qp_scheduler]
        modify_funcs = [lambda trainer: reinitialize_pq_layers(trainer, 0.5)]
        modification_rate = len(train_loader) // 2  # Reinit every half epoch

    else:

        model_pg = get_param_groups(model.named_parameters(), weight_decay, no_wd_bias_norm)
        optimizer = optim.AdamW(model_pg, lr=lr, weight_decay=weight_decay)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=10 * len(train_loader))
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = (num_epochs - 10) * len(train_loader), eta_min=eta_min)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[10 * len(train_loader)]
        )

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
        device=device
    )
    trainer.train()

if __name__ == '__main__':
    main()
