import argparse
import yaml
import os
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
from models.BP_ViT import BPVit
import torch
from torchvision.models import vit_b_16
from training_utils.trainer import Trainer
from training_utils.data_loader_factory import get_data_loaders
import torch.nn as nn
import torch.optim as optim


def replace_linear(module, new_linear_cls, **kwargs):
    """
    Replace nn.Linear or BP_Linear layers with new_linear_cls, optionally copying weights if possible.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) or child.__class__.__name__ == 'Linear':
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            new_linear = new_linear_cls(in_features, out_features, **kwargs, bias=bias)
            # Copy weights and bias if possible
            with torch.no_grad():
                if hasattr(new_linear, 'weight') and hasattr(child, 'weight'):
                    new_linear.weight.copy_(child.weight)
                if hasattr(new_linear, 'bias') and hasattr(child, 'bias') and child.bias is not None:
                    new_linear.bias.copy_(child.bias)
            setattr(module, name, new_linear)
        else:
            replace_linear(child, new_linear_cls, **kwargs)


def reinitialize_pq_layers(trainer, r=None):
    """Reinitialize P and Q matrices for all rAFA layers in the model and clear qp_optimizer state"""
    for module in trainer.model.modules():
        if hasattr(module, 'init_svd_approx'):
            module.init_svd_approx()
    # Clear qp_optimizer state (assume it's the second optimizer in the list)
    if len(trainer.optimizers) > 1:
        trainer.optimizers[1].state.clear()


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




def main():
    parser = argparse.ArgumentParser(description='Hybrid BP/LDFA ViT Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs with BP before switching to LDFA')
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

    ldfa_rank = config.get('ldfa_rank', 16)
    qp_lr = config.get('qp_lr', lr)
    qp_weight_decay = config.get('qp_weight_decay', weight_decay)
    model_name = config.get('model_name', 'vit_b_16')
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    log_name = config.get('log_name', 'hybrid_run')

    warmup_epochs = args.warmup_epochs
    assert warmup_epochs < num_epochs, "warmup_epochs must be less than total num_epochs"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size)

    # Model
    model = BPVit(
        image_size=image_size,
        patch_size=4,
        num_classes=num_classes,
        dim=384,
        depth=6,
        heads=8,
        mlp_dim=384,
        dropout=0.1,
        emb_dropout=0.1,
    )

    # --- Phase 1: BP warmup ---
    replace_linear(model, BP_Linear)
    model = model.to(device)

    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]
    steps_per_epoch = len(train_loader)
    metrics = [accuracy_metric]

    checkpoint_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}'
    log_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}/logs'

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        anneal_strategy='linear',
        pct_start=0.15
    )
    schedulers = [scheduler]
    optimizers = [optimizer]

    trainer = Trainer(
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics=metrics,
        num_epochs=warmup_epochs,
        loss_fns=loss_fns,
        model_modify_fns=None,
        model_modify_iters=None,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    print(f"[Hybrid] Starting BP warmup for {warmup_epochs} epochs...")
    trainer.train()

    # --- Phase 2: Switch to LDFA ---
    print("[Hybrid] Switching to LDFA_Linear layers and continuing training...")
    # Transfer weights from BP_Linear to LDFA_Linear
    replace_linear(model, LDFA_Linear, rank=ldfa_rank)
    model = model.to(device)

    # Update trainer's model
    trainer.model = model

    # Update optimizer param groups in-place to point to new model's parameters (for non-LDFA params)
    new_params = [p for name, p in model.named_parameters() if not (('P' in name) or ('Q' in name))]
    trainer.optimizers[0].param_groups[0]['params'] = new_params

    # Add new optimizer for LDFA P/Q params
    qp_params = [p for name, p in model.named_parameters() if ('P' in name) or ('Q' in name)]
    if qp_params:
        qp_optimizer = optim.AdamW(qp_params, lr=qp_lr, weight_decay=qp_weight_decay)
        trainer.optimizers.append(qp_optimizer)
        # Add dummy scheduler for new optimizer (constant LR)
        qp_scheduler = torch.optim.lr_scheduler.LambdaLR(qp_optimizer, lr_lambda=lambda epoch: 1.0)
        trainer.schedulers.append(qp_scheduler)

    # Set LDFA modification functions and rate
    trainer.model_modify_fns = [lambda trainer: reinitialize_pq_layers(trainer, 0.5)]
    trainer.model_modify_iters = 50

    # Continue training with same trainer, optimizer, scheduler, and TensorBoard writer
    trainer.num_epochs = num_epochs - warmup_epochs
    print(f"[Hybrid] Continuing training with LDFA for {trainer.num_epochs} epochs...")
    trainer.train()


if __name__ == '__main__':
    main()
    main()
