import yaml
import os
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
import torch
from torchvision.models import vit_b_16
from training_utils.trainer import Trainer
from training_utils.data_loader_factory import get_data_loaders
import torch.nn as nn
import torch.optim as optim

def accuracy_metric(outputs, targets):
    # If outputs is a dict (e.g., {'logits': ...}), extract logits
    if isinstance(outputs, dict) and 'logits' in outputs:
        outputs = outputs['logits']
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc, 'accuracy'

def main():
    # Load config from YAML
    config_path = os.path.join(os.path.dirname(__file__), '../configs/train_vit_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = config.get('dataset', 'cifar10')
    batch_size = config.get('batch_size', 64)
    num_epochs = config.get('num_epochs', 150)
    lr = config.get('learning_rate', 5e-4)
    use_ldfa_linear = config.get('use_ldfa_linear', True)
    ldfa_rank = config.get('ldfa_rank', 16)
    model_name = config.get('model_name', 'vit_b_16')
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    log_name = config.get('log_name', 'default_run')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size)

    # Model
    model = vit_b_16(num_classes=num_classes, image_size=image_size)
    if use_ldfa_linear:
        def replace_linear(module, new_linear_cls, **kwargs):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    in_features = child.in_features
                    out_features = child.out_features
                    bias = child.bias is not None
                    new_linear = new_linear_cls(in_features, out_features, **kwargs, bias=bias)
                    setattr(module, name, new_linear)
                else:
                    replace_linear(child, new_linear_cls, **kwargs)
        replace_linear(model, LDFA_Linear, rank=ldfa_rank)
    model = model.to(device)

    # Loss and optimizer
    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Add OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        anneal_strategy='linear',
        pct_start=0.15
    )
    # Wrap scheduler in a dict to indicate it should be stepped per batch
    schedulers = [scheduler]

    # Metrics
    metrics = [accuracy_metric]

    # Set a name for this run (from config)
    checkpoint_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}'
    log_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}/logs'

    trainer = Trainer(
        model=model,
        optimizers=[optimizer],
        schedulers=schedulers,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics=metrics,
        num_epochs=num_epochs,
        loss_fns=loss_fns,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    trainer.train()

if __name__ == '__main__':
    main()
