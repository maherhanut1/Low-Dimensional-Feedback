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
    # Config
    dataset = 'cifar10'
    batch_size = 256
    num_iterations = 391 * 150
    evaluation_iterations = 391
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size)

    # Model
    model = vit_b_16(num_classes=10, image_size=32)
    model = model.to(device)

    # Loss and optimizer
    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Add OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    epochs = num_iterations // steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        anneal_strategy='linear',
        pct_start=0.3
    )
    # Wrap scheduler in a dict to indicate it should be stepped per batch
    schedulers = [scheduler]

    # Metrics
    metrics = [accuracy_metric]

    # Set a name for this run (can be changed as needed)
    name = 'default_run'  # You can set this dynamically or via CLI/config

    checkpoint_dir = f'artifacts/training_checkpoints/cifar10/vit16b/{name}'
    log_dir = f'artifacts/training_checkpoints/cifar10/vit16b/{name}/logs'

    trainer = Trainer(
        model=model,
        optimizers=[optimizer],
        schedulers=schedulers,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics=metrics,
        num_iterations=num_iterations,
        evaluation_iterations=evaluation_iterations,
        loss_fns=loss_fns,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    trainer.train()

if __name__ == '__main__':
    main()
