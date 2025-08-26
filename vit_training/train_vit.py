import torch
from torchvision.models import vit_b_16
from training_utils.trainer import Trainer
from training_utils.data_loader_factory import get_data_loaders
import torch.nn as nn
import torch.optim as optim

def accuracy_metric(data_loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc, 'accuracy'

def main():
    # Config
    dataset = 'cifar10'
    batch_size = 64
    num_iterations = 200
    evaluation_iterations = 20
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size)

    # Model
    model = vit_b_16(num_classes=10, image_size=32)
    model = model.to(device)

    # Loss and optimizer
    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedulers = []

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
