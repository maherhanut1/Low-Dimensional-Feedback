import argparse
import yaml
import os
from modules.opt_layers.LDFA_Conv import Conv2d as LDFA_Conv2d
from models.ConvNet import CIFAR10CNN
import torch
from torchvision.models import resnet18
from training_utils.trainer import Trainer
from training_utils.data_loader_factory import get_data_loaders
import torch.nn as nn
import torch.optim as optim


def replace_conv2d(module, new_conv_cls, retain_weights=False, **kwargs):
    """
    Recursively replaces all nn.Conv2d layers in a module with a new
    convolutional layer class.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = new_conv_cls(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                **kwargs
            )

            if retain_weights:
                # Retain original weights and bias if possible
                with torch.no_grad():
                    new_conv.weight.data = (child.weight.data)
                    if child.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data = (child.bias.data)

            setattr(module, name, new_conv)
        else:
            replace_conv2d(child, new_conv_cls, retain_weights=retain_weights, **kwargs)


def reinitialize_pq_layers(trainer, r=None):
    """Reinitialize P and Q matrices for all rAFA layers in the model and clear qp_optimizer state"""
    for module in trainer.model.modules():
        if hasattr(module, 'init_svd_approx'):
            module.init_svd_approx()
    # Clear qp_optimizer state (assume it's the second optimizer in the list)
    if len(trainer.optimizers) > 1:
        trainer.optimizers[1].state.clear()
    
    print(f'Reinitialized P and Q matrices for all LDFA layers. r={r}')


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

    use_ldfa_linear = config.get('use_ldfa_linear', True)
    ldfa_rank = config.get('ldfa_rank', 32)
    qp_lr = config.get('qp_lr', lr)
    qp_weight_decay = config.get('qp_weight_decay', weight_decay)
    model_name = config.get('model_name', 'vit_b_16')
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    log_name = config.get('log_name', 'default_run')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size)

    # Model
    # model = resnet18(weights='DEFAULT')
    # # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.maxpool = nn.Identity() # CIFAR-10 is small, maxpool is often removed
    # model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = CIFAR10CNN(num_classes=num_classes)

    if use_ldfa_linear:
        replace_conv2d(model, LDFA_Conv2d, retain_weights=True, rank=ldfa_rank)

    model = model.to(device)

    print('*******', use_ldfa_linear, "###########")

    # Loss and optimizer
    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]
    # Add OneCycleLR scheduler
    steps_per_epoch = len(train_loader)

    # Metrics
    metrics = [accuracy_metric]

    # Set a name for this run (from config)
    checkpoint_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}'
    log_dir = f'artifacts/training_checkpoints/{dataset}/{model_name}/{log_name}/logs'


    if use_ldfa_linear:
        # modifiable_modules = [module for module in model.modules() if hasattr(module, 'init_svd_approx')]
        qp_params = []
        model_params = []
            
        for name, param in model.named_parameters():
            if 'P' in name or 'Q' in name:
                qp_params.append(param)
            else:
                model_params.append(param)
        
        model_optimizer = optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        qp_optimizer = optim.AdamW(qp_params, lr=qp_lr, weight_decay=qp_weight_decay)


        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        model_optimizer,
        gamma=0.98)
    

        qp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        qp_optimizer,
        gamma=0.98)
        
        #qp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(qp_optimizer, T_max=num_epochs*steps_per_epoch, eta_min=1e-6)

        optimizers = [model_optimizer, qp_optimizer]
        schedulers = [model_scheduler, qp_scheduler]
        modify_funcs = [lambda trainer: reinitialize_pq_layers(trainer, 0.5)]
        modification_rate = 391

    else:

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.98)
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
        scheduler_per_epoch=True,
    )
    trainer.train()

if __name__ == '__main__':
    main()
