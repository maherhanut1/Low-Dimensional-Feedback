import argparse
import yaml
import os
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
from models.FC import FC
import torch
from training_utils.trainer import Trainer
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def replace_linear(module, new_linear_cls, **kwargs):
    for name, child in module.named_children():
        if 'patch_embed' in name:
            continue  # Special case for ViT classifier head
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            curr_kwargs = kwargs.copy()
           
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
    for module in trainer.model.modules():
        if hasattr(module, 'init_svd_approx'):
            module.init_svd_approx()
    # Clear qp_optimizer state (assume it's the second optimizer in the list)
    # if len(trainer.optimizers) > 1:
    #     trainer.optimizers[0].state.clear()
    #     trainer.optimizers[1].state.clear()


def get_cifar100_loaders(batch_size=32, num_subset_classes=1000, root='./data', num_workers= 4):
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    test_idx = [i for i in range(len(testset.targets)) if testset.targets[i] < num_subset_classes]
    train_idx = [i for i in range(len(trainset.targets)) if trainset.targets[i] < num_subset_classes]
    testset = torch.utils.data.Subset(testset, test_idx)
    trainset = torch.utils.data.Subset(trainset, train_idx)
    testloader = DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=1)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader, testloader

def get_cifar10_loaders(batch_size=32, root='./data', num_workers= 4):
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

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
    eta_min = config.get('eta_min', 1e-6)
    qp_eta_min = config.get('qp_eta_min', 1e-6)

    use_ldfa_linear = config.get('use_ldfa_linear', True)
    ldfa_rank = config.get('ldfa_rank', 32)
    qp_lr = config.get('qp_lr', lr)
    qp_weight_decay = config.get('qp_weight_decay', weight_decay)
    model_name = config.get('model_name', 'FC')
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    num_subset_classes = config.get('num_subset_classes', None)  # For CIFAR-100 subset
    log_name = config.get('log_name', 'default_run')
    embed_dim = config.get('embed_dim')
    drop_rate = config.get('drop_rate')

    device = 'cuda' #'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update num_classes if using CIFAR-100 subset BEFORE creating data loaders and model
    if dataset.lower() == 'cifar100' and num_subset_classes is not None and num_subset_classes < 100:
        print(f"Overriding num_classes from {num_classes} to {num_subset_classes} for CIFAR-100 subset")
        num_classes = num_subset_classes
    
    # Data
    if dataset.lower() == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, root='./data', num_workers=12)
    elif dataset.lower() == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size, root='./data', num_subset_classes=num_subset_classes,  num_workers=12)
    # elif dataset.lower() == 'imagenet':
    #     # You may want to set the path in your config as 'imagenet_dir'
    #     imagenet_dir = config.get('imagenet_dir', '/home/maherhanut/Documents/data/imagenet')
    #     train_loader, test_loader = get_imagenet_loaders(data_dir=imagenet_dir, batch_size=batch_size)
    # else:
    #     raise ValueError(f"Unknown dataset: {dataset}")
    
    model = FC(3 * (image_size ** 2), hidden_dim=embed_dim, num_classes=num_classes, drop_out_ratio=drop_rate, device=device)
    
    if use_ldfa_linear:
        replace_linear(model, LDFA_Linear, rank=ldfa_rank)
    # else:
    #     replace_linear(model, BP_Linear)
    model = model.to(device)

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
        model_params = []
            
        for name, param in model.named_parameters():
            if 'P' in name or 'Q' in name:
                qp_params.append(param)
            else:
                model_params.append(param)

        model_optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay, amsgrad=True)
        qp_optimizer = optim.Adam(qp_params, lr=qp_lr, weight_decay=qp_weight_decay, amsgrad=True)

        qp_scheduler = torch.optim.lr_scheduler.ExponentialLR(qp_optimizer, gamma=0.975)
        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.975)

        optimizers = [model_optimizer, qp_optimizer]
        schedulers = [main_scheduler, qp_scheduler]
        modify_funcs = [lambda trainer: reinitialize_pq_layers(trainer, 0.5)]
        modification_rate = len(train_loader) // 2  # Reinit every half epoch

    else:

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=10 * len(train_loader))
        # main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = (num_epochs - 10) * len(train_loader), eta_min=eta_min)
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_scheduler, main_scheduler],
        #     # milestones=[10 * len(train_loader)]
        # )

        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)


        schedulers = [main_scheduler]
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
