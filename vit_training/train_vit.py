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
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            new_linear = new_linear_cls(in_features, out_features, **kwargs, bias=bias)
            setattr(module, name, new_linear)
        else:
            replace_linear(child, new_linear_cls, **kwargs)


# def ldfa_layers_loss(model):
#     loss = 0.0
#     for module in model.modules():
#         if hasattr(module, "P") and hasattr(module, "Q") and hasattr(module, "weight"):
#             PQ = module.P @ module.Q
#             diff = PQ - module.weight.detach()
#             loss += torch.norm(diff, p='fro') ** 2
#     return loss

def reinitialize_pq_layers(trainer, r=None):
    """Reinitialize P and Q matrices for all rAFA layers in the model and clear qp_optimizer state"""
    for module in trainer.model.modules():
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

    device = 'cuda' #'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size)

    # Model
    model = BPVit(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=384,
        depth=6,
        heads=8,
        mlp_dim=384,
        dropout=0.1,
        emb_dropout=0.1,
    )

    if use_ldfa_linear:
        replace_linear(model, LDFA_Linear, rank=ldfa_rank)
    else:
        replace_linear(model, BP_Linear)
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
        qp_optimizer = optim.Adam(qp_params, lr=qp_lr, weight_decay=qp_weight_decay, betas=(0.1, 0.99))


        model_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model_optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.05,
            anneal_strategy='linear',
            div_factor=25.0,
            final_div_factor=1e3,
        )

        qp_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            qp_optimizer,
            max_lr=qp_lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.05,
            anneal_strategy='linear',
            div_factor=10.0,
            final_div_factor=1e3,
        )

        optimizers = [model_optimizer, qp_optimizer]
        schedulers = [model_scheduler, qp_scheduler]
        modify_funcs = [lambda trainer: reinitialize_pq_layers(trainer, 0.5)]
        modification_rate = 391

    else:

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer,
        #     gamma=0.95**(1/len(train_loader))
        # )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.05,
            anneal_strategy='linear',
            div_factor=25.0,
            final_div_factor=1e3,
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
