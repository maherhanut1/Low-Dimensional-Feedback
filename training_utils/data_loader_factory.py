import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# For COCO segmentation
try:
    from torchvision.datasets import CocoDetection
except ImportError:
    CocoDetection = None

def get_data_loaders(dataset_name: str, batch_size: int = 64, root: str = './data', train_transform=None, test_transform=None, coco_annFile_train=None, coco_annFile_val=None, num_workers: int = 8):
    """
    Returns train_loader, test_loader for the specified dataset.
    dataset_name: 'cifar10', 'cifar100', or 'coco_segmentation'
    """
    if dataset_name.lower() == 'cifar10':
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, prefetch_factor=2
        )
        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=num_workers,
            pin_memory=True, prefetch_factor=2
        )
        return train_loader, test_loader

    elif dataset_name.lower() == 'cifar100':
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, prefetch_factor=2
        )
        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=num_workers,
            pin_memory=True, prefetch_factor=2
        )
        return train_loader, test_loader

    elif dataset_name.lower() == 'coco_segmentation':
        if CocoDetection is None:
            raise ImportError('torchvision is not built with COCO support. Please install pycocotools and torchvision with COCO.')
        if coco_annFile_train is None or coco_annFile_val is None:
            raise ValueError('For COCO, coco_annFile_train and coco_annFile_val must be provided.')
        if train_transform is None:
            train_transform = transforms.ToTensor()
        if test_transform is None:
            test_transform = transforms.ToTensor()
        train_set = CocoDetection(root=root+'/train2017', annFile=coco_annFile_train, transform=train_transform)
        test_set = CocoDetection(root=root+'/val2017', annFile=coco_annFile_val, transform=test_transform)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, prefetch_factor=2
        )
        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=num_workers,
            pin_memory=True, prefetch_factor=2
        )
        return train_loader, test_loader

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
