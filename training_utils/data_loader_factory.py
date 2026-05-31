import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# --- CIFAR10 AutoAugment Policy Classes (as provided) ---
class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)
class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)
class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)
class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)
class Rotate(object):
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)
class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))
class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, int(magnitude))
class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, int(magnitude))
class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))
class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))
class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))
class Autocontrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)
class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)
class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)
class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.array([8, 8, 7, 7, 6, 6, 5, 5, 4, 4]),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": Autocontrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]
    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img
class CIFAR10Policy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1)
        ]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# For COCO segmentation
try:
    from torchvision.datasets import CocoDetection
except ImportError:
    CocoDetection = None


def get_deterministic_class_subset(num_classes_total, num_subset_classes):
    """
    Get a deterministic subset of classes - simply the first n classes.
    This ensures the same subset is used across all experiments and avoids label remapping.
    
    Args:
        num_classes_total: Total number of classes (e.g., 100 for CIFAR-100)
        num_subset_classes: Number of classes to select (e.g., 10, 20, 50)
    
    Returns:
        List of selected class indices (0 to num_subset_classes-1)
    """
    if num_subset_classes >= num_classes_total:
        return list(range(num_classes_total))
    
    # Simply use the first n classes
    return list(range(num_subset_classes))


def filter_dataset_by_classes(dataset, selected_classes):
    """
    Filter a dataset to only include samples from selected classes.
    Since we use the first n classes, labels are already in [0, n-1] range - no remapping needed.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        selected_classes: List of class indices to keep (should be [0, 1, 2, ..., n-1])
    
    Returns:
        Subset of the dataset with filtered samples
    """
    # Find indices of samples with selected classes
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    indices = [i for i, label in enumerate(targets) if label in selected_classes]
    
    # Create and return subset - no label remapping needed since we use first n classes
    return Subset(dataset, indices)


def get_cifar10_loaders(batch_size=64, root='./data', num_workers=8):
    
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return train_loader, test_loader

def get_cifar100_loaders(batch_size=64, root='./data', num_workers=8, num_subset_classes=None):
    """
    Get CIFAR-100 data loaders, optionally with a subset of classes.
    
    Args:
        batch_size: Batch size for training
        root: Root directory for data
        num_workers: Number of workers for data loading
        num_subset_classes: If specified, uses only this many classes (e.g., 10, 20, 50).
                           Uses the same deterministic subset across experiments.
    
    Returns:
        train_loader, test_loader
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
    
    # Apply class subset filtering if specified
    if num_subset_classes is not None and num_subset_classes < 100:
        selected_classes = get_deterministic_class_subset(100, num_subset_classes)
        print(f"Using subset of {num_subset_classes} classes from CIFAR-100: {selected_classes}")
        train_set = filter_dataset_by_classes(train_set, selected_classes)
        test_set = filter_dataset_by_classes(test_set, selected_classes)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return train_loader, test_loader

def get_coco_segmentation_loaders(batch_size=64, root='./data', coco_annFile_train=None, coco_annFile_val=None, num_workers=8):
    if CocoDetection is None:
        raise ImportError('torchvision is not built with COCO support. Please install pycocotools and torchvision with COCO.')
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    train_set = CocoDetection(root=root+'/train2017', annFile=coco_annFile_train, transform=train_transform)
    test_set = CocoDetection(root=root+'/val2017', annFile=coco_annFile_val, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return train_loader, test_loader

def get_imagenet100_loaders(data_dir, batch_size=128, num_workers=16,
                            class_list_file='./data/imagenet100_classes.txt',
                            color_jitter=0.3):
    """
    Get ImageNet-100 data loaders — a 100-class subset of full ImageNet.

    Expects the standard ImageNet folder layout at data_dir:
        data_dir/train/<synset_id>/...
        data_dir/val/<synset_id>/...

    Augmentation follows standard ViT/DeiT ImageNet recipe:
      - Train: RandomResizedCrop(224), RandomHorizontalFlip, RandAugment(n=2, m=9),
               ToTensor, Normalize, RandomErasing(p=0.25)
      - Val:   Resize(256), CenterCrop(224), ToTensor, Normalize

    Args:
        data_dir:         Path to the extracted ImageNet root (with train/ and val/).
        batch_size:       Training batch size.
        num_workers:      DataLoader workers.
        class_list_file:  Text file with one synset id per line (100 lines).
    """
    # Load the 100 selected synset ids
    with open(class_list_file, 'r') as f:
        selected_classes = [line.strip() for line in f if line.strip()]
    assert len(selected_classes) == 100, f"Expected 100 classes, got {len(selected_classes)}"

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Build full datasets first so we can read their class_to_idx mapping
    full_train = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                      transform=train_transform)
    full_val   = datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
                                      transform=val_transform)

    # Filter samples to only the 100 selected classes
    selected_set = set(selected_classes)
    train_indices = [i for i, (_, lbl) in enumerate(full_train.samples)
                     if full_train.classes[lbl] in selected_set]
    val_indices   = [i for i, (_, lbl) in enumerate(full_val.samples)
                     if full_val.classes[lbl] in selected_set]

    # Remap labels 0-99 in sorted class order
    sorted_classes = sorted(selected_classes)
    train_label_remap = {full_train.class_to_idx[c]: new_lbl
                         for new_lbl, c in enumerate(sorted_classes)
                         if c in full_train.class_to_idx}
    val_label_remap   = {full_val.class_to_idx[c]: new_lbl
                         for new_lbl, c in enumerate(sorted_classes)
                         if c in full_val.class_to_idx}

    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, label_remap):
            self.dataset = dataset
            self.indices = indices
            self.label_remap = label_remap
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, lbl = self.dataset[self.indices[idx]]
            return img, self.label_remap[lbl]

    train_dataset = RemappedSubset(full_train, train_indices, train_label_remap)
    val_dataset   = RemappedSubset(full_val,   val_indices,   val_label_remap)

    print(f"ImageNet-100: {len(train_dataset)} train / {len(val_dataset)} val samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)
    return train_loader, val_loader


def get_tiny_imagenet_loaders(data_dir='./data/tiny-imagenet-200', batch_size=128, num_workers=8):
    """
    Get Tiny ImageNet data loaders.
    Tiny ImageNet: 200 classes, 64x64 images, 500 train / 50 val per class.
    Expects data_dir to have train/ and val/ subdirectories, each with per-class subfolders.
    """
    mean = [0.4802, 0.4481, 0.3975]
    std  = [0.2770, 0.2691, 0.2821]

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir, 'val'),   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return train_loader, val_loader


def get_imagenet_loaders(data_dir, batch_size=128, num_workers=16, color_jitter=0.3):
    """
    Full ImageNet-1K loader at 224×224 following the DeiT/ViT recipe.
    Mixup and CutMix are applied in the training loop via torchvision MixUp/CutMix,
    NOT here — transforms here are purely spatial/colour augmentation.

    Train: RandomResizedCrop(224, BICUBIC) + RandAugment(n=2, m=9)
           + RandomErasing(p=0.25)
    Val:   Resize(256, BICUBIC) + CenterCrop(224)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=val_transform)

    print(f"ImageNet-1K: {len(train_dataset)} train / {len(val_dataset)} val samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=4, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=4, persistent_workers=True)
    return train_loader, val_loader
