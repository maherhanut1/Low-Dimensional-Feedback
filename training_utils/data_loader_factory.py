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
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# For COCO segmentation
try:
    from torchvision.datasets import CocoDetection
except ImportError:
    CocoDetection = None
    
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes  # Number of regions to cut out
        self.length = length    # Length of the square region

    def __call__(self, img):
        h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_cifar10_loaders(batch_size=64, root='./data', num_workers=8):
    
#     train_transform = transforms.Compose([dd
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16),
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

def get_cifar100_loaders(batch_size=64, root='./data', num_workers=8):
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

def get_imagenet_loaders(data_dir, batch_size=128, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(56, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
