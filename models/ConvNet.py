import torch.nn as nn
import torch

# from models.layers.rAFA_conv import Conv2d as AFAConv

class CIFAR10CNNBP(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNNBP, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # [Batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [Batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),        # [Batch, 64, 16, 16]

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [Batch, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# [Batch, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),        # [Batch, 128, 8, 8]

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# [Batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# [Batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),        # [Batch, 256, 4, 4]

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# [Batch, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# [Batch, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # No MaxPool here to prevent reducing spatial dimensions too much
        )

        # Adaptive pooling to ensure the output is of size [Batch, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)            # Feature extraction
        x = self.avgpool(x)             # Global average pooling
        x = torch.flatten(x, 1)         # Flatten tensor
        x = self.classifier(x)          # Classification
        return x