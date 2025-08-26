import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16
from training_utils.trainer import Trainer
from training_utils.data_loader_factory import get_data_loaders

# Simple segmentation head for ViT backbone

# Modern decoder head: progressive upsampling with conv layers (inspired by SETR/Segmenter)
class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    def forward(self, x):
        return self.decode(x)

class ViT_Segmentation(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = vit_b_16(weights=None)  # No pretraining
        self.decoder = SegmentationDecoder(self.backbone.hidden_dim, num_classes)

    def forward(self, x):
        # Patchify and run transformer
        features = self.backbone._process_input(x)  # (B, 3, H, W) -> (B, num_patches, hidden_dim)
        features = self.backbone.encoder(self.backbone.conv_proj(x))  # (B, hidden_dim, H/16, W/16)
        # Remove class token if present
        if hasattr(self.backbone, 'cls_token'):
            features = features[:, 1:, ...]
        # Reshape to (B, C, H, W)
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return self.decoder(features)

# Example metric for segmentation: mean IoU (placeholder, replace with your own)
def mean_iou_metric(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    intersection = ((preds == targets) & (targets > 0)).sum().item()
    union = ((preds > 0) | (targets > 0)).sum().item()
    miou = intersection / union if union > 0 else 0.0
    return miou, 'mean_iou'

def main():
    dataset = 'coco'
    batch_size = 8
    num_iterations = 10000
    evaluation_iterations = 500
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 21  # adjust for your COCO setup

    train_loader, test_loader = get_data_loaders(dataset, batch_size=batch_size, task='segmentation')

    model = ViT_Segmentation(num_classes=num_classes).to(device)

    loss_fns = [(nn.CrossEntropyLoss(), 1.0)]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedulers = []
    metrics = [mean_iou_metric]

    name = 'vit_segmentation_scratch'
    checkpoint_dir = f'artifacts/training_checkpoints/coco/vit_segmentation/{name}'
    log_dir = f'artifacts/training_checkpoints/coco/vit_segmentation/{name}/logs'

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
