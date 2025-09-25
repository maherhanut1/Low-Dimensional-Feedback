import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import vit_b_16, ViT_B_16_Weights
from einops import rearrange
from einops.layers.torch import Rearrange


import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # --- MODIFICATION ---
        # Combine Q, K, V projections into a single linear layer for efficiency.
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # --- MODIFICATION ---
        # Project to q, k, v all at once and then split.
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class BPVit(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (class token) or mean (mean pooling)'

        # --- MODIFICATION ---
        # Removed redundant LayerNorm layers. The first block in the transformer
        # will apply LayerNorm to the patch embeddings.
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # --- MODIFICATION ---
    # Removed unused 'gt' argument from the forward pass signature.
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Correctly repeat the class token for the batch
        cls_tokens = self.cls_token.repeat(b, 1, 1)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


################### VIT_16_B based ################


class ViTForCifar10(nn.Module):
    """
    A wrapper for the ViT model, modified for CIFAR-10.
    This version assumes input tensors are ALREADY preprocessed.
    """
    def __init__(self, num_classes=10, in_channels=3, image_size=32, patch_size=4):
        super().__init__()
        
        # --- 1. Load Pre-trained ViT Model ---
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # --- 2. Update All Necessary Internal Model Attributes ---
        
        # a) THE DEFINITIVE FIX: Update the patch_size attribute 🔧
        self.vit.patch_size = patch_size
        
        # b) Update the image size attribute
        self.vit.image_size = image_size
        
        # c) Calculate the new number of patches and sequence length
        num_patches = (image_size // patch_size) ** 2
        seq_length = num_patches + 1
        
        # d) Update the encoder's sequence length attribute
        self.vit.encoder.seq_length = seq_length
        
        # --- 3. Modify the Model Architecture ---
        hidden_dim = self.vit.hidden_dim

        # a) Modify the Patch Embedding layer to match the new patch size
        self.vit.conv_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # b) Modify the Positional Embeddings to match the new sequence length
        self.vit.encoder.pos_embedding = nn.Parameter(
            torch.randn(1, seq_length, hidden_dim)
        )

        # c) Modify the Classifier Head for the new number of classes
        self.vit.heads.head = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on a preprocessed tensor.
        
        Args:
            x (torch.Tensor): A preprocessed tensor of shape (B, 3, 32, 32).
        
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.vit(x)