import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from timm.models.vision_transformer import VisionTransformer
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
import types

# Helper to recursively replace all nn.Linear with custom linear

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

def measure_flops(model, x, y, n_iters=5):
    # Forward FLOPs
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        for _ in range(n_iters):
            out = model(x)
    fwd_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None]) / n_iters
    # Backward FLOPs
    out = model(x)
    loss = (out * y).sum() if isinstance(out, torch.Tensor) else (out.logits * y).sum()
    model.zero_grad()
    x.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        for _ in range(n_iters):
            loss.backward(retain_graph=True)
    bwd_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None]) / n_iters
    return fwd_flops, bwd_flops

def main():

    import yaml
    import os
    
    parser = argparse.ArgumentParser(description='Benchmark ViT FLOPs with LDFA')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load parameters from config
    batch_size = config.get('batch_size', 32)
    image_size = config.get('image_size', 32)
    num_classes = config.get('num_classes', 10)
    ranks = config.get('ranks', [4, 8, 16, 32, 64, 128, 256, 384])
    device = config.get('device', 'cpu')
    dtype_str = config.get('dtype', 'float32')
    n_iters = config.get('n_iters', 10)
    
    # Load model parameters
    patch_size = config.get('patch_size', 4)
    in_chans = config.get('in_chans', 3)
    embed_dim = config.get('embed_dim', 384)
    depth = config.get('depth', 8)
    num_heads = config.get('num_heads', 8)
    mlp_ratio = config.get('mlp_ratio', 2.0)
    qkv_bias = config.get('qkv_bias', True)
    drop_rate = config.get('drop_rate', 0.1)
    attn_drop_rate = config.get('attn_drop_rate', 0.1)
    drop_path_rate = config.get('drop_path_rate', 0.1)

    dtype = torch.float32 if dtype_str == 'float32' else torch.float16

    # Baseline (BP_Linear)
    model_bp = VisionTransformer(img_size=image_size,
                              patch_size=patch_size,
                              in_chans=in_chans,
                              num_classes=num_classes,
                              embed_dim=embed_dim,
                              depth=depth,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              drop_rate=drop_rate,
                              attn_drop_rate=attn_drop_rate,
                              drop_path_rate=drop_path_rate)
    
    replace_linear(model_bp, BP_Linear)
    model_bp = model_bp.to(device=device, dtype=dtype)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(batch_size, num_classes, device=device, dtype=dtype)
    print(f"Measuring BP_Linear...")
    fwd_bp, bwd_bp = measure_flops(model_bp, x, y, n_iters=n_iters)
    print(f"BP_Linear: Forward {fwd_bp/1e9:.2f} GFLOPs, Backward {bwd_bp/1e9:.2f} GFLOPs")

    # LDFA_Linear for different r
    r_list = ranks
    fwd_ldfa_list = []
    bwd_ldfa_list = []
    for r in r_list:
        model_ldfa = VisionTransformer(img_size=image_size,
                              patch_size=patch_size,
                              in_chans=in_chans,
                              num_classes=num_classes,
                              embed_dim=embed_dim,
                              depth=depth,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              drop_rate=drop_rate,
                              attn_drop_rate=attn_drop_rate,
                              drop_path_rate=drop_path_rate)
        replace_linear(model_ldfa, LDFA_Linear, rank=r)
        model_ldfa = model_ldfa.to(device=device, dtype=dtype)
        print(f"Measuring LDFA_Linear (rank={r})...")
        fwd_ldfa, bwd_ldfa = measure_flops(model_ldfa, x, y, n_iters=n_iters)
        fwd_ldfa_list.append(fwd_ldfa)
        bwd_ldfa_list.append(bwd_ldfa)
        print(f"LDFA_Linear (r={r}): Forward {fwd_ldfa/1e9:.2f} GFLOPs, Backward {bwd_ldfa/1e9:.2f} GFLOPs")

    # Plot
    import numpy as np
    plt.figure(figsize=(10, 4))
    # Only plot backward FLOPs
    plt.subplot(1, 2, 1)
    plt.axhline(bwd_bp/1e9, color='g', linestyle='--', label='BP_Linear Backward')
    plt.plot(r_list, [f/1e9 for f in bwd_ldfa_list], marker='s', label='LDFA_Linear Backward')
    plt.xlabel('Rank (r)', fontweight='bold')
    plt.ylabel('Average Backward FLOPs (GFLOPs)', fontweight='bold')
    plt.title('Backward FLOPs vs Rank', fontweight='bold')
    plt.legend()
    plt.grid(True)

    # Bar plot for backward savings
    plt.subplot(1, 2, 2)
    savings = [f / bwd_bp if bwd_bp > 0 else 0 for f in bwd_ldfa_list]
    plt.bar([str(r) for r in r_list], savings, color='skyblue')
    plt.axhline(1, color='r', linestyle='--', label='BP_Linear (baseline)')
    plt.xlabel('Rank (r)', fontweight='bold')
    plt.ylabel('Backward FLOPs Ratio', fontweight='bold')
    plt.title('LDFA Backward FLOPs Ratio vs BP_Linear', fontweight='bold')
    for i, val in enumerate(savings):
        plt.text(i, val, f"{val:.2f}x", ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('vit_flops_benchmark.pdf')
    plt.savefig('vit_flops_benchmark.png')
    plt.show()

if __name__ == '__main__':
    main()
