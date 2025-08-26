import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16, vit_h_14
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--model', type=str, default='vit_b_16', choices=['vit_b_16', 'vit_h_14'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--ranks', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--n_iters', type=int, default=3)
    args = parser.parse_args()

    # If config is provided, override args with YAML
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    dtype = torch.float32 if args.dtype == 'float32' else torch.float16
    device = args.device
    n_iters = args.n_iters

    # Load model
    if args.model == 'vit_b_16':
        model_fn = vit_b_16
    elif args.model == 'vit_h_14':
        model_fn = vit_h_14
    else:
        raise ValueError('Unknown model')

    # Baseline (BP_Linear)
    model_bp = model_fn(num_classes=args.num_classes)
    replace_linear(model_bp, BP_Linear)
    model_bp = model_bp.to(device=device, dtype=dtype)
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(args.batch_size, args.num_classes, device=device, dtype=dtype)
    print(f"Measuring BP_Linear for {args.model}...")
    fwd_bp, bwd_bp = measure_flops(model_bp, x, y, n_iters=n_iters)
    print(f"BP_Linear: Forward {fwd_bp/1e9:.2f} GFLOPs, Backward {bwd_bp/1e9:.2f} GFLOPs")

    # LDFA_Linear for different r
    r_list = args.ranks
    fwd_ldfa_list = []
    bwd_ldfa_list = []
    for r in r_list:
        model_ldfa = model_fn(num_classes=args.num_classes)
        replace_linear(model_ldfa, LDFA_Linear, rank=r)
        model_ldfa = model_ldfa.to(device=device, dtype=dtype)
        print(f"Measuring LDFA_Linear for {args.model} (rank={r})...")
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
    plt.xlabel('Rank (r)')
    plt.ylabel('Average Backward FLOPs (GFLOPs)')
    plt.title(f'Backward FLOPs vs Rank for {args.model}')
    plt.legend()
    plt.grid(True)

    # Bar plot for backward savings
    plt.subplot(1, 2, 2)
    savings = [bwd_bp / f if f > 0 else 0 for f in bwd_ldfa_list]
    plt.bar([str(r) for r in r_list], savings, color='skyblue')
    plt.axhline(1, color='r', linestyle='--', label='BP_Linear (baseline)')
    plt.xlabel('Rank (r)')
    plt.ylabel('Backward FLOPs Savings (x times less)')
    plt.title('LDFA Backward FLOPs Savings vs BP_Linear')
    for i, val in enumerate(savings):
        plt.text(i, val, f"{val:.2f}x", ha='center', va='bottom', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
