import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
import torch.nn as nn

def measure_backward_flops(layer, x, y):
    out = layer(x)
    loss = (out * y).sum()
    layer.zero_grad()
    x.grad = None
    n_iters = 10  # You can make this configurable if desired
    total_flops = 0
    for _ in range(n_iters):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            loss.backward(retain_graph=True)
        flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None])
        total_flops += flops
        layer.zero_grad()
        x.grad = None
    avg_flops = total_flops / n_iters
    return avg_flops, n_iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=1204)
    parser.add_argument('--out_features', type=int, default=1204)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--r_list', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float32')
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == 'float32' else torch.float16
    device = args.device
    in_features = args.in_features
    out_features = args.out_features
    batch_size = args.batch_size
    seq_len = args.seq_len
    r_list = args.r_list

    # Input shape: (batch_size, seq_len, in_features)
    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(batch_size, seq_len, out_features, device=device, dtype=dtype)

    # Standard Linear baseline
    std = nn.Linear(in_features, out_features).to(device=device, dtype=dtype)
    std_flops, n_iters = measure_backward_flops(std, x, y)
    print(f"Standard Linear average backward FLOPs: {std_flops/1e9:.2f} GFLOPs (over {n_iters} iters)")

    # LDFA Linear for different r
    ldfa_flops_list = []
    for r in r_list:
        ldfa = LDFA_Linear(in_features, out_features, r, update_P=True, update_Q=True).to(device=device, dtype=dtype)
        flops, _ = measure_backward_flops(ldfa, x, y)
        ldfa_flops_list.append(flops)
        print(f"LDFA Linear (rank={r}) average backward FLOPs: {flops/1e9:.2f} GFLOPs (over {n_iters} iters)")

    # Plot line plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.axhline(std_flops/1e9, color='r', linestyle='--', label='Standard Linear')
    plt.plot(r_list, [f/1e9 for f in ldfa_flops_list], marker='o', label='LDFA Linear')
    plt.xlabel('Rank (r)')
    plt.ylabel(f'Average Backward FLOPs (GFLOPs)')
    plt.title('Average Backward FLOPs: Standard Linear vs LDFA Linear')
    plt.legend()
    plt.grid(True)

    # Bar plot for savings
    plt.subplot(1, 2, 2)
    savings = [std_flops / f if f > 0 else 0 for f in ldfa_flops_list]
    plt.bar([str(r) for r in r_list], savings, color='skyblue')
    plt.axhline(1, color='r', linestyle='--', label='Standard Linear (baseline)')
    plt.xlabel('Rank (r)')
    plt.ylabel('FLOPs Savings (x times less)')
    plt.title('LDFA FLOPs Savings vs Standard Linear')
    for i, val in enumerate(savings):
        plt.text(i, val, f"{val:.2f}x", ha='center', va='bottom', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
