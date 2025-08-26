import torch
import torch.nn as nn
import time
import argparse
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear
import torch.profiler

def benchmark_linear(in_features, out_features, rank, batch_size, seq_len, device='cpu', dtype=torch.float32, n_warmup=10, n_iters=100):
    # Custom Linear
    ldfa = LDFA_Linear(in_features, out_features, rank, update_P=False, update_Q=False).to(device=device, dtype=dtype)
    # Standard Linear
    std = BP_Linear(in_features, out_features).to(device=device, dtype=dtype)

    # Input shape: (batch_size, seq_len, in_features)
    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(batch_size, seq_len, out_features, device=device, dtype=dtype)


    # Forward pass and loss outside timing loop for LDFA
    out_ldfa = ldfa(x)
    loss_ldfa = (out_ldfa * y).sum()
    ldfa.zero_grad()
    x.grad = None
    # Warmup backward
    for _ in range(n_warmup):
        loss_ldfa.backward(retain_graph=True)
        ldfa.zero_grad()
        x.grad = None
    # Benchmark LDFA Linear backward only
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start = time.time()
    for _ in range(n_iters):
        loss_ldfa.backward(retain_graph=True)
        ldfa.zero_grad()
        x.grad = None
    torch.cuda.synchronize() if device.startswith('cuda') else None
    ldfa_time = time.time() - start

    # Forward pass and loss outside timing loop for standard Linear
    out_std = std(x)
    loss_std = (out_std * y).sum()
    std.zero_grad()
    x.grad = None
    for _ in range(n_warmup):
        loss_std.backward(retain_graph=True)
        std.zero_grad()
        x.grad = None
    # Benchmark standard Linear backward only
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start = time.time()
    for _ in range(n_iters):
        loss_std.backward(retain_graph=True)
        std.zero_grad()
        x.grad = None
    torch.cuda.synchronize() if device.startswith('cuda') else None
    std_time = time.time() - start


    # Measure actual FLOPs using torch.profiler (single backward pass)
    # LDFA backward FLOPs
    ldfa.zero_grad()
    x.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        loss_ldfa.backward()
    ldfa_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None])
    # Standard Linear backward FLOPs
    std.zero_grad()
    x.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        loss_std.backward()
    std_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None])
    print(f"LDFA Linear backward: {ldfa_time:.4f}s, {ldfa_flops/1e9:.2f} GFLOPs (actual, single backward)")
    print(f"Standard Linear backward: {std_time:.4f}s, {std_flops/1e9:.2f} GFLOPs (actual, single backward)")
    print(f"Speedup: {std_time/ldfa_time:.2f}x (LDFA vs Standard)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=1024)
    parser.add_argument('--out_features', type=int, default=1024*4)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--n_iters', type=int, default=10)
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == 'float32' else torch.float16
    benchmark_linear(
        args.in_features, args.out_features, args.rank,
        args.batch_size, args.seq_len, args.device, dtype, n_iters=args.n_iters
    )

if __name__ == '__main__':
    main()
