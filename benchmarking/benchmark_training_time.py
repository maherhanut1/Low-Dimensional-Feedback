"""
Wall-clock training time benchmark: fwd+bwd and bwd-only per batch.
No DataLoader, no validation — pure GPU compute time via CUDA events.

Benchmarks:
  Part 1 — ImageNet100 setting (image_size=160, BS=512)
            BP | LDFA-32 | LDFA-64 | LDFA-72 | LDFA-128
  Part 2 — Same methods × batch sizes [64, 128, 256, 512, 1024]
  Part 3 — ImageNet1K setting (image_size=224, BS=512)
            BP | LDFA-192 | LDFA-MultiR

Usage:
  python benchmarking/benchmark_training_time.py
  python benchmarking/benchmark_training_time.py --device cuda:1 --compile --n_iters 100
"""

import os
import sys
import argparse
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from timm.models.vision_transformer import VisionTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear

mpl.rcParams['font.family'] = 'serif'

# ── Architecture definitions ───────────────────────────────────────────────────

ARCH_BASE = dict(
    patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
    mlp_ratio=4.0, qkv_bias=True,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,  # no stochasticity for clean timing
)

IMAGENET100_ARCH = dict(**ARCH_BASE, image_size=224, num_classes=100)
IMAGENET1K_ARCH  = dict(**ARCH_BASE, image_size=224, num_classes=1000)

# ── Method definitions ─────────────────────────────────────────────────────────
# label, use_ldfa, uniform_rank (None = per-layer), layer_rank_map

IMAGENET100_METHODS = [
    ('BP',       False, None, None),
    ('LDFA-32',  True,  32,   None),
    ('LDFA-64',  True,  64,   None),
    ('LDFA-72',  True,  72,   None),
    ('LDFA-128', True,  128,  None),
]

IMAGENET1K_METHODS = [
    ('BP',           False, None, None),
    ('LDFA-192',     True,  192,  None),
    # MultiR (Setting 1): blocks 0-4 → r=96, 5-6 → r=128, 7-8 → r=192, 9-11 → BP (-1)
    ('LDFA-MultiR',  True,  None, {0:96, 1:96, 2:96, 3:96, 4:96,
                                   5:128, 6:128,
                                   7:192, 8:192,
                                   9:-1, 10:-1, 11:-1}),
    # MultiR2 (Setting 2 light): blocks 0-2 → r=64, 3-5 → r=96, 6-7 → r=128, 8-9 → r=192, 10-11 → BP
    ('LDFA-MultiR2', True,  None, {0:64, 1:64, 2:64,
                                   3:96, 4:96, 5:96,
                                   6:128, 7:128,
                                   8:192, 9:192,
                                   10:-1, 11:-1}),
]

BATCH_SIZES_SWEEP = [64, 128, 256, 512, 1024]
DEFAULT_BS = 512


# ── Model building ─────────────────────────────────────────────────────────────

def replace_linear(module, new_cls, **kwargs):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            kw = dict(kwargs)
            if 'rank' in kw and 'qkv' in name:
                kw['rank'] = kw['rank'] * 3
            setattr(module, name,
                    new_cls(child.in_features, child.out_features,
                            bias=(child.bias is not None), **kw))
        else:
            replace_linear(child, new_cls, **kwargs)


def replace_linear_by_layer(model, layer_rank_map, default_rank=-1):
    for i, block in enumerate(model.blocks):
        rank = layer_rank_map.get(i, default_rank)
        if rank == -1:
            replace_linear(block, BP_Linear)
        else:
            replace_linear(block, LDFA_Linear, rank=rank)


def build_model(arch, use_ldfa, uniform_rank, layer_ranks, device, dtype, do_compile):
    model = VisionTransformer(
        img_size=arch['image_size'], patch_size=arch['patch_size'],
        in_chans=arch['in_chans'], num_classes=arch['num_classes'],
        embed_dim=arch['embed_dim'], depth=arch['depth'],
        num_heads=arch['num_heads'], mlp_ratio=arch['mlp_ratio'],
        qkv_bias=arch['qkv_bias'],
        drop_rate=arch['drop_rate'], attn_drop_rate=arch['attn_drop_rate'],
        drop_path_rate=arch['drop_path_rate'],
    )

    if not use_ldfa:
        replace_linear(model, BP_Linear)
    elif layer_ranks is not None:
        replace_linear_by_layer(model, layer_ranks, default_rank=-1)
    else:
        replace_linear(model, LDFA_Linear, rank=uniform_rank)

    model = model.to(device=device, dtype=dtype)
    if do_compile:
        model = torch.compile(model)
    return model


# ── Timing ─────────────────────────────────────────────────────────────────────

def time_batch(model, arch, batch_size, device, dtype, n_warmup, n_iters):
    """
    Returns (fwd_bwd_mean_ms, fwd_bwd_std_ms, bwd_mean_ms, bwd_std_ms).
    All timing via CUDA events — no Python overhead inside the loop.
    """
    model.train()
    x = torch.randn(batch_size, arch['in_chans'], arch['image_size'], arch['image_size'],
                    device=device, dtype=dtype)
    y = torch.randn(batch_size, arch['num_classes'], device=device, dtype=dtype)

    # ── Warmup ────────────────────────────────────────────────────────────────
    for _ in range(n_warmup):
        model.zero_grad()
        loss = model(x).mul(y).sum()
        loss.backward()
    torch.cuda.synchronize(device)

    # ── fwd + bwd ─────────────────────────────────────────────────────────────
    t_fwd_bwd = []
    for _ in range(n_iters):
        model.zero_grad()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        loss = model(x).mul(y).sum()
        loss.backward()
        t1.record()
        torch.cuda.synchronize(device)
        t_fwd_bwd.append(t0.elapsed_time(t1))

    # ── bwd only (fwd computed outside timing) ────────────────────────────────
    t_bwd = []
    for _ in range(n_iters):
        model.zero_grad()
        loss = model(x).mul(y).sum()
        torch.cuda.synchronize(device)          # ensure fwd is fully done
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        loss.backward()
        t1.record()
        torch.cuda.synchronize(device)
        t_bwd.append(t0.elapsed_time(t1))

    return (float(np.mean(t_fwd_bwd)), float(np.std(t_fwd_bwd)),
            float(np.mean(t_bwd)),     float(np.std(t_bwd)))


@contextmanager
def oom_guard(label, batch_size):
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] {label} @ BS={batch_size} — skipping")
        torch.cuda.empty_cache()


# ── Plotting helpers ───────────────────────────────────────────────────────────

def bar_chart(results, title, output_path, ylabel='Time (ms)'):
    """
    results: list of (label, fwd_bwd_mean, fwd_bwd_std, bwd_mean, bwd_std)
    """
    labels = [r[0] for r in results]
    fb_mean = np.array([r[1] for r in results])
    fb_std  = np.array([r[2] for r in results])
    bw_mean = np.array([r[3] for r in results])
    bw_std  = np.array([r[4] for r in results])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    bars1 = ax.bar(x - w/2, fb_mean, w, yerr=fb_std, label='fwd+bwd',
                   color='#2171B5', alpha=0.85, capsize=4,
                   error_kw={'elinewidth': 1.5})
    bars2 = ax.bar(x + w/2, bw_mean, w, yerr=bw_std, label='bwd only',
                   color='#EF6548', alpha=0.85, capsize=4,
                   error_kw={'elinewidth': 1.5})

    # annotate values on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + fb_std[0]*0.1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + bw_std[0]*0.1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'{output_path}.{ext}',
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}.{{png,pdf}}")


def line_chart(sweep_results, title, output_path, ylabel='Time (ms)'):
    """
    sweep_results: dict label → list of (bs, fwd_bwd_mean, bwd_mean)
    Produces two sub-plots: fwd+bwd and bwd-only.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(sweep_results)))

    for (label, pts), col in zip(sweep_results.items(), colors):
        if not pts:
            continue
        bs_vals   = [p[0] for p in pts]
        fb_vals   = [p[1] for p in pts]
        fb_stds   = [p[2] for p in pts]
        bwd_vals  = [p[3] for p in pts]
        bwd_stds  = [p[4] for p in pts]
        ax1.errorbar(bs_vals, fb_vals, yerr=fb_stds, label=label,
                     marker='o', linewidth=2, capsize=3, color=col)
        ax2.errorbar(bs_vals, bwd_vals, yerr=bwd_stds, label=label,
                     marker='o', linewidth=2, capsize=3, color=col)

    for ax, subtitle in [(ax1, 'fwd + bwd'), (ax2, 'bwd only')]:
        ax.set_xlabel('Batch size', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(subtitle, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'{output_path}.{ext}',
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}.{{png,pdf}}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)
    dtype  = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []   # for final combined CSV

    def save_part_csv(rows, part_name):
        """Save a per-part CSV immediately after each part finishes."""
        if not rows:
            return
        path = os.path.join(args.output_dir, f'{part_name}.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  CSV saved: {path}")

    def bench(label, arch, use_ldfa, uniform_rank, layer_ranks, bs):
        """Build model, run timing, return result tuple, free memory."""
        model = build_model(arch, use_ldfa, uniform_rank, layer_ranks,
                            device, dtype, args.compile)
        fb_m, fb_s, bw_m, bw_s = time_batch(
            model, arch, bs, device, dtype,
            n_warmup=args.n_warmup, n_iters=args.n_iters)
        del model
        torch.cuda.empty_cache()
        return fb_m, fb_s, bw_m, bw_s

    def bench_bs_sweep(label, arch, use_ldfa, uniform_rank, layer_ranks, batch_sizes):
        """Build model once, sweep over batch sizes."""
        print(f"  Building [{label}] ...", end='', flush=True)
        model = build_model(arch, use_ldfa, uniform_rank, layer_ranks,
                            device, dtype, args.compile)
        print(" done")
        results = []
        for bs in batch_sizes:
            print(f"    BS={bs} ...", end='', flush=True)
            try:
                with oom_guard(label, bs):
                    fb_m, fb_s, bw_m, bw_s = time_batch(
                        model, arch, bs, device, dtype,
                        n_warmup=args.n_warmup, n_iters=args.n_iters)
                    results.append((bs, fb_m, fb_s, bw_m, bw_s))
                    print(f"  fwd+bwd={fb_m:.1f}±{fb_s:.1f} ms  bwd={bw_m:.1f}±{bw_s:.1f} ms")
            except Exception as e:
                print(f"  ERROR: {e}")
        del model
        torch.cuda.empty_cache()
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1 — ImageNet100, fixed BS=512
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(f"PART 1 — ImageNet100 (image_size=224, BS={DEFAULT_BS})")
    print("="*60)
    part1_results = []
    for label, use_ldfa, rank, layer_ranks in IMAGENET100_METHODS:
        print(f"  [{label}] ...", end='', flush=True)
        try:
            with oom_guard(label, DEFAULT_BS):
                fb_m, fb_s, bw_m, bw_s = bench(
                    label, IMAGENET100_ARCH, use_ldfa, rank, layer_ranks, DEFAULT_BS)
                part1_results.append((label, fb_m, fb_s, bw_m, bw_s))
                all_rows.append({
                    'part': 'imagenet100_fixed_bs',
                    'label': label, 'batch_size': DEFAULT_BS,
                    'fwd_bwd_mean_ms': fb_m, 'fwd_bwd_std_ms': fb_s,
                    'bwd_mean_ms': bw_m, 'bwd_std_ms': bw_s,
                })
                print(f"  fwd+bwd={fb_m:.1f}±{fb_s:.1f} ms  bwd={bw_m:.1f}±{bw_s:.1f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")

    bar_chart(part1_results,
              title=f'ImageNet100 · ViT-Base/16 · BS={DEFAULT_BS} · bfloat16',
              output_path=os.path.join(args.output_dir, 'part1_imagenet100_bs512'))

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2 — ImageNet100, sweep over batch sizes
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 2 — ImageNet100 · batch size sweep")
    print("="*60)
    sweep_results = {m[0]: [] for m in IMAGENET100_METHODS}
    for label, use_ldfa, rank, layer_ranks in IMAGENET100_METHODS:
        pts = bench_bs_sweep(label, IMAGENET100_ARCH, use_ldfa, rank, layer_ranks, BATCH_SIZES_SWEEP)
        sweep_results[label] = pts
        for bs, fb_m, fb_s, bw_m, bw_s in pts:
            all_rows.append({
                'part': 'imagenet100_bs_sweep',
                'label': label, 'batch_size': bs,
                'fwd_bwd_mean_ms': fb_m, 'fwd_bwd_std_ms': fb_s,
                'bwd_mean_ms': bw_m, 'bwd_std_ms': bw_s,
            })

    line_chart(sweep_results,
               title='ImageNet100 · ViT-Base/16 · batch size sweep · bfloat16',
               output_path=os.path.join(args.output_dir, 'part2_imagenet100_bs_sweep'))

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3 — ImageNet1K, fixed BS=512
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(f"PART 3 — ImageNet1K (image_size=224, BS={DEFAULT_BS})")
    print("="*60)
    part3_results = []
    for label, use_ldfa, rank, layer_ranks in IMAGENET1K_METHODS:
        print(f"  [{label}] ...", end='', flush=True)
        try:
            with oom_guard(label, DEFAULT_BS):
                fb_m, fb_s, bw_m, bw_s = bench(
                    label, IMAGENET1K_ARCH, use_ldfa, rank, layer_ranks, DEFAULT_BS)
                part3_results.append((label, fb_m, fb_s, bw_m, bw_s))
                all_rows.append({
                    'part': 'imagenet1k_fixed_bs',
                    'label': label, 'batch_size': DEFAULT_BS,
                    'fwd_bwd_mean_ms': fb_m, 'fwd_bwd_std_ms': fb_s,
                    'bwd_mean_ms': bw_m, 'bwd_std_ms': bw_s,
                })
                print(f"  fwd+bwd={fb_m:.1f}±{fb_s:.1f} ms  bwd={bw_m:.1f}±{bw_s:.1f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")

    bar_chart(part3_results,
              title=f'ImageNet1K · ViT-Base/16 · BS={DEFAULT_BS} · bfloat16',
              output_path=os.path.join(args.output_dir, 'part3_imagenet1k_bs512'))

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4 — ImageNet1K, sweep over batch sizes
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 4 — ImageNet1K · batch size sweep")
    print("="*60)
    sweep_results_1k = {m[0]: [] for m in IMAGENET1K_METHODS}
    for label, use_ldfa, rank, layer_ranks in IMAGENET1K_METHODS:
        pts = bench_bs_sweep(label, IMAGENET1K_ARCH, use_ldfa, rank, layer_ranks, BATCH_SIZES_SWEEP)
        sweep_results_1k[label] = pts
        for bs, fb_m, fb_s, bw_m, bw_s in pts:
            all_rows.append({
                'part': 'imagenet1k_bs_sweep',
                'label': label, 'batch_size': bs,
                'fwd_bwd_mean_ms': fb_m, 'fwd_bwd_std_ms': fb_s,
                'bwd_mean_ms': bw_m, 'bwd_std_ms': bw_s,
            })

    line_chart(sweep_results_1k,
               title='ImageNet1K · ViT-Base/16 · batch size sweep · bfloat16',
               output_path=os.path.join(args.output_dir, 'part4_imagenet1k_bs_sweep'))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'training_time_benchmark.csv')
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY — fwd+bwd vs bwd-only at BS=512")
    print("="*60)
    for part_label, results in [
        (f'ImageNet100 (BS={DEFAULT_BS})', part1_results),
        (f'ImageNet1K  (BS={DEFAULT_BS})', part3_results),
    ]:
        if not results:
            continue
        bp_fb = next((r[1] for r in results if r[0] == 'BP'), None)
        print(f"\n{part_label}:")
        print(f"  {'Method':<16} {'fwd+bwd (ms)':>14} {'bwd-only (ms)':>14} {'bwd/total':>10}")
        print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*10}")
        for label, fb_m, fb_s, bw_m, bw_s in results:
            ratio = bw_m / fb_m * 100 if fb_m > 0 else float('nan')
            overhead = ''
            if bp_fb is not None and label != 'BP':
                overhead = f"  (+{(fb_m - bp_fb)/bp_fb*100:.1f}% vs BP)"
            print(f"  {label:<16} {fb_m:>10.1f}±{fb_s:<3.1f}  {bw_m:>10.1f}±{bw_s:<3.1f}  "
                  f"{ratio:>8.1f}%{overhead}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark training time: fwd+bwd and bwd-only')
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='experiment_plots/training_time_benchmark')
    parser.add_argument('--n_warmup',   type=int, default=15,
                        help='Warmup iterations before timing')
    parser.add_argument('--n_iters',    type=int, default=50,
                        help='Timing iterations to average over')
    parser.add_argument('--no_compile',  action='store_true',
                        help='Disable torch.compile (compile is ON by default)')
    args = parser.parse_args()

    args.compile = not args.no_compile
    print(f"Device : {args.device}  ({torch.cuda.get_device_name(args.device)})")
    print(f"Warmup : {args.n_warmup}  Iters: {args.n_iters}")
    print(f"Compile: {args.compile}")
    print(f"Output : {args.output_dir}")

    run(args)
