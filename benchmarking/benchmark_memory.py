"""
GPU memory footprint benchmark — 3 measurements per method:

  1. model_mb          — model weights + buffers on GPU (no data, no optimizer)
  2. fwd_bwd_mb        — peak during fwd+bwd (model + input batch + activations + gradients)
  3. full_training_mb  — peak during real training step:
                         optimizer states already resident + fwd+bwd happening simultaneously

Usage:
  python benchmarking/benchmark_memory.py
  python benchmarking/benchmark_memory.py --device cuda:1 --output_dir experiment_plots/memory_benchmark
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
from modules.opt_layers.BP_Linear  import Linear as BP_Linear

mpl.rcParams['font.family'] = 'serif'

# ── Architecture / method definitions ─────────────────────────────────────────

ARCH_BASE = dict(
    patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
    mlp_ratio=4.0, qkv_bias=True,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
)

IMAGENET100_ARCH = dict(**ARCH_BASE, image_size=224, num_classes=100)
IMAGENET1K_ARCH  = dict(**ARCH_BASE, image_size=224, num_classes=1000)

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
    ('LDFA-MultiR',  True,  None, {0:96,  1:96,  2:96,  3:96,  4:96,
                                   5:128, 6:128,
                                   7:192, 8:192,
                                   9:-1,  10:-1, 11:-1}),
    ('LDFA-MultiR2', True,  None, {0:64,  1:64,  2:64,
                                   3:96,  4:96,  5:96,
                                   6:128, 7:128,
                                   8:192, 9:192,
                                   10:-1, 11:-1}),
]

BATCH_SIZES_SWEEP = [64, 128, 256, 512]
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


def build_model(arch, use_ldfa, uniform_rank, layer_ranks, device, dtype):
    model = VisionTransformer(
        img_size=arch['image_size'],   patch_size=arch['patch_size'],
        in_chans=arch['in_chans'],     num_classes=arch['num_classes'],
        embed_dim=arch['embed_dim'],   depth=arch['depth'],
        num_heads=arch['num_heads'],   mlp_ratio=arch['mlp_ratio'],
        qkv_bias=arch['qkv_bias'],
        drop_rate=arch['drop_rate'],   attn_drop_rate=arch['attn_drop_rate'],
        drop_path_rate=arch['drop_path_rate'],
    )
    if not use_ldfa:
        replace_linear(model, BP_Linear)
    elif layer_ranks is not None:
        replace_linear_by_layer(model, layer_ranks, default_rank=-1)
    else:
        replace_linear(model, LDFA_Linear, rank=uniform_rank)
    return model.to(device=device, dtype=dtype)


def mb(n_bytes):
    return n_bytes / 1024 / 1024


# ── Memory measurement ─────────────────────────────────────────────────────────

def measure_memory(arch, use_ldfa, uniform_rank, layer_ranks, batch_size, device, dtype):
    """
    Returns exactly 3 measurements:

      model_mb         — model weights + buffers resident on GPU.
                         No data, no optimizer.

      fwd_bwd_mb       — peak VRAM during fwd+bwd with NO optimizer.
                         = model + input batch + activations + gradients.

      full_training_mb — peak VRAM during a real training step where AdamW
                         optimizer states are already resident from the previous step.
                         = model + input batch + activations + gradients + optimizer states.
    """
    torch.cuda.empty_cache()

    # ── 1. Model only ─────────────────────────────────────────────────────────
    model = build_model(arch, use_ldfa, uniform_rank, layer_ranks, device, dtype)
    model.train()
    torch.cuda.synchronize(device)
    model_mb = mb(torch.cuda.memory_reserved(device))

    # ── 2. Fwd+Bwd peak (no optimizer) ────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    x    = torch.randn(batch_size, arch['in_chans'],
                       arch['image_size'], arch['image_size'],
                       device=device, dtype=dtype)
    out  = model(x)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize(device)
    fwd_bwd_mb = mb(torch.cuda.max_memory_reserved(device))

    # Clear grads + activations
    for p in model.parameters():
        p.grad = None
    del x, out, loss
    torch.cuda.empty_cache()

    # ── 3. Full training peak (optimizer states already resident) ─────────────
    # Warm-up step: creates AdamW exp_avg + exp_avg_sq tensors on GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x    = torch.randn(batch_size, arch['in_chans'],
                       arch['image_size'], arch['image_size'],
                       device=device, dtype=dtype)
    out  = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()        # ← allocates optimizer state tensors
    optimizer.zero_grad()
    del x, out, loss
    torch.cuda.empty_cache()

    # Real measurement: optimizer states already in memory
    torch.cuda.reset_peak_memory_stats(device)
    x    = torch.randn(batch_size, arch['in_chans'],
                       arch['image_size'], arch['image_size'],
                       device=device, dtype=dtype)
    out  = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    full_training_mb = mb(torch.cuda.max_memory_reserved(device))

    del model, x, out, loss, optimizer
    torch.cuda.empty_cache()

    return {
        'model_mb':         model_mb,
        'fwd_bwd_mb':       fwd_bwd_mb,
        'full_training_mb': full_training_mb,
    }


@contextmanager
def oom_guard(label, batch_size):
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] {label} @ BS={batch_size} — skipping")
        torch.cuda.empty_cache()


# ── Plotting helpers ───────────────────────────────────────────────────────────

def bar_chart_memory(results, title, output_path):
    """results: list of (label, model_mb, fwd_bwd_mb, full_training_mb)"""
    labels     = [r[0] for r in results]
    model_mem  = np.array([r[1] for r in results])
    fwd_bwd    = np.array([r[2] for r in results])
    full_train = np.array([r[3] for r in results])

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.7), 5))

    b1 = ax.bar(x - w, model_mem,  w, label='1 · Model only',                   color='#525252', alpha=0.85)
    b2 = ax.bar(x,     fwd_bwd,    w, label='2 · Fwd+Bwd (no optimizer)',       color='#2171B5', alpha=0.85)
    b3 = ax.bar(x + w, full_train, w, label='3 · Full training (+optimizer)',   color='#EF6548', alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 50,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('GPU Memory (MB)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(axis='y', labelsize=11)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'{output_path}.{ext}',
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}.{{png,pdf}}")


def line_chart_memory(sweep_results, title, output_path):
    """sweep_results: dict  label → list of (bs, model_mb, fwd_bwd_mb, full_training_mb)"""
    subtitles = ['1 · Model only', '2 · Fwd+Bwd (no optimizer)', '3 · Full training (+optimizer)']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(sweep_results)))

    for (label, pts), col in zip(sweep_results.items(), colors):
        if not pts:
            continue
        bs_vals = [p[0] for p in pts]
        for ax, idx, subtitle in zip(axes, [1, 2, 3], subtitles):
            ax.plot(bs_vals, [p[idx] for p in pts],
                    label=label, marker='o', linewidth=2, color=col)

    for ax, subtitle in zip(axes, subtitles):
        ax.set_xlabel('Batch size', fontsize=12, fontweight='bold')
        ax.set_ylabel('GPU Memory (MB)', fontsize=12, fontweight='bold')
        ax.set_title(subtitle, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'{output_path}.{ext}',
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}.{{png,pdf}}")


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  CSV : {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def bench(label, arch, use_ldfa, uniform_rank, layer_ranks, bs, device, dtype):
    s = measure_memory(arch, use_ldfa, uniform_rank, layer_ranks, bs, device, dtype)
    print(f"  [{label:<14}]  "
          f"model={s['model_mb']:6.0f} MB  "
          f"fwd+bwd={s['fwd_bwd_mb']:7.0f} MB  "
          f"full_training={s['full_training_mb']:7.0f} MB")
    return s


def run(args):
    device = torch.device(args.device)
    dtype  = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1 — ImageNet100, fixed BS=512
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(f"PART 1 — ImageNet100  (image_size=224, BS={DEFAULT_BS})")
    print("="*60)
    part1_bar  = []
    part1_rows = []
    for label, use_ldfa, rank, layer_ranks in IMAGENET100_METHODS:
        try:
            with oom_guard(label, DEFAULT_BS):
                s = bench(label, IMAGENET100_ARCH, use_ldfa, rank, layer_ranks,
                          DEFAULT_BS, device, dtype)
                part1_bar.append((label, s['model_mb'], s['fwd_bwd_mb'], s['full_training_mb']))
                row = {'part': 'imagenet100_bs512', 'label': label, 'batch_size': DEFAULT_BS, **s}
                part1_rows.append(row); all_rows.append(row)
        except Exception as e:
            print(f"  ERROR [{label}]: {e}")

    bar_chart_memory(part1_bar,
                     title=f'ImageNet100 · ViT-Base/16 · BS={DEFAULT_BS} · bfloat16',
                     output_path=os.path.join(args.output_dir, 'part1_imagenet100_bs512'))
    save_csv(part1_rows, os.path.join(args.output_dir, 'part1_imagenet100_bs512.csv'))

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2 — ImageNet100, batch size sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 2 — ImageNet100 · batch size sweep")
    print("="*60)
    sweep100   = {m[0]: [] for m in IMAGENET100_METHODS}
    part2_rows = []
    for label, use_ldfa, rank, layer_ranks in IMAGENET100_METHODS:
        print(f"  [{label}]")
        for bs in BATCH_SIZES_SWEEP:
            print(f"    BS={bs} ...", end='', flush=True)
            try:
                with oom_guard(label, bs):
                    s = bench(label, IMAGENET100_ARCH, use_ldfa, rank, layer_ranks,
                              bs, device, dtype)
                    sweep100[label].append((bs, s['model_mb'], s['fwd_bwd_mb'], s['full_training_mb']))
                    row = {'part': 'imagenet100_bs_sweep', 'label': label, 'batch_size': bs, **s}
                    part2_rows.append(row); all_rows.append(row)
            except Exception as e:
                print(f"  ERROR [{label}@{bs}]: {e}")

    line_chart_memory(sweep100,
                      title='ImageNet100 · ViT-Base/16 · batch size sweep · bfloat16',
                      output_path=os.path.join(args.output_dir, 'part2_imagenet100_bs_sweep'))
    save_csv(part2_rows, os.path.join(args.output_dir, 'part2_imagenet100_bs_sweep.csv'))

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3 — ImageNet1K, fixed BS=512
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(f"PART 3 — ImageNet1K  (image_size=224, BS={DEFAULT_BS})")
    print("="*60)
    part3_bar  = []
    part3_rows = []
    for label, use_ldfa, rank, layer_ranks in IMAGENET1K_METHODS:
        try:
            with oom_guard(label, DEFAULT_BS):
                s = bench(label, IMAGENET1K_ARCH, use_ldfa, rank, layer_ranks,
                          DEFAULT_BS, device, dtype)
                part3_bar.append((label, s['model_mb'], s['fwd_bwd_mb'], s['full_training_mb']))
                row = {'part': 'imagenet1k_bs512', 'label': label, 'batch_size': DEFAULT_BS, **s}
                part3_rows.append(row); all_rows.append(row)
        except Exception as e:
            print(f"  ERROR [{label}]: {e}")

    bar_chart_memory(part3_bar,
                     title=f'ImageNet1K · ViT-Base/16 · BS={DEFAULT_BS} · bfloat16',
                     output_path=os.path.join(args.output_dir, 'part3_imagenet1k_bs512'))
    save_csv(part3_rows, os.path.join(args.output_dir, 'part3_imagenet1k_bs512.csv'))

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4 — ImageNet1K, batch size sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 4 — ImageNet1K · batch size sweep")
    print("="*60)
    sweep1k    = {m[0]: [] for m in IMAGENET1K_METHODS}
    part4_rows = []
    for label, use_ldfa, rank, layer_ranks in IMAGENET1K_METHODS:
        print(f"  [{label}]")
        for bs in BATCH_SIZES_SWEEP:
            print(f"    BS={bs} ...", end='', flush=True)
            try:
                with oom_guard(label, bs):
                    s = bench(label, IMAGENET1K_ARCH, use_ldfa, rank, layer_ranks,
                              bs, device, dtype)
                    sweep1k[label].append((bs, s['model_mb'], s['fwd_bwd_mb'], s['full_training_mb']))
                    row = {'part': 'imagenet1k_bs_sweep', 'label': label, 'batch_size': bs, **s}
                    part4_rows.append(row); all_rows.append(row)
            except Exception as e:
                print(f"  ERROR [{label}@{bs}]: {e}")

    line_chart_memory(sweep1k,
                      title='ImageNet1K · ViT-Base/16 · batch size sweep · bfloat16',
                      output_path=os.path.join(args.output_dir, 'part4_imagenet1k_bs_sweep'))
    save_csv(part4_rows, os.path.join(args.output_dir, 'part4_imagenet1k_bs_sweep.csv'))

    # ── Combined CSV ──────────────────────────────────────────────────────────
    all_csv = os.path.join(args.output_dir, 'memory_benchmark.csv')
    pd.DataFrame(all_rows).to_csv(all_csv, index=False)
    print(f"\nAll results → {all_csv}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"SUMMARY — BS={DEFAULT_BS}")
    print("="*60)
    for part_label, bar_data in [('ImageNet100', part1_bar), ('ImageNet1K', part3_bar)]:
        if not bar_data:
            continue
        bp_full = next((r[3] for r in bar_data if r[0] == 'BP'), None)
        print(f"\n{part_label}:")
        print(f"  {'Method':<16} {'Model (MB)':>11} {'Fwd+Bwd (MB)':>13} "
              f"{'Full training (MB)':>19} {'vs BP':>8}")
        print(f"  {'-'*16} {'-'*11} {'-'*13} {'-'*19} {'-'*8}")
        for label, m_mb, fb_mb, ft_mb in bar_data:
            vs_bp = ''
            if bp_full is not None and label != 'BP':
                vs_bp = f"{(ft_mb - bp_full) / bp_full * 100:+.1f}%"
            print(f"  {label:<16} {m_mb:>11.0f} {fb_mb:>13.0f} {ft_mb:>19.0f} {vs_bp:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str,
                        default='experiment_plots/memory_benchmark')
    args = parser.parse_args()

    print(f"Device : {args.device}  ({torch.cuda.get_device_name(args.device)})")
    print(f"Output : {args.output_dir}")
    run(args)
