"""
Combined script: aggregate training_summary_detailed.csv → benchmark FLOPs → plot.

Task groups and their corresponding training configs:
  BP           → configs/imagenet1k_configs/train_vit_imagenet1k_BP.yaml
  LDFA-192     → configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_192.yaml
  LDFA-MultiR  → configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank.yaml

Usage:
  python scripts/plot_imagenet1k_flops_accuracy.py \
      --detailed_csv  experiment_plots/imagenet1k/training_summary_detailed.csv \
      --bench_config  configs/benchmarking_configs/vit_benchmarking_configs_imagenet1k.yaml \
      --output_dir    experiment_plots/imagenet1k
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'serif'

# ── task → (display label, training config path, use_ldfa_override) ───────────
# use_ldfa_override=False  → BP_Linear regardless of what the config says
# use_ldfa_override=None   → respect use_ldfa_linear from the config
TASK_MAP = [
    ('LDFA_96_imagenet1k',
     'LDFA-96',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_96.yaml',
     None),    # all layers rank 96
    ('LDFA_128_imagenet1k',
     'LDFA-128',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_128.yaml',
     None),    # all layers rank 128
    ('LDFA_144_imagenet1k',
     'LDFA-144',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_144.yaml',
     None),    # all layers rank 144
    ('LDFA_multiR_setting2_imagenet1k',
        'LDFA-MultiR-2',
        'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank_setting2_light.yaml',
        None),    # MultiR Set 2 (lighter): per-layer ranks from config
    ('LDFA_multiR_setting1_imagenet1k',
        'LDFA-MultiR',
        'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank.yaml',
        None),    # use_ldfa_linear=true + per-layer ranks from config
    ('LDFA_192_s_imagenet1k',
        'LDFA-192',
        'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_192.yaml',
        None),    # use_ldfa_linear=true + all layers rank 192 from config
    ('BP_imagenet1k',
        'BP',
        'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank.yaml',  # same arch as LDFA
        False),   # override: force use_ldfa_linear=False → pure BP_Linear
]

PLOT_ORDER = ['BP', 'LDFA-MultiR', 'LDFA-MultiR-2', 'LDFA-192', 'LDFA-144', 'LDFA-128', 'LDFA-96']


# ── model building ────────────────────────────────────────────────────────────

def replace_linear(module, new_linear_cls, **kwargs):
    """Recursively replace nn.Linear with new_linear_cls (QKV layers get rank×3)."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            curr_kwargs = dict(kwargs)
            if 'rank' in curr_kwargs and 'qkv' in name:
                curr_kwargs['rank'] = curr_kwargs['rank'] * 3
            new_linear = new_linear_cls(
                child.in_features, child.out_features,
                **curr_kwargs, bias=(child.bias is not None))
            setattr(module, name, new_linear)
        else:
            replace_linear(child, new_linear_cls, **kwargs)


def replace_linear_by_layer(model, layer_rank_map, default_rank=-1):
    """Replace linear layers per ViT block using a per-layer rank map.
    rank == -1  → BP_Linear (standard backprop, no low-rank feedback)
    rank > 0    → LDFA_Linear with that rank
    """
    for i, block in enumerate(model.blocks):
        rank = layer_rank_map.get(i, default_rank)
        if rank == -1:
            replace_linear(block, BP_Linear)
        else:
            replace_linear(block, LDFA_Linear, rank=rank)


def build_model(bench_cfg, method_cfg, device, dtype):
    """Build a VisionTransformer using architecture from bench_cfg, LDFA settings from method_cfg."""
    model = VisionTransformer(
        img_size=bench_cfg['image_size'],
        patch_size=bench_cfg['patch_size'],
        in_chans=bench_cfg['in_chans'],
        num_classes=bench_cfg['num_classes'],
        embed_dim=bench_cfg['embed_dim'],
        depth=bench_cfg['depth'],
        num_heads=bench_cfg['num_heads'],
        mlp_ratio=bench_cfg['mlp_ratio'],
        qkv_bias=bench_cfg['qkv_bias'],
        drop_rate=bench_cfg.get('drop_rate', 0.0),
        attn_drop_rate=bench_cfg.get('attn_drop_rate', 0.0),
        drop_path_rate=bench_cfg.get('drop_path_rate', 0.0),
    )

    use_ldfa = method_cfg.get('use_ldfa_linear', False)
    if not use_ldfa:
        replace_linear(model, BP_Linear)
    else:
        layer_ranks_raw = method_cfg.get('ldfa_layer_ranks', None)
        if layer_ranks_raw is not None:
            # Per-layer rank map (multirank setting)
            layer_ranks = {int(k): int(v) for k, v in layer_ranks_raw.items()}
            replace_linear_by_layer(model, layer_ranks,
                                    default_rank=int(method_cfg.get('ldfa_rank', -1)))
        else:
            # Uniform rank for all layers
            rank = int(method_cfg.get('ldfa_rank', 128))
            replace_linear(model, LDFA_Linear, rank=rank)

    return model.to(device=device, dtype=dtype)


# ── FLOPs measurement ─────────────────────────────────────────────────────────

def measure_flops(model, x, y, n_iters=3):
    """Return total (fwd + bwd) FLOPs for one batch."""
    # Forward
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True) as prof:
            for _ in range(n_iters):
                _ = model(x)
    fwd = sum(e.flops for e in prof.key_averages()
              if hasattr(e, 'flops') and e.flops is not None) / n_iters

    # Backward (recompute graph each iter to avoid memory accumulation)
    x_proxy = x.float().requires_grad_(True)
    bwd = 0.0
    for _ in range(n_iters):
        # Graph is built outside the profiler, so only the backward pass is profiled
        out = model(x_proxy.to(x.dtype))
        loss = (out * y).sum() if isinstance(out, torch.Tensor) else (out.logits * y).sum()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True) as prof:
            loss.backward()
        model.zero_grad()
        bwd += sum(e.flops for e in prof.key_averages()
                   if hasattr(e, 'flops') and e.flops is not None)
    bwd /= n_iters

    return fwd + bwd  # total FLOPs per batch


def get_flops_per_batch(label, method_cfg_path, bench_cfg, use_ldfa_override=None):
    """Return fwd + bwd GFLOPs for one batch of the given method.

    method_cfg_path: training config that provides ldfa_layer_ranks / ldfa_rank
    use_ldfa_override: if not None, force use_ldfa_linear to this value (e.g. False for BP)
    """
    print(f"  [{label}] Benchmarking FLOPs...")
    with open(method_cfg_path, 'r') as f:
        method_cfg = yaml.safe_load(f)
    # Apply override (e.g. BP: force use_ldfa_linear=False regardless of config)
    if use_ldfa_override is not None:
        method_cfg['use_ldfa_linear'] = use_ldfa_override

    device = bench_cfg['device']
    dtype  = torch.bfloat16 if bench_cfg.get('dtype', 'bfloat16') == 'bfloat16' else torch.float32
    bs     = bench_cfg['batch_size']
    n_iters = bench_cfg.get('n_iters', 3)

    model = build_model(bench_cfg, method_cfg, device, dtype)
    x = torch.randn(bs, 3, bench_cfg['image_size'], bench_cfg['image_size'],
                    device=device, dtype=dtype)
    y = torch.randn(bs, bench_cfg['num_classes'], device=device, dtype=dtype)

    total_flops = measure_flops(model, x, y, n_iters=n_iters)
    flops_gflops = total_flops / 1e9
    del model
    torch.cuda.empty_cache()

    print(f"  [{label}] {flops_gflops:.2f} GFLOPs/batch")
    return flops_gflops


# ── figure ────────────────────────────────────────────────────────────────────

COLOR_MAP = {'BP': '#000000',
             'LDFA-96': '#08306B', 'LDFA-128': '#08519C', 'LDFA-144': '#2171B5',
             'LDFA-192': '#4292C6', 'LDFA-MultiR': '#6BAED6',
             'LDFA-MultiR-2': '#9ECAE1'}
COLOR_LINE = '#B34700'

# Vertical layout as fractions of the axes height: the accuracy error bars of all
# methods fall inside BAR_TOP_BAND, the FLOPs line (± s.e.m.) inside FLOPS_BAND,
# so the line stays inside the bars and clear of the accuracy error bars.
BAR_TOP_BAND = (0.72, 0.86)
FLOPS_BAND   = (0.12, 0.58)


def _band_limits(lo_val, hi_val, band):
    """Axis limits that place [lo_val, hi_val] at the given axes-fraction band."""
    span = max(hi_val - lo_val, 1e-12) / (band[1] - band[0])
    lo = lo_val - band[0] * span
    return [lo, lo + span]


def plot_flops_accuracy_figure(labels, acc_vals, acc_stds, flops_vals, flops_sems,
                               out_base, acc_ylim=None, flops_ylim=None,
                               missing_text='never\nreached BP'):
    """Bars: top-1 accuracy (± s.d.); line: total PFLOPs (± s.e.m.).

    Methods with NaN FLOPs (never reached the threshold) get no line point and
    a note inside their bar. Saves out_base.{png,svg,pdf}.
    """
    x = np.arange(len(labels))
    acc_vals, acc_stds = np.asarray(acc_vals, float), np.nan_to_num(np.asarray(acc_stds, float))
    flops_vals = np.asarray(flops_vals, float)
    flops_sems = np.nan_to_num(np.asarray(flops_sems, float))
    finite = np.isfinite(flops_vals)

    # Fixed size (same as the original 6-method figures) so panels match in the paper
    fig, ax1 = plt.subplots(figsize=(9.6, 6))

    ax1.bar(labels, acc_vals, yerr=acc_stds,
            color=[COLOR_MAP.get(l, '#4292C6') for l in labels], alpha=0.75, capsize=5,
            error_kw={'elinewidth': 2, 'capthick': 2})
    if len(labels) > 6 or max(len(l) for l in labels) > 11:
        # Too many methods for one-line names at the fixed figure width
        ax1.set_xticks(x)
        ax1.set_xticklabels([l.replace('LDFA-', 'LDFA-\n') for l in labels])
    ax1.set_ylabel('Top-1 Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_ylim(acc_ylim if acc_ylim is not None else
                 _band_limits((acc_vals - acc_stds).min(), (acc_vals + acc_stds).max(),
                              BAR_TOP_BAND))

    ax2 = ax1.twinx()
    ax2.errorbar(x[finite], flops_vals[finite], yerr=flops_sems[finite],
                 color=COLOR_LINE, marker='s', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, elinewidth=1.5)
    ax2.set_ylabel('Computational Cost (PFLOPs)', fontsize=16, fontweight='bold',
                   color=COLOR_LINE)
    ax2.tick_params(axis='y', labelsize=14)
    if flops_ylim is not None:
        ax2.set_ylim(flops_ylim)
    elif finite.any():
        ax2.set_ylim(_band_limits((flops_vals - flops_sems)[finite].min(),
                                  (flops_vals + flops_sems)[finite].max(), FLOPS_BAND))
    for i in np.where(~finite)[0]:
        lo, hi = ax2.get_ylim()
        ax2.text(i, lo + (hi - lo) * 0.03, missing_text, ha='center', va='bottom',
                 fontsize=9, color=COLOR_LINE, fontstyle='italic')

    # Accuracy value above each error bar
    acc_lim = ax1.get_ylim()
    for i, (v, s) in enumerate(zip(acc_vals, acc_stds)):
        ax1.text(i, v + s + (acc_lim[1] - acc_lim[0]) * 0.01, f'{v:.2%}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.tight_layout()
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(f'{out_base}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_base}.{{png,svg,pdf}}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(detailed_csv, bench_config_path, output_dir, acc_ylim=None, flops_ylim=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(bench_config_path, 'r') as f:
        bench_cfg = yaml.safe_load(f)

    batches_per_epoch = bench_cfg['batches_per_epoch']
    bench_bs  = bench_cfg['batch_size']
    train_bs  = bench_cfg.get('train_batch_size', bench_bs)
    batch_scale = train_bs / bench_bs  # scale profiling FLOPs to actual training batch size

    # ── 1. Load CSV and group by method ───────────────────────────────────────
    df = pd.read_csv(detailed_csv)
    for col in ['Max_Val_Acc_Top1', 'Max_Val_Acc_Top2', 'Step_to_Max']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    groups = {}  # label → {'rows': [], 'config_path': str, 'use_ldfa_override': ...}
    for _, row in df.iterrows():
        matched = False
        for key, label, cfg_path, use_ldfa_override in TASK_MAP:
            if key in row['Task']:
                if label not in groups:
                    groups[label] = {'rows': [], 'config_path': cfg_path,
                                     'use_ldfa_override': use_ldfa_override}
                groups[label]['rows'].append(row)
                matched = True
                break
        if not matched:
            print(f"  Warning: unrecognized task '{row['Task']}', skipping")

    print("=== Grouped experiments ===")
    for label, gdata in groups.items():
        print(f"  {label}: {len(gdata['rows'])} experiments")

    # ── 2. Aggregate metrics ───────────────────────────────────────────────────
    print("\n=== Aggregated metrics ===")
    summary = {}
    for label, gdata in groups.items():
        rows  = gdata['rows']
        accs  = np.array([r['Max_Val_Acc_Top1'] for r in rows])
        top2  = np.array([r['Max_Val_Acc_Top2'] for r in rows])
        steps = np.array([r['Step_to_Max'] for r in rows])
        n = len(rows)
        summary[label] = {
            'mean_acc':   float(np.mean(accs)),
            'std_acc':    float(np.std(accs,  ddof=1) if n > 1 else 0.0),
            'mean_top2':  float(np.mean(top2)),
            'std_top2':   float(np.std(top2,  ddof=1) if n > 1 else 0.0),
            'mean_steps': float(np.mean(steps)),
            'std_steps':  float(np.std(steps, ddof=1) if n > 1 else 0.0),
            'n':          n,
            'config_path': gdata['config_path'],
            'use_ldfa_override': gdata['use_ldfa_override'],
        }
        print(f"  {label}: acc={np.mean(accs):.4f}±{np.std(accs, ddof=1) if n>1 else 0:.4f}  "
              f"steps_to_max={np.mean(steps):.1f}±{np.std(steps, ddof=1) if n>1 else 0:.1f}")

    # ── 3. Benchmark FLOPs ────────────────────────────────────────────────────
    print("\n=== FLOPs benchmarking ===")
    for label, sdata in summary.items():
        flops_batch = get_flops_per_batch(
            label, sdata['config_path'], bench_cfg,
            use_ldfa_override=sdata['use_ldfa_override']
        )
        flops_epoch = flops_batch * batch_scale * batches_per_epoch  # GFLOPs/epoch
        n = sdata['n']
        sdata['flops_epoch_gflops'] = flops_epoch
        sdata['total_pflops']     = (sdata['mean_steps'] * flops_epoch) / 1e6
        sdata['total_pflops_std'] = (sdata['std_steps']  * flops_epoch) / 1e6
        sdata['total_pflops_sem'] = sdata['total_pflops_std'] / (np.sqrt(n) if n > 1 else 1.0)
        print(f"  {label}: {flops_epoch/1e3:.1f} TFLOPs/epoch  ({flops_epoch/1e6:.2f} PFLOPs/epoch) → "
              f"total {sdata['total_pflops']:.1f} PFLOPs to max")

    # Compute % FLOPs saved vs BP
    bp_pflops = summary['BP']['total_pflops']
    print("\n=== FLOPs savings vs BP ===")
    for label, sdata in summary.items():
        saved_pct = (bp_pflops - sdata['total_pflops']) / bp_pflops * 100.0
        sdata['flops_saved_pct'] = saved_pct
        if label == 'BP':
            print(f"  {label}: {sdata['total_pflops']:.1f} PFLOPs  (reference)")
        else:
            print(f"  {label}: {sdata['total_pflops']:.1f} PFLOPs  "
                  f"→ {saved_pct:+.1f}% vs BP  "
                  f"({abs(bp_pflops - sdata['total_pflops']):.1f} PFLOPs saved)")

    # ── 4. Save summary CSV ───────────────────────────────────────────────────
    order = [l for l in PLOT_ORDER if l in summary] + \
            [l for l in summary if l not in PLOT_ORDER]
    rows_out = []
    for label in order:
        s = summary[label]
        rows_out.append({
            'Method':           label,
            'N':                s['n'],
            'Mean_Top1_Acc':    f"{s['mean_acc']:.4f}",
            'Std_Top1_Acc':     f"{s['std_acc']:.4f}",
            'Mean_Top2_Acc':    f"{s['mean_top2']:.4f}",
            'Std_Top2_Acc':     f"{s['std_top2']:.4f}",
            'Mean_Steps_to_Max': f"{s['mean_steps']:.1f}",
            'Std_Steps_to_Max':  f"{s['std_steps']:.1f}",
            'FLOPs_per_epoch_TFLOPs': f"{s['flops_epoch_gflops']/1e3:.1f}",
            'FLOPs_per_epoch_PFLOPs': f"{s['flops_epoch_gflops']/1e6:.3f}",
            'Total_FLOPs_PFLOPs':     f"{s['total_pflops']:.2f}",
            'Total_FLOPs_SEM_PFLOPs': f"{s['total_pflops_sem']:.2f}",
            'FLOPs_Saved_vs_BP_pct':  f"{s['flops_saved_pct']:+.2f}",
        })
    summary_df = pd.DataFrame(rows_out)
    csv_out = os.path.join(output_dir, 'imagenet1k_flops_accuracy_summary.csv')
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSummary CSV saved to: {csv_out}")
    print(summary_df.to_string(index=False))

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    print("\n=== Creating plot ===")
    labels     = order
    acc_vals   = [summary[l]['mean_acc']       for l in labels]
    acc_stds   = [summary[l]['std_acc']        for l in labels]
    flops_vals = [summary[l]['total_pflops']   for l in labels]
    flops_sems = [summary[l]['total_pflops_sem'] for l in labels]

    plot_flops_accuracy_figure(labels, acc_vals, acc_stds, flops_vals, flops_sems,
                               os.path.join(output_dir, 'imagenet1k_flops_accuracy'),
                               acc_ylim=acc_ylim, flops_ylim=flops_ylim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate + benchmark + plot for ImageNet-1K')
    parser.add_argument('--detailed_csv',  type=str,
                        default='experiment_plots/imagenet1k/training_summary_detailed.csv')
    parser.add_argument('--bench_config',  type=str,
                        default='configs/benchmarking_configs/vit_benchmarking_configs_imagenet1k.yaml')
    parser.add_argument('--output_dir',    type=str,
                        default='experiment_plots/imagenet1k')
    parser.add_argument('--acc_ylim', type=float, nargs=2, default=None,
                        metavar=('Y_MIN', 'Y_MAX'),
                        help='Y-axis limits for accuracy in decimal (e.g. 0.75 0.82)')
    parser.add_argument('--flops_ylim', type=float, nargs=2, default=None,
                        metavar=('Y_MIN', 'Y_MAX'),
                        help='Y-axis limits for computational cost in PFLOPs (e.g. 30000 60000)')
    args = parser.parse_args()

    run(args.detailed_csv, args.bench_config, args.output_dir,
        acc_ylim=args.acc_ylim, flops_ylim=args.flops_ylim)
