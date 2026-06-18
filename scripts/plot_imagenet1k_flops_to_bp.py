"""
Plot: FLOPs-to-threshold comparison for ImageNet-1K.

For BP      : FLOPs = Step_to_Max       × FLOPs_per_epoch  (cost to reach BP's own peak)
For LDFA    : FLOPs = Step_to_Match_BP  × FLOPs_per_epoch  (cost to first match BP's mean peak)

Step_to_Match_BP is computed by extract_training_summary.py and stored in the CSV.
Accuracy bars still show each method's own max accuracy.

Usage:
  python scripts/plot_imagenet1k_flops_to_bp.py \\
      --detailed_csv  experiment_plots/imagenet1k/training_summary_detailed.csv \\
      --bench_config  configs/benchmarking_configs/vit_benchmarking_configs_imagenet1k.yaml \\
      --output_dir    experiment_plots/imagenet1k
"""

import os
import sys
import json
import hashlib
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

# ── task map ──────────────────────────────────────────────────────────────────
# (csv_key_substring, display_label, training_config_path, use_ldfa_override)
TASK_MAP = [
    ('LDFA_multiR_setting1_imagenet1k',
     'LDFA-MultiR',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank.yaml',
     None),
    ('LDFA_192_s_imagenet1k',
     'LDFA-192',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank_192.yaml',
     None),
    ('BP_imagenet1k',
     'BP',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank.yaml',
     False),
]

PLOT_ORDER = ['BP', 'LDFA-192', 'LDFA-MultiR']


# ── model building (shared with plot_imagenet1k_flops_accuracy.py) ────────────

def replace_linear(module, new_linear_cls, **kwargs):
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
    for i, block in enumerate(model.blocks):
        rank = layer_rank_map.get(i, default_rank)
        if rank == -1:
            replace_linear(block, BP_Linear)
        else:
            replace_linear(block, LDFA_Linear, rank=rank)


def build_model(bench_cfg, method_cfg, device, dtype):
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
            layer_ranks = {int(k): int(v) for k, v in layer_ranks_raw.items()}
            replace_linear_by_layer(model, layer_ranks,
                                    default_rank=int(method_cfg.get('ldfa_rank', -1)))
        else:
            rank = int(method_cfg.get('ldfa_rank', 128))
            replace_linear(model, LDFA_Linear, rank=rank)
    return model.to(device=device, dtype=dtype)


# ── FLOPs measurement ─────────────────────────────────────────────────────────

def measure_flops(model, x, y, n_iters=3):
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True) as prof:
            for _ in range(n_iters):
                _ = model(x)
    fwd = sum(e.flops for e in prof.key_averages()
              if hasattr(e, 'flops') and e.flops is not None) / n_iters

    x_proxy = x.float().requires_grad_(True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        with_flops=True) as prof:
        for _ in range(n_iters):
            out = model(x_proxy.to(x.dtype))
            loss = (out * y).sum() if isinstance(out, torch.Tensor) else (out.logits * y).sum()
            loss.backward()
            model.zero_grad()
    bwd = sum(e.flops for e in prof.key_averages()
              if hasattr(e, 'flops') and e.flops is not None) / n_iters

    return fwd + bwd


def get_flops_per_batch(label, method_cfg_path, bench_cfg, cache_dir, use_ldfa_override=None):
    """Return GFLOPs/batch, using the shared cache from plot_imagenet1k_flops_accuracy.py."""
    with open(method_cfg_path, 'r') as f:
        method_text = f.read()
    override_str = str(use_ldfa_override)
    cache_key = hashlib.md5(
        (method_text + str(sorted(bench_cfg.items())) + override_str).encode()
    ).hexdigest()[:12]
    cache_file = os.path.join(cache_dir, f'flops_cache_{cache_key}.json')

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print(f"  [{label}] Loaded cached FLOPs: {data['flops_gflops']:.2f} GFLOPs/batch")
        return data['flops_gflops']

    print(f"  [{label}] Benchmarking FLOPs...")
    with open(method_cfg_path, 'r') as f:
        method_cfg = yaml.safe_load(f)
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

    with open(cache_file, 'w') as f:
        json.dump({'flops_gflops': flops_gflops, 'method': label}, f, indent=2)
    print(f"  [{label}] {flops_gflops:.2f} GFLOPs/batch → cached")
    return flops_gflops


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(detailed_csv, bench_config_path, output_dir,
        acc_ylim=None, flops_ylim=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(bench_config_path, 'r') as f:
        bench_cfg = yaml.safe_load(f)

    batches_per_epoch = bench_cfg['batches_per_epoch']
    bench_bs    = bench_cfg['batch_size']
    train_bs    = bench_cfg.get('train_batch_size', bench_bs)
    batch_scale = train_bs / bench_bs

    # ── 1. Load CSV and group by method ───────────────────────────────────────
    df = pd.read_csv(detailed_csv)
    for col in ['Max_Val_Acc_Top1', 'Step_to_Max', 'Step_to_Match_BP', 'BP_Threshold']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    groups = {}   # label → {'rows': [], 'config_path': str, 'use_ldfa_override': ...}
    for _, row in df.iterrows():
        for key, label, cfg_path, override in TASK_MAP:
            if key in row['Task']:
                if label not in groups:
                    groups[label] = {'rows': [], 'config_path': cfg_path,
                                     'use_ldfa_override': override}
                groups[label]['rows'].append(row)
                break

    print("=== Grouped experiments ===")
    for label, gdata in groups.items():
        print(f"  {label}: {len(gdata['rows'])} experiments")

    # ── 2. Read BP threshold from CSV ─────────────────────────────────────────
    bp_rows = groups.get('BP', {}).get('rows', [])
    if not bp_rows:
        raise RuntimeError("No BP experiments found in the CSV.")
    bp_threshold = float(bp_rows[0]['BP_Threshold'])
    print(f"\n=== BP threshold (from CSV) ===")
    print(f"  {bp_threshold:.4f}")

    # ── 3. Compute effective steps per method (directly from CSV) ─────────────
    print("\n=== Effective epochs (from CSV) ===")
    effective_steps = {}   # label → list of steps

    for label, gdata in groups.items():
        steps_list = []
        for row in gdata['rows']:
            exp_num = int(row['Exp_Number'])
            if label == 'BP':
                s = row['Step_to_Max']
            else:
                s = row['Step_to_Match_BP']

            if pd.isna(s):
                print(f"  [{label}] exp{exp_num}: Step_to_Match_BP is N/A, skipping")
                continue
            s = int(s)
            steps_list.append(s)
            col_name = 'Step_to_Max' if label == 'BP' else 'Step_to_Match_BP'
            print(f"  [{label}] exp{exp_num}: {col_name} = {s}")
        effective_steps[label] = steps_list

    # ── 4. Aggregate metrics ──────────────────────────────────────────────────
    print("\n=== Aggregated metrics ===")
    summary = {}
    for label, gdata in groups.items():
        rows  = gdata['rows']
        accs  = np.array([r['Max_Val_Acc_Top1'] for r in rows])
        steps = np.array(effective_steps.get(label, []), dtype=float)
        n_acc   = len(accs)
        n_steps = len(steps)

        summary[label] = {
            'mean_acc':    float(np.mean(accs)),
            'std_acc':     float(np.std(accs,  ddof=1) if n_acc > 1 else 0.0),
            'mean_steps':  float(np.mean(steps)) if n_steps > 0 else float('nan'),
            'std_steps':   float(np.std(steps, ddof=1) if n_steps > 1 else 0.0),
            'n_acc':       n_acc,
            'n_steps':     n_steps,
            'config_path': gdata['config_path'],
            'use_ldfa_override': gdata['use_ldfa_override'],
        }
        step_desc = ('Step_to_Max' if label == 'BP'
                     else f'First epoch ≥ BP threshold ({bp_threshold:.4f})')
        print(f"  {label}: acc={summary[label]['mean_acc']:.4f}±{summary[label]['std_acc']:.4f}  "
              f"{step_desc}={summary[label]['mean_steps']:.1f}±{summary[label]['std_steps']:.1f}  "
              f"(n={n_steps})")

    # ── 5. Benchmark FLOPs ────────────────────────────────────────────────────
    print("\n=== FLOPs benchmarking ===")
    for label, sdata in summary.items():
        flops_batch = get_flops_per_batch(
            label, sdata['config_path'], bench_cfg, output_dir,
            use_ldfa_override=sdata['use_ldfa_override']
        )
        flops_epoch_gflops = flops_batch * batch_scale * batches_per_epoch
        n = sdata['n_steps']
        sdata['flops_epoch_gflops'] = flops_epoch_gflops
        sdata['total_pflops']       = (sdata['mean_steps'] * flops_epoch_gflops) / 1e6
        sdata['total_pflops_std']   = (sdata['std_steps']  * flops_epoch_gflops) / 1e6
        sdata['total_pflops_sem']   = sdata['total_pflops_std'] / (np.sqrt(n) if n > 1 else 1.0)
        print(f"  {label}: {flops_epoch_gflops/1e3:.1f} TFLOPs/epoch → "
              f"total {sdata['total_pflops']:.1f} PFLOPs")

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

    # ── 6. Save summary CSV ───────────────────────────────────────────────────
    order = [l for l in PLOT_ORDER if l in summary] + \
            [l for l in summary   if l not in PLOT_ORDER]
    rows_out = []
    for label in order:
        s = summary[label]
        step_metric_label = ('Step_to_Max' if label == 'BP'
                             else f'First_Epoch_ge_{bp_threshold:.4f}')
        rows_out.append({
            'Method':                  label,
            'N_acc':                   s['n_acc'],
            'N_steps':                 s['n_steps'],
            'Mean_Top1_Acc':           f"{s['mean_acc']:.4f}",
            'Std_Top1_Acc':            f"{s['std_acc']:.4f}",
            'BP_threshold':            f"{bp_threshold:.4f}",
            'Step_Metric':             step_metric_label,
            'Mean_Effective_Epochs':   f"{s['mean_steps']:.1f}",
            'Std_Effective_Epochs':    f"{s['std_steps']:.1f}",
            'FLOPs_per_epoch_TFLOPs':  f"{s['flops_epoch_gflops']/1e3:.1f}",
            'Total_FLOPs_PFLOPs':      f"{s['total_pflops']:.2f}",
            'Total_FLOPs_SEM_PFLOPs':  f"{s['total_pflops_sem']:.2f}",
            'FLOPs_Saved_vs_BP_pct':   f"{s['flops_saved_pct']:+.2f}",
        })
    summary_df = pd.DataFrame(rows_out)
    csv_out = os.path.join(output_dir, 'imagenet1k_flops_to_bp_summary.csv')
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSummary CSV saved to: {csv_out}")
    print(summary_df.to_string(index=False))

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    print("\n=== Creating plot ===")
    labels     = order
    acc_vals   = [summary[l]['mean_acc']        for l in labels]
    acc_stds   = [summary[l]['std_acc']         for l in labels]
    flops_vals = [summary[l]['total_pflops']    for l in labels]
    flops_sems = [summary[l]['total_pflops_sem'] for l in labels]

    color_map  = {'BP': '#000000', 'LDFA-192': '#2171B5', 'LDFA-MultiR': '#6BAED6'}
    bar_colors = [color_map.get(l, '#4292C6') for l in labels]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.bar(labels, acc_vals, yerr=acc_stds,
            color=bar_colors, alpha=0.75, capsize=5,
            error_kw={'elinewidth': 2, 'capthick': 2})

    ax1.set_ylabel('Top-1 Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=14)

    if acc_ylim is not None:
        ax1.set_ylim([acc_ylim[0], acc_ylim[1]])
    else:
        acc_span = max(acc_vals) - min(acc_vals)
        acc_pad  = max(acc_span * 0.5, 0.01)
        ax1.set_ylim([min(acc_vals) - acc_pad, max(acc_vals) + acc_pad * 0.1])

    ax2 = ax1.twinx()
    color_line = '#B34700'
    ax2.errorbar(labels, flops_vals, yerr=flops_sems,
                 color=color_line, marker='s', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, elinewidth=1.5)
    ax2.set_ylabel('Computational Cost (PFLOPs)', fontsize=16, fontweight='bold',
                   color=color_line)
    ax2.tick_params(axis='y', labelsize=14)

    if flops_ylim is not None:
        ax2.set_ylim([flops_ylim[0], flops_ylim[1]])
    else:
        flops_span = max(flops_vals) - min(flops_vals)
        flops_pad  = max(flops_span * 0.5, max(flops_vals) * 0.05)
        ax2.set_ylim([min(flops_vals) - flops_pad, max(flops_vals) + flops_pad])

    # Annotate accuracy above each bar
    ylim = ax1.get_ylim()
    for i, (v, s) in enumerate(zip(acc_vals, acc_stds)):
        ax1.text(i, v + s + (ylim[1] - ylim[0]) * 0.01, f'{v:.2%}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['png', 'svg', 'pdf']:
        out_path = os.path.join(output_dir, f'imagenet1k_flops_to_bp.{ext}')
        plt.savefig(out_path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
    print(f"Saved: {output_dir}/imagenet1k_flops_to_bp.{{png,svg,pdf}}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot FLOPs-to-threshold for ImageNet-1K: '
                    'BP uses cost-to-own-peak; LDFA uses cost-to-match-BP-peak.')
    parser.add_argument('--detailed_csv', type=str,
                        default='experiment_plots/imagenet1k/training_summary_detailed.csv')
    parser.add_argument('--bench_config', type=str,
                        default='configs/benchmarking_configs/vit_benchmarking_configs_imagenet1k.yaml')
    parser.add_argument('--output_dir', type=str,
                        default='experiment_plots/imagenet1k')
    parser.add_argument('--acc_ylim',   type=float, nargs=2, default=None,
                        metavar=('Y_MIN', 'Y_MAX'))
    parser.add_argument('--flops_ylim', type=float, nargs=2, default=None,
                        metavar=('Y_MIN', 'Y_MAX'))
    args = parser.parse_args()

    run(args.detailed_csv, args.bench_config, args.output_dir,
        acc_ylim=args.acc_ylim, flops_ylim=args.flops_ylim)
