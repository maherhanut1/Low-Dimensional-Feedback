"""
Plot: FLOPs-to-threshold comparison for ImageNet-1K.

For BP      : FLOPs = (Step_to_Max + 1)       × FLOPs_per_epoch  (cost to reach BP's own peak)
For LDFA    : FLOPs = (Step_to_Match_BP + 1)  × FLOPs_per_epoch  (cost to first match BP's mean peak)
(step s is logged after s + 1 completed epochs)

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
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# get_flops_per_batch counts FLOPs op by op (OpFlopCounter), not with torch.profiler, which reports 0
# for fused attention, layer norm and GELU (see measure_flops in plot_imagenet1k_flops_accuracy.py)
from plot_imagenet1k_flops_accuracy import (plot_flops_accuracy_figure, get_flops_per_batch,
                                            REFACTORIZATIONS_PER_EPOCH)

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'serif'

# ── task map ──────────────────────────────────────────────────────────────────
# (csv_key_substring, display_label, training_config_path, use_ldfa_override)
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
     None),
    ('LDFA_192_s_imagenet1k',
     'LDFA-192',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_192.yaml',
     None),
    ('BP_imagenet1k',
     'BP',
     'configs/imagenet1k_configs/train_vit_imagenet1k_LDFA_multirank.yaml',
     False),
]

PLOT_ORDER = ['BP', 'LDFA-MultiR', 'LDFA-MultiR-2', 'LDFA-192', 'LDFA-144', 'LDFA-128', 'LDFA-96']


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
                print(f"  [{label}] exp{exp_num}: Step_to_Match_BP is N/A "
                      f"(never reached BP threshold), skipping")
                continue
            s = int(s)
            # TensorBoard step s is logged after s + 1 completed epochs (Trainer.train logs the 0-based epoch index)
            steps_list.append(s + 1)
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
        flops_batch, opt_step, refactor = get_flops_per_batch(
            label, sdata['config_path'], bench_cfg,
            use_ldfa_override=sdata['use_ldfa_override']
        )
        flops_epoch_gflops = (flops_batch * batch_scale * batches_per_epoch    # forward + backward
                              + opt_step * batches_per_epoch                   # one optimizer step per batch
                              + refactor * REFACTORIZATIONS_PER_EPOCH)         # Q,P SVD refactorizations
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
    # Methods that never reach the BP threshold have no cost; leave them out of the plot
    labels     = [l for l in order if np.isfinite(summary[l]['total_pflops'])]
    acc_vals   = [summary[l]['mean_acc']        for l in labels]
    acc_stds   = [summary[l]['std_acc']         for l in labels]
    flops_vals = [summary[l]['total_pflops']    for l in labels]
    flops_sems = [summary[l]['total_pflops_sem'] for l in labels]

    plot_flops_accuracy_figure(labels, acc_vals, acc_stds, flops_vals, flops_sems,
                               os.path.join(output_dir, 'imagenet1k_flops_to_bp'),
                               acc_ylim=acc_ylim, flops_ylim=flops_ylim)


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
