"""
Aggregate per-experiment training_summary_detailed.csv into:
  - rank_accuracy_summary.csv     (mean/std of top1/top2 accuracy per rank)
  - convergence_summary.csv       (mean/std of steps to 90% of last accuracy per rank)

Task names are expected to contain the rank as:
  LDFA_<rank>_...  ->  rank = <rank>  (int)
  BP_...           ->  rank = 'BP'
"""

import argparse
import re
import os
import pandas as pd
import numpy as np


def extract_rank(task_name):
    """Return int rank for LDFA tasks, or 'BP' for BP tasks."""
    ldfa_match = re.match(r'^LDFA_(\d+)_', task_name)
    if ldfa_match:
        return int(ldfa_match.group(1))
    if task_name.upper().startswith('BP'):
        return 'BP'
    return None


def aggregate(detailed_csv, output_dir):
    df = pd.read_csv(detailed_csv)

    # Parse numeric columns
    for col in ['Max_Val_Acc_Top1', 'Max_Val_Acc_Top2',
                'Last_Epoch_Acc_Top1', 'Last_Epoch_Acc_Top2', 'Step_90pct_Last']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Rank'] = df['Task'].apply(extract_rank)
    df = df[df['Rank'].notna()]

    # ---------- accuracy summary ----------
    acc_rows = []
    for rank, grp in df.groupby('Rank', sort=False):
        acc_rows.append({
            'Rank': rank,
            'Mean_Top1_Acc': grp['Max_Val_Acc_Top1'].mean(),
            'Std_Top1_Acc':  grp['Max_Val_Acc_Top1'].std(ddof=1),
            'Mean_Top2_Acc': grp['Max_Val_Acc_Top2'].mean(),
            'Std_Top2_Acc':  grp['Max_Val_Acc_Top2'].std(ddof=1),
            'N': len(grp),
        })
    acc_df = pd.DataFrame(acc_rows)
    # Sort: BP first, then LDFA ranks descending
    acc_df['_sort'] = acc_df['Rank'].apply(lambda r: -1 if r == 'BP' else -int(r))
    acc_df = acc_df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    # Represent rank as string with .0 suffix for LDFA (matches existing plot script expectation)
    acc_df['Rank'] = acc_df['Rank'].apply(lambda r: r if r == 'BP' else f"{r}.0")

    acc_out = os.path.join(output_dir, 'rank_accuracy_summary.csv')
    acc_df.to_csv(acc_out, index=False)
    print(f"Accuracy summary saved to: {acc_out}")
    print(acc_df.to_string(index=False))

    # ---------- convergence summary ----------
    def make_convergence_csv(step_col, out_filename):
        conv_rows = []
        for rank, grp in df.groupby('Rank', sort=False):
            valid = pd.to_numeric(grp[step_col], errors='coerce').dropna()
            if len(valid) == 0:
                print(f"Warning: No {step_col} data for rank {rank}, skipping")
                continue
            conv_rows.append({
                'Rank': rank,
                'Mean_Steps_to_90pct': valid.mean(),
                'Std_Steps_to_90pct':  valid.std(ddof=1) if len(valid) > 1 else 0.0,
                'N': len(valid),
            })
        conv_df = pd.DataFrame(conv_rows)
        conv_df['_sort'] = conv_df['Rank'].apply(lambda r: -1 if r == 'BP' else -int(r))
        conv_df = conv_df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
        out = os.path.join(output_dir, out_filename)
        conv_df.to_csv(out, index=False)
        print(f"\n{out_filename} saved to: {out}")
        print(conv_df.to_string(index=False))

    make_convergence_csv('Step_90pct_Last', 'convergence_summary_90pct_last.csv')
    make_convergence_csv('Step_to_Max',     'convergence_summary_to_max.csv')

    # Keep backward-compatible default (90% of last)
    import shutil
    shutil.copy(
        os.path.join(output_dir, 'convergence_summary_90pct_last.csv'),
        os.path.join(output_dir, 'convergence_summary.csv')
    )
    print(f"\nDefault convergence_summary.csv -> 90pct_last")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate per-experiment CSV into summary CSVs')
    parser.add_argument('--detailed_csv', type=str,
                        default='experiment_plots/imagenet100/training_summary_detailed.csv',
                        help='Path to training_summary_detailed.csv')
    parser.add_argument('--output_dir', type=str,
                        default='experiment_plots/imagenet100',
                        help='Directory to save summary CSVs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    aggregate(args.detailed_csv, args.output_dir)
