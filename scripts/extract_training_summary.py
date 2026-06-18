import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import csv
from datetime import datetime
import argparse


def is_ldfa_task(task_name):
    """Check if task is an LDFA task"""
    # Search anywhere in the task name, not just at the start
    pattern = r'[Ll][Dd][Ff][Aa]_(\d+)'
    return bool(re.search(pattern, task_name))


def extract_training_summary_to_csv(experiments, output_csv='training_summary.csv', bp_pattern='BP'):
    """
    Extract training summary information to CSV.

    Two-pass approach:
      Pass 1 – load every experiment's accuracy curve and compute per-exp metrics.
      After pass 1 – compute BP threshold = mean(max_acc) across BP experiments.
      Pass 2 – for each experiment find the first epoch where acc >= BP threshold
               and store it as Step_to_Match_BP.

    Args:
        experiments: Dictionary of experiments (from parse_experiment_folders)
        output_csv:  Output CSV file path
        bp_pattern:  Substring that identifies BP tasks (default 'BP')
    """

    # ── Pass 1: load all data ─────────────────────────────────────────────────
    raw_records = []   # list of dicts: 'fields' (CSV row) + 'acc_steps'/'acc_values'

    for task_name, exp_list in experiments.items():
        for exp_num, exp_folder in exp_list:
            try:
                data = load_tensorboard_data(
                    str(exp_folder),
                    ['eval/metric_accuracy', 'eval/metric_top2_accuracy']
                )

                acc_data  = data.get('eval/metric_accuracy', {})
                top2_data = data.get('eval/metric_top2_accuracy', {})

                if not acc_data.get('values') or not acc_data.get('steps'):
                    print(f"No validation data found for {task_name}_exp_{exp_num}")
                    continue

                acc_steps  = acc_data['steps']
                acc_values = acc_data['values']

                max_acc_idx  = np.argmax(acc_values)
                max_acc      = acc_values[max_acc_idx]
                max_acc_step = acc_steps[max_acc_idx]
                last_acc     = acc_values[-1]
                last_step    = acc_steps[-1]

                # Top-2 accuracy
                if top2_data.get('values'):
                    top2_values   = top2_data['values']
                    top2_steps    = top2_data['steps']
                    max_top2_idx  = np.argmax(top2_values)
                    max_top2      = top2_values[max_top2_idx]
                    max_top2_step = top2_steps[max_top2_idx]
                    last_top2     = top2_values[-1]
                else:
                    max_top2 = max_top2_step = last_top2 = 'N/A'

                # First epoch to reach own maximum
                step_to_max = None
                for s, v in zip(acc_steps, acc_values):
                    if v >= max_acc:
                        step_to_max = s
                        break

                # First epoch to reach 90 % of final accuracy
                step_90pct_last = None
                for s, v in zip(acc_steps, acc_values):
                    if v >= 0.9 * last_acc:
                        step_90pct_last = s
                        break

                # Wall-clock time of last logged event
                ea = EventAccumulator(str(exp_folder) + '/logs')
                ea.Reload()
                last_time = 'N/A'
                if 'eval/metric_accuracy' in ea.scalars.Keys():
                    events = ea.scalars.Items('eval/metric_accuracy')
                    if events:
                        last_time = datetime.fromtimestamp(
                            events[-1].wall_time).strftime('%Y-%m-%d %H:%M:%S')

                raw_records.append({
                    'fields': {
                        'Task':                  task_name,
                        'Exp_Number':            exp_num,
                        'Max_Val_Acc_Top1':      f"{max_acc:.4f}",
                        'Max_Val_Acc_Top1_Step': max_acc_step,
                        'Max_Val_Acc_Top2':      f"{max_top2:.4f}" if max_top2 != 'N/A' else 'N/A',
                        'Max_Val_Acc_Top2_Step': max_top2_step,
                        'Last_Epoch_Acc_Top1':   f"{last_acc:.4f}",
                        'Last_Epoch_Acc_Top2':   f"{last_top2:.4f}" if last_top2 != 'N/A' else 'N/A',
                        'Last_Epoch_Step':       last_step,
                        'Last_Epoch_Time':       last_time,
                        'Step_to_Max':           step_to_max if step_to_max is not None else 'N/A',
                        'Step_90pct_Last':       step_90pct_last if step_90pct_last is not None else 'N/A',
                    },
                    'acc_steps':  acc_steps,
                    'acc_values': acc_values,
                    'max_acc':    max_acc,
                })

            except Exception as e:
                print(f"Error processing {task_name}_exp_{exp_num}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # ── Compute BP threshold ──────────────────────────────────────────────────
    bp_accs = [rec['max_acc'] for rec in raw_records
               if bp_pattern in rec['fields']['Task']]
    if bp_accs:
        bp_threshold = float(np.mean(bp_accs))
        print(f"\nBP threshold (mean of {len(bp_accs)} BP experiments): {bp_threshold:.4f}")
    else:
        bp_threshold = None
        print("\nWarning: No BP experiments found — Step_to_Match_BP will be N/A for all rows")

    # ── Pass 2: compute Step_to_Match_BP and assemble final rows ─────────────
    summary_data = []
    for rec in raw_records:
        row = dict(rec['fields'])

        if bp_threshold is not None:
            step_to_match_bp = None
            for s, v in zip(rec['acc_steps'], rec['acc_values']):
                if v >= bp_threshold:
                    step_to_match_bp = s
                    break
            row['BP_Threshold']     = f"{bp_threshold:.4f}"
            row['Step_to_Match_BP'] = step_to_match_bp if step_to_match_bp is not None else 'N/A'
        else:
            row['BP_Threshold']     = 'N/A'
            row['Step_to_Match_BP'] = 'N/A'

        task = rec['fields']['Task']
        print(f"  {task}_exp{rec['fields']['Exp_Number']}: "
              f"Step_to_Max={row['Step_to_Max']}  "
              f"Step_to_Match_BP={row['Step_to_Match_BP']}")
        summary_data.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if summary_data:
        fieldnames = [
            'Task', 'Exp_Number',
            'Max_Val_Acc_Top1', 'Max_Val_Acc_Top1_Step',
            'Max_Val_Acc_Top2', 'Max_Val_Acc_Top2_Step',
            'Last_Epoch_Acc_Top1', 'Last_Epoch_Acc_Top2',
            'Last_Epoch_Step', 'Last_Epoch_Time',
            'Step_to_Max', 'Step_90pct_Last',
            'BP_Threshold', 'Step_to_Match_BP',
        ]
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\nTraining summary saved to {output_csv}")
    else:
        print("No data to write to CSV")


def load_tensorboard_data(log_dir, metrics=['eval/metric_accuracy', 'eval/metric_top2_accuracy', 'eval/total_loss']):
    """Load TensorBoard data for specified metrics."""
    ea = EventAccumulator(log_dir + '/logs')
    ea.Reload()
    
    data = {}
    for metric in metrics:
        if metric in ea.scalars.Keys():
            scalar_events = ea.scalars.Items(metric)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[metric] = {'steps': steps, 'values': values}
        else:
            print(f"Warning: Metric '{metric}' not found in {log_dir}")
            data[metric] = {'steps': [], 'values': []}
    
    return data


def parse_experiment_folders(base_dir, ignore_patterns=None):
    """Parse experiment folders and group by task name.
    
    Args:
        base_dir: Base directory containing experiment folders
        ignore_patterns: List of substrings; folders containing any of these will be skipped
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    if ignore_patterns is None:
        ignore_patterns = []

    experiments = defaultdict(list)
    
    # Support both _exp_N and _expN naming conventions
    pattern = re.compile(r'^(.+?)_exp_?(\d+)$')
    
    for folder in base_path.iterdir():
        if folder.is_dir():
            # Skip folders matching any ignore pattern
            if any(pat in folder.name for pat in ignore_patterns):
                print(f"Ignoring folder (matches ignore pattern): {folder.name}")
                continue

            match = pattern.match(folder.name)
            if match:
                task_name = match.group(1)
                exp_number = int(match.group(2))
                experiments[task_name].append((exp_number, folder))
            else:
                print(f"Skipping folder with unexpected format: {folder.name}")
    
    # Sort experiments by experiment number
    for task_name in experiments:
        experiments[task_name].sort(key=lambda x: x[0])
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Extract training summary from TensorBoard logs')
    parser.add_argument('--exp_dir', type=str, 
                        default='artifacts/training_checkpoints/cifar10/vit_b_16',
                        help='Path to experiments directory')
    parser.add_argument('--output_dir', type=str, 
                        default='experiment_plots',
                        help='Directory to save output CSV')
    parser.add_argument('--ignore', type=str, nargs='*', default=[],
                        help='Substrings to ignore in folder names (e.g. no_X3)')
    parser.add_argument('--bp_pattern', type=str, default='BP',
                        help='Substring that identifies BP task folders (default: BP)')
    args = parser.parse_args()
    
    # Configuration
    base_dir = args.exp_dir
    save_dir = args.output_dir
    output_csv = os.path.join(save_dir, "training_summary_detailed.csv")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading experiments from: {base_dir}")
    
    # Parse experiment folders
    experiments = parse_experiment_folders(base_dir, ignore_patterns=args.ignore)
    
    print(f"Found {len(experiments)} different tasks:")
    for task_name, exp_list in experiments.items():
        print(f"  {task_name}: {len(exp_list)} experiments")
    
    # Extract training summary to CSV
    extract_training_summary_to_csv(experiments, output_csv, bp_pattern=args.bp_pattern)
    
    print(f"\nDone! Summary saved to: {output_csv}")


if __name__ == '__main__':
    main()
