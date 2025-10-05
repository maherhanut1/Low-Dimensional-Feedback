import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
from pathlib import Path

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

def parse_experiment_folders(base_dir):
    """Parse experiment folders and group by task name."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    experiments = defaultdict(list)
    
    # Pattern to match task_name_exp_number
    pattern = re.compile(r'^(.+)_exp_(\d+)$')
    
    for folder in base_path.iterdir():
        if folder.is_dir():
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

def interpolate_data(steps_list, values_list, target_steps):
    """Interpolate data to common step points."""
    interpolated_values = []
    
    for steps, values in zip(steps_list, values_list):
        if len(steps) == 0 or len(values) == 0:
            interpolated_values.append(np.full(len(target_steps), np.nan))
            continue
            
        # Interpolate to target steps
        interp_values = np.interp(target_steps, steps, values)
        interpolated_values.append(interp_values)
    
    return np.array(interpolated_values)

def average_experiments(experiments_data):
    """Average data across experiments for each task."""
    averaged_data = {}
    
    for task_name, exp_data_list in experiments_data.items():
        if not exp_data_list:
            continue
            
        print(f"Processing task: {task_name} ({len(exp_data_list)} experiments)")
        
        task_averages = {}
        
        # Get all metrics from first experiment
        first_exp = exp_data_list[0]
        metrics = first_exp.keys()
        
        for metric in metrics:
            # Collect all steps and values for this metric across experiments
            all_steps = []
            all_values = []
            
            for exp_data in exp_data_list:
                if metric in exp_data and exp_data[metric]['steps']:
                    all_steps.extend(exp_data[metric]['steps'])
                    
            if not all_steps:
                continue
                
            # Create common step points
            min_step = min(all_steps)
            max_step = max(all_steps)
            target_steps = np.linspace(min_step, max_step, 100)
            
            # Collect values for interpolation
            steps_list = []
            values_list = []
            
            for exp_data in exp_data_list:
                if metric in exp_data:
                    steps_list.append(exp_data[metric]['steps'])
                    values_list.append(exp_data[metric]['values'])
            
            # Interpolate all experiments to common steps
            interpolated_values = interpolate_data(steps_list, values_list, target_steps)
            
            # Calculate mean and std
            mean_values = np.nanmean(interpolated_values, axis=0)
            std_values = np.nanstd(interpolated_values, axis=0)
            
            task_averages[metric] = {
                'steps': target_steps,
                'mean': mean_values,
                'std': std_values,
                'num_experiments': len(exp_data_list)
            }
        
        averaged_data[task_name] = task_averages
    
    return averaged_data

def plot_results(averaged_data, save_dir='plots'):
    """Create plots for all metrics and tasks."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.tab10(np.linspace(0, 1, len(averaged_data)))
    
    # Get all metrics
    all_metrics = set()
    for task_data in averaged_data.values():
        all_metrics.update(task_data.keys())
    
    for metric in all_metrics:
        plt.figure(figsize=(12, 8))
        
        for i, (task_name, task_data) in enumerate(averaged_data.items()):
            if metric not in task_data:
                continue
                
            data = task_data[metric]
            steps = data['steps']
            mean_vals = data['mean']
            std_vals = data['std']
            
            # Plot mean with shaded std
            plt.plot(steps, mean_vals, label=f"{task_name} (n={data['num_experiments']})", 
                    color=colors[i], linewidth=2)
            plt.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, 
                           alpha=0.2, color=colors[i])
        
        plt.xlabel('Steps')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Across Tasks')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(save_dir, f'{metric}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def create_summary_table(averaged_data, save_dir='plots'):
    """Create a summary table with final metrics for each task."""
    summary_data = []
    
    for task_name, task_data in averaged_data.items():
        row = {'Task': task_name}
        
        for metric, data in task_data.items():
            if len(data['mean']) > 0:
                final_mean = data['mean'][-1]
                final_std = data['std'][-1]
                row[f'{metric}_mean'] = final_mean
                row[f'{metric}_std'] = final_std
                row[f'{metric}_formatted'] = f"{final_mean:.4f} ± {final_std:.4f}"
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    df.to_csv(os.path.join(save_dir, 'experiment_summary.csv'), index=False)
    
    # Display formatted table
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY - FINAL METRICS")
    print("="*80)
    
    for _, row in df.iterrows():
        print(f"\nTask: {row['Task']}")
        for col in df.columns:
            if col.endswith('_formatted'):
                metric_name = col.replace('_formatted', '').replace('_', ' ').title()
                print(f"  {metric_name}: {row[col]}")
    
    return df

def plot_bar_comparison(averaged_data, save_dir='plots'):
    """Create bar plots comparing final metrics across tasks."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract final accuracy values
    task_names = []
    accuracy_means = []
    accuracy_stds = []
    
    for task_name, task_data in averaged_data.items():
        if 'accuracy' in task_data and len(task_data['accuracy']['mean']) > 0:
            task_names.append(task_name)
            accuracy_means.append(task_data['accuracy']['mean'][-1])
            accuracy_stds.append(task_data['accuracy']['std'][-1])
    
    if not task_names:
        print("No accuracy data found for bar plot")
        return
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    # Create bars
    x_pos = np.arange(len(task_names))
    bars = plt.bar(x_pos, accuracy_means, yerr=accuracy_stds, 
                   capsize=5, alpha=0.7, color='skyblue', 
                   edgecolor='navy', linewidth=1)
    
    # Customize plot
    plt.xlabel('Experiment', fontsize=12, fontweight='bold')
    plt.ylabel('Final Accuracy', fontsize=12, fontweight='bold')
    plt.title('Final Accuracy Comparison Across Experiments', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, task_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(accuracy_means, accuracy_stds)):
        plt.text(i, mean_val + std_val + 0.005, f'{mean_val:.3f}±{std_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'accuracy_bar_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL ACCURACY COMPARISON")
    print("="*60)
    for i, (task, mean_val, std_val) in enumerate(zip(task_names, accuracy_means, accuracy_stds)):
        print(f"{task}: {mean_val:.4f} ± {std_val:.4f}")

def plot_multiple_metrics_bars(averaged_data, metrics=['accuracy', 'top5_accuracy'], save_dir='plots'):
    """Create grouped bar plots for multiple metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect data for all metrics
    task_names = list(averaged_data.keys())
    metrics_data = {metric: {'means': [], 'stds': []} for metric in metrics}
    
    for task_name in task_names:
        task_data = averaged_data[task_name]
        
        for metric in metrics:
            if metric in task_data and len(task_data[metric]['mean']) > 0:
                metrics_data[metric]['means'].append(task_data[metric]['mean'][-1])
                metrics_data[metric]['stds'].append(task_data[metric]['std'][-1])
            else:
                metrics_data[metric]['means'].append(0)
                metrics_data[metric]['stds'].append(0)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(task_names))
    width = 0.35
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, metrics_data[metric]['means'], 
                     width, yerr=metrics_data[metric]['stds'], 
                     label=metric.replace('_', ' ').title(), 
                     alpha=0.7, color=colors[i % len(colors)], 
                     capsize=3, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for j, (mean_val, std_val) in enumerate(zip(metrics_data[metric]['means'], 
                                                   metrics_data[metric]['stds'])):
            if mean_val > 0:  # Only show labels for valid data
                ax.text(x[j] + offset, mean_val + std_val + 0.01, 
                       f'{mean_val:.3f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Experiment', fontweight='bold', fontsize=12)
    ax.set_ylabel('Final Value', fontweight='bold', fontsize=12)
    ax.set_title('Final Metrics Comparison Across Experiments', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multiple_metrics_bar_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    base_dir = "artifacts/training_checkpoints/cifar10/vit_b_16"
    metrics =['eval/metric_accuracy', 'eval/metric_top2_accuracy', 'eval/total_loss']  # Adjust as needed
    save_dir = 'experiment_plots'
    
    print(f"Loading experiments from: {base_dir}")
    
    # Parse experiment folders
    experiments = parse_experiment_folders(base_dir)
    
    print(f"Found {len(experiments)} different tasks:")
    for task_name, exp_list in experiments.items():
        print(f"  {task_name}: {len(exp_list)} experiments")
    
    # Load TensorBoard data for all experiments
    experiments_data = {}
    
    for task_name, exp_list in experiments.items():
        task_data = []
        
        for exp_num, exp_folder in exp_list:
            print(f"Loading data for {task_name}_exp_{exp_num}...")
            
            try:
                data = load_tensorboard_data(str(exp_folder), metrics)
                task_data.append(data)
            except Exception as e:
                print(f"Error loading {exp_folder}: {e}")
                continue
        
        if task_data:
            experiments_data[task_name] = task_data
    
    # Average across experiments
    print("\nAveraging experiments...")
    averaged_data = average_experiments(experiments_data)
    
    # Create plots
    print("\nCreating plots...")
    plot_results(averaged_data, save_dir)
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(averaged_data, save_dir)
    
       # Add bar plots
    print("\nCreating bar plots...")
    plot_bar_comparison(averaged_data, save_dir)
    plot_multiple_metrics_bars(averaged_data, ['eval/metric_accuracy', 'eval/metric_top2_accuracy'], save_dir)
    
    
    print(f"\nPlots and summary saved to: {save_dir}")

if __name__ == '__main__':
    main()