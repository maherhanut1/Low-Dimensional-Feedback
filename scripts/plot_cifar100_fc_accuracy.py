import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import os
import argparse
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir, tags):
    """
    Load specific scalar data from TensorBoard logs
    
    Args:
        log_dir: Directory containing TensorBoard event files
        tags: List of scalar tags to extract
    
    Returns:
        Dictionary mapping tag names to {'steps': [], 'values': []}
    """
    log_path = Path(log_dir) / 'logs'
    if not log_path.exists():
        log_path = Path(log_dir)
    
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    data = {}
    for tag in tags:
        if tag in ea.scalars.Keys():
            events = ea.scalars.Items(tag)
            data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
    
    return data


def extract_rank_from_folder(folder_name):
    """
    Extract rank from folder name with pattern LDFA_{rank}_
    Returns None if no rank found (e.g., for BP folders)
    """
    pattern = r'LDFA_(\d+)_'
    match = re.search(pattern, folder_name)
    if match:
        return int(match.group(1))
    return None


def is_bp_folder(folder_name):
    """Check if folder is a BP (backpropagation) experiment"""
    return folder_name.startswith('BP_')


def extract_max_accuracy_from_experiments(base_dir, experiment_type='fc'):
    """
    Extract maximum validation accuracy from all experiments in a directory
    
    Args:
        base_dir: Base directory containing experiment folders (e.g., artifacts/training_checkpoints/cifar100/fc)
        experiment_type: Type of experiment (fc, fc_50, fc_75)
    
    Returns:
        DataFrame with columns: Experiment, Rank, Is_BP, Max_Acc_Top1, Max_Acc_Top5
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"ERROR: Directory {base_dir} does not exist")
        return None
    
    results = []
    
    # Iterate through all experiment folders
    for exp_folder in sorted(base_path.iterdir()):
        if not exp_folder.is_dir():
            continue
        
        folder_name = exp_folder.name
        print(f"Processing: {folder_name}")
        
        # Load TensorBoard data
        try:
            data = load_tensorboard_data(
                str(exp_folder),
                ['eval/metric_accuracy', 'eval/metric_top5_accuracy']
            )
            
            acc_data = data.get('eval/metric_accuracy', {})
            top5_data = data.get('eval/metric_top5_accuracy', {})
            
            if not acc_data.get('values'):
                print(f"  No validation data found")
                continue
            
            # Extract metrics
            acc_values = np.array(acc_data['values'])
            max_acc_top1 = np.max(acc_values)
            
            # Top5 accuracy
            if top5_data.get('values'):
                top5_values = np.array(top5_data['values'])
                max_acc_top5 = np.max(top5_values)
            else:
                max_acc_top5 = np.nan
            
            # Extract rank and BP info
            rank = extract_rank_from_folder(folder_name)
            is_bp = is_bp_folder(folder_name)
            
            results.append({
                'Experiment': folder_name,
                'Rank': rank if rank is not None else 'BP',
                'Is_BP': is_bp,
                'Max_Acc_Top1': max_acc_top1,
                'Max_Acc_Top5': max_acc_top5
            })
            
            print(f"  Max Top-1 Acc: {max_acc_top1:.4f}, Max Top-5 Acc: {max_acc_top5:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not results:
        print("No valid experiments found")
        return None
    
    return pd.DataFrame(results)


def plot_accuracy_vs_rank(df, experiment_type='fc', save_dir='experiment_plots/cifar100'):
    """
    Plot maximum accuracy vs rank for LDFA experiments with BP baseline
    
    Args:
        df: DataFrame with experiment results
        experiment_type: Type of experiment (fc, fc_50, fc_75)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate LDFA and BP experiments
    ldfa_df = df[~df['Is_BP']].copy()
    bp_df = df[df['Is_BP']].copy()
    
    if len(ldfa_df) == 0:
        print("ERROR: No LDFA experiments found!")
        return
    
    # Ensure Rank is numeric for LDFA
    ldfa_df['Rank'] = ldfa_df['Rank'].astype(int)
    
    # Group by rank and calculate mean/std for LDFA
    ldfa_grouped = ldfa_df.groupby('Rank')['Max_Acc_Top1'].agg(['mean', 'std', 'count']).reset_index()
    ldfa_grouped_top5 = ldfa_df.groupby('Rank')['Max_Acc_Top5'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error (SE = std / sqrt(n))
    ldfa_grouped['se'] = ldfa_grouped['std'] / np.sqrt(ldfa_grouped['count'])
    ldfa_grouped_top5['se'] = ldfa_grouped_top5['std'] / np.sqrt(ldfa_grouped_top5['count'])
    
    # Calculate BP mean and std
    if len(bp_df) > 0:
        bp_mean = bp_df['Max_Acc_Top1'].mean()
        bp_std = bp_df['Max_Acc_Top1'].std()
        bp_se = bp_std / np.sqrt(len(bp_df))
        bp_count = len(bp_df)
        
        bp_mean_top5 = bp_df['Max_Acc_Top5'].mean()
        bp_std_top5 = bp_df['Max_Acc_Top5'].std()
        bp_se_top5 = bp_std_top5 / np.sqrt(len(bp_df))
        
        print(f"\nBP Average Accuracy (Top-1): {bp_mean:.4f} ± {bp_se:.4f} (n={bp_count})")
        print(f"BP Average Accuracy (Top-5): {bp_mean_top5:.4f} ± {bp_se_top5:.4f} (n={bp_count})")
    else:
        bp_mean = bp_std = bp_se = bp_mean_top5 = bp_std_top5 = bp_se_top5 = np.nan
        bp_count = 0
        print("\nWarning: No BP experiments found")
    
    # Sort by rank (descending order)
    ldfa_grouped = ldfa_grouped.sort_values('Rank', ascending=False)
    ldfa_grouped_top5 = ldfa_grouped_top5.sort_values('Rank', ascending=False)
    
    print(f"\nLDFA Results by Rank for {experiment_type} (sorted high to low):")
    for _, row in ldfa_grouped.iterrows():
        print(f"  Rank {int(row['Rank'])}: {row['mean']:.4f} ± {row['se']:.4f} (n={int(row['count'])})")
    
    # ==================== PLOT 1: Top-1 Accuracy ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare x-axis positions and labels
    ranks = ldfa_grouped['Rank'].values
    x_positions = np.arange(len(ranks))
    
    # Color scheme: Blues gradient from lighter to darker for higher ranks
    ldfa_count = len(ranks)
    blues = plt.cm.Blues(np.linspace(0.35, 0.85, ldfa_count))[::-1]  # Reverse so darker is for higher ranks
    
    # Plot LDFA as bars or points
    ax.errorbar(x_positions, ldfa_grouped['mean'].values, yerr=ldfa_grouped['se'].values,
                fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
                color='steelblue', ecolor='steelblue', label='LDFA')
    
    # Plot BP baseline as horizontal line
    if not np.isnan(bp_mean):
        ax.axhline(bp_mean, color='black', linewidth=2, linestyle='-', label='BP Baseline')
        # Add shaded region for BP std
        ax.axhspan(bp_mean - bp_se, bp_mean + bp_se, alpha=0.2, color='black')
    
    # Formatting
    ax.set_xlabel('Backward Rank', fontsize=16, fontweight='bold')
    ax.set_ylabel('Maximum Top-1 Accuracy', fontsize=16, fontweight='bold')
    ax.set_title(f'Maximum Accuracy vs Rank - CIFAR-100 ({experiment_type.upper()})', fontsize=18, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(r)) for r in ranks], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_file = os.path.join(save_dir, f'accuracy_vs_rank_{experiment_type}_top1.png')
    output_file_svg = os.path.join(save_dir, f'accuracy_vs_rank_{experiment_type}_top1.svg')
    output_file_pdf = os.path.join(save_dir, f'accuracy_vs_rank_{experiment_type}_top1.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"\nTop-1 plot saved to: {output_file}")
    print(f"Top-1 plot saved to: {output_file_svg}")
    print(f"Top-1 plot saved to: {output_file_pdf}")
    
    # ==================== PLOT 2: Top-5 Accuracy ====================
    if not ldfa_grouped_top5['mean'].isna().all():
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot LDFA
        ax2.errorbar(x_positions, ldfa_grouped_top5['mean'].values, yerr=ldfa_grouped_top5['se'].values,
                    fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='steelblue', ecolor='steelblue', label='LDFA')
        
        # Plot BP baseline
        if not np.isnan(bp_mean_top5):
            ax2.axhline(bp_mean_top5, color='black', linewidth=2, linestyle='-', label='BP Baseline')
            ax2.axhspan(bp_mean_top5 - bp_se_top5, bp_mean_top5 + bp_se_top5, alpha=0.2, color='black')
        
        # Formatting
        ax2.set_xlabel('Backward Rank', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Maximum Top-5 Accuracy', fontsize=16, fontweight='bold')
        ax2.set_title(f'Maximum Accuracy vs Rank - CIFAR-100 ({experiment_type.upper()})', fontsize=18, fontweight='bold')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([str(int(r)) for r in ranks], fontsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.legend(fontsize=12, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        output_file2 = os.path.join(save_dir, f'accuracy_vs_rank_{experiment_type}_top5.png')
        output_file2_svg = os.path.join(save_dir, f'accuracy_vs_rank_{experiment_type}_top5.svg')
        output_file2_pdf = os.path.join(save_dir, f'accuracy_vs_rank_{experiment_type}_top5.pdf')
        plt.savefig(output_file2, dpi=300, bbox_inches='tight')
        plt.savefig(output_file2_svg, bbox_inches='tight')
        plt.savefig(output_file2_pdf, bbox_inches='tight')
        print(f"\nTop-5 plot saved to: {output_file2}")
        print(f"Top-5 plot saved to: {output_file2_svg}")
        print(f"Top-5 plot saved to: {output_file2_pdf}")


def plot_all_experiments_combined(experiments_data, save_dir='experiment_plots/cifar100'):
    """
    Plot all experiments (fc, fc_50, fc_75) in a single combined plot
    
    Args:
        experiments_data: Dictionary mapping experiment_type to DataFrame
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color schemes for each experiment type
    colors = {
        'fc': {'ldfa': 'steelblue', 'bp': 'navy', 'label': '100 Classes'},
        'fc_75': {'ldfa': 'coral', 'bp': 'darkred', 'label': '75 Classes'},
        'fc_50': {'ldfa': 'mediumseagreen', 'bp': 'darkgreen', 'label': '50 Classes'}
    }
    
    # Markers for each experiment type
    markers = {
        'fc': 'o',
        'fc_75': 's',
        'fc_50': '^'
    }
    
    for exp_type, df in experiments_data.items():
        if df is None or len(df) == 0:
            continue
        
        # Separate LDFA and BP experiments
        ldfa_df = df[~df['Is_BP']].copy()
        bp_df = df[df['Is_BP']].copy()
        
        if len(ldfa_df) == 0:
            print(f"WARNING: No LDFA experiments found for {exp_type}")
            continue
        
        # Ensure Rank is numeric for LDFA
        ldfa_df['Rank'] = ldfa_df['Rank'].astype(int)
        
        # Group by rank and calculate mean/std for LDFA
        ldfa_grouped = ldfa_df.groupby('Rank')['Max_Acc_Top1'].agg(['mean', 'std', 'count']).reset_index()
        ldfa_grouped['se'] = ldfa_grouped['std'] / np.sqrt(ldfa_grouped['count'])
        
        # Sort by rank (ascending for plotting)
        ldfa_grouped = ldfa_grouped.sort_values('Rank', ascending=True)
        
        # Calculate BP mean and std
        if len(bp_df) > 0:
            bp_mean = bp_df['Max_Acc_Top1'].mean()
            bp_std = bp_df['Max_Acc_Top1'].std()
            bp_se = bp_std / np.sqrt(len(bp_df))
            
            print(f"\n{exp_type.upper()} - BP Average Accuracy: {bp_mean:.4f} ± {bp_se:.4f}")
        else:
            bp_mean = bp_se = np.nan
            print(f"\n{exp_type.upper()} - WARNING: No BP experiments found")
        
        print(f"{exp_type.upper()} - LDFA Results by Rank:")
        for _, row in ldfa_grouped.iterrows():
            print(f"  Rank {int(row['Rank'])}: {row['mean']:.4f} ± {row['se']:.4f} (n={int(row['count'])})")
        
        # Get color scheme
        color_scheme = colors[exp_type]
        marker = markers[exp_type]
        
        # Plot LDFA curve
        ax.errorbar(ldfa_grouped['Rank'].values, ldfa_grouped['mean'].values, 
                   yerr=ldfa_grouped['se'].values,
                   fmt=f'-{marker}', markersize=8, capsize=4, capthick=1.5, linewidth=2,
                   color=color_scheme['ldfa'], ecolor=color_scheme['ldfa'],
                   label=f'LDFA - {color_scheme["label"]}')
        
        # Plot BP baseline as horizontal line
        if not np.isnan(bp_mean):
            # Get the x-range for this experiment's ranks
            rank_min = ldfa_grouped['Rank'].min()
            rank_max = ldfa_grouped['Rank'].max()
            
            ax.hlines(bp_mean, rank_min, rank_max, 
                     color=color_scheme['bp'], linewidth=2.5, linestyle='--',
                     label=f'BP - {color_scheme["label"]}')
            
            # Add shaded region for BP std
            ax.fill_between([rank_min, rank_max], 
                          bp_mean - bp_se, bp_mean + bp_se,
                          alpha=0.15, color=color_scheme['bp'])
    
    # Formatting
    ax.set_xlabel('Backward Rank', fontsize=18, fontweight='bold')
    ax.set_ylabel('Maximum Top-1 Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('Maximum Accuracy vs Rank - CIFAR-100 Subsampling', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_file = os.path.join(save_dir, 'accuracy_vs_rank_combined_all.png')
    output_file_svg = os.path.join(save_dir, 'accuracy_vs_rank_combined_all.svg')
    output_file_pdf = os.path.join(save_dir, 'accuracy_vs_rank_combined_all.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_file}")
    print(f"Combined plot saved to: {output_file_svg}")
    print(f"Combined plot saved to: {output_file_pdf}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot accuracy vs rank for CIFAR-100 FC experiments')
    parser.add_argument('--base_dir', type=str,
                        default='artifacts/training_checkpoints/cifar100',
                        help='Base directory containing fc, fc_50, fc_75 folders')
    parser.add_argument('--output_dir', type=str,
                        default='experiment_plots/cifar100',
                        help='Directory to save plots')
    parser.add_argument('--combined', action='store_true', default=True,
                        help='Create combined plot with all experiments')
    parser.add_argument('--individual', action='store_true',
                        help='Create individual plots for each experiment type')
    
    args = parser.parse_args()
    
    # Process all three experiment types
    experiment_types = ['fc', 'fc_50', 'fc_75']
    experiments_data = {}
    
    for exp_type in experiment_types:
        exp_dir = os.path.join(args.base_dir, exp_type)
        print(f"\n{'='*60}")
        print(f"Processing {exp_type.upper()}")
        print(f"Directory: {exp_dir}")
        print('='*60)
        
        # Extract data
        df = extract_max_accuracy_from_experiments(exp_dir, exp_type)
        
        if df is not None:
            experiments_data[exp_type] = df
            
            # Save to CSV
            csv_file = os.path.join(args.output_dir, f'accuracy_summary_{exp_type}.csv')
            os.makedirs(args.output_dir, exist_ok=True)
            df.to_csv(csv_file, index=False)
            print(f"Data saved to: {csv_file}")
            
            # Create individual plots if requested
            if args.individual:
                plot_accuracy_vs_rank(df, exp_type, args.output_dir)
        else:
            print(f"Failed to extract data for {exp_type}")
    
    # Create combined plot
    if args.combined and experiments_data:
        print(f"\n{'='*60}")
        print("Creating combined plot")
        print('='*60)
        plot_all_experiments_combined(experiments_data, args.output_dir)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
