import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import os
import argparse


def extract_rank_from_task(task_name):
    """
    Extract rank from task name with pattern LDFA_{rank}_ or ldfa_{rank}_
    Returns None if no rank found (e.g., for BP tasks)
    """
    # Search anywhere in the task name, not just at the start
    pattern = r'[Ll][Dd][Ff][Aa]_(\d+)'
    match = re.search(pattern, task_name)
    if match:
        return int(match.group(1))
    return None


def is_bp_task(task_name):
    """Check if task is a BP (backpropagation) task"""
    # Check if BP appears anywhere in the task name
    return bool(re.search(r'^BP_|_BP_|_BP$', task_name, re.IGNORECASE))


def plot_convergence_comparison(csv_file, save_dir='experiment_plots'):
    """
    Plot convergence speed comparison showing steps to reach 90% of final accuracy.
    
    Args:
        csv_file: Path to the training summary CSV file
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} experiments from {csv_file}")
    
    # Extract ranks and separate BP tasks
    df['Rank'] = df['Task'].apply(extract_rank_from_task)
    df['Is_BP'] = df['Task'].apply(is_bp_task)

    # Convert step to 90% of last to numeric (handle 'N/A' values)
    df['Step_90pct_Last_Numeric'] = pd.to_numeric(df['Step_90pct_Last'], errors='coerce')
    
    # Calculate convergence ratio (step to 90% of last / last step)
    df['Last_Epoch_Step_Numeric'] = pd.to_numeric(df['Last_Epoch_Step'], errors='coerce')
    df['Convergence_Ratio'] = df['Step_90pct_Last_Numeric'] / df['Last_Epoch_Step_Numeric']
    
    # Separate LDFA and BP tasks
    ldfa_df = df[df['Rank'].notna()].copy()
    bp_df = df[df['Is_BP']].copy()
    
    if len(ldfa_df) == 0:
        print("ERROR: No LDFA experiments found!")
        return None
    
    # Group by rank/task and calculate statistics
    ldfa_grouped_steps = ldfa_df.groupby('Rank')['Step_90pct_Last_Numeric'].agg(['mean', 'std', 'count']).reset_index()
    ldfa_grouped_ratio = ldfa_df.groupby('Rank')['Convergence_Ratio'].agg(['mean', 'std', 'count']).reset_index()
    
    if len(bp_df) > 0:
        bp_mean_steps = bp_df['Step_90pct_Last_Numeric'].mean()
        bp_std_steps = bp_df['Step_90pct_Last_Numeric'].std()
        bp_mean_ratio = bp_df['Convergence_Ratio'].mean()
        bp_std_ratio = bp_df['Convergence_Ratio'].std()
        bp_count = len(bp_df)
        
        print(f"\nBP Convergence:")
        print(f"  Steps to 90% of final: {bp_mean_steps:.1f} ± {bp_std_steps:.1f}")
        print(f"  Convergence ratio: {bp_mean_ratio:.3f} ± {bp_std_ratio:.3f}")
    else:
        bp_mean_steps = bp_std_steps = bp_mean_ratio = bp_std_ratio = np.nan
        bp_count = 0
    
    print("\nLDFA Convergence by Rank:")
    for _, row in ldfa_grouped_steps.iterrows():
        ratio_row = ldfa_grouped_ratio[ldfa_grouped_ratio['Rank'] == row['Rank']].iloc[0]
        print(f"  Rank {int(row['Rank'])}: {row['mean']:.1f} ± {row['std']:.1f} steps, "
              f"ratio: {ratio_row['mean']:.3f} ± {ratio_row['std']:.3f}")
    
    # Sort by rank (descending for bar plot)
    ldfa_grouped_steps = ldfa_grouped_steps.sort_values('Rank', ascending=False)
    ldfa_grouped_ratio = ldfa_grouped_ratio.sort_values('Rank', ascending=False)
    
    # Generate colors (lighter for smaller ranks, darker for larger)
    blues = plt.cm.Blues(np.linspace(0.35, 0.85, len(ldfa_grouped_steps)))
    
    # ==================== PLOT 1: Steps to 90% Max Accuracy ====================
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Prepare data with BP first, then LDFA ranks (high to low)
    if not np.isnan(bp_mean_steps):
        all_labels = ['BP'] + [str(int(r)) for r in ldfa_grouped_steps['Rank'].values]
        all_means = [bp_mean_steps] + ldfa_grouped_steps['mean'].tolist()
        all_stds = [bp_std_steps] + ldfa_grouped_steps['std'].tolist()
        colors = ['black'] + list(reversed(blues))
    else:
        all_labels = [str(int(r)) for r in ldfa_grouped_steps['Rank'].values]
        all_means = ldfa_grouped_steps['mean'].tolist()
        all_stds = ldfa_grouped_steps['std'].tolist()
        colors = list(reversed(blues))
    
    x_pos = np.arange(len(all_labels))
    bars1 = ax1.bar(x_pos, all_means, yerr=all_stds,
                    width=0.8,
                    capsize=5,
                    color=colors,
                    edgecolor='none',
                    error_kw={'capthick': 2})
    
    ax1.set_xlabel('Rank', fontsize=12, fontname='Arial')
    ax1.set_ylabel('Steps to Reach 90% of Final Accuracy', fontsize=12, fontname='Arial')
    ax1.set_title('Convergence Speed: Steps to 90% of Final Accuracy', fontsize=14, fontname='Arial')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_labels, fontsize=10, fontname='Arial')
    ax1.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file1 = os.path.join(save_dir, 'convergence_steps_to_90pct.png')
    output_file1_svg = os.path.join(save_dir, 'convergence_steps_to_90pct.svg')
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.savefig(output_file1_svg, bbox_inches='tight')
    print(f"\nConvergence steps plot saved to: {output_file1}")
    plt.show()
    
    # ==================== PLOT 2: Convergence Ratio (Normalized) ====================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Prepare data with BP first, then LDFA ranks (high to low)
    if not np.isnan(bp_mean_ratio):
        all_labels_ratio = ['BP'] + [str(int(r)) for r in ldfa_grouped_ratio['Rank'].values]
        all_means_ratio = [bp_mean_ratio] + ldfa_grouped_ratio['mean'].tolist()
        all_stds_ratio = [bp_std_ratio] + ldfa_grouped_ratio['std'].tolist()
        colors_ratio = ['black'] + list(reversed(blues))
    else:
        all_labels_ratio = [str(int(r)) for r in ldfa_grouped_ratio['Rank'].values]
        all_means_ratio = ldfa_grouped_ratio['mean'].tolist()
        all_stds_ratio = ldfa_grouped_ratio['std'].tolist()
        colors_ratio = list(reversed(blues))
    
    x_pos_ratio = np.arange(len(all_labels_ratio))
    bars2 = ax2.bar(x_pos_ratio, all_means_ratio, yerr=all_stds_ratio,
                    width=0.8,
                    capsize=5,
                    color=colors_ratio,
                    edgecolor='none',
                    error_kw={'capthick': 2})
    
    ax2.set_xlabel('Rank', fontsize=12, fontname='Arial')
    ax2.set_ylabel('Convergence Ratio (fraction of training)', fontsize=12, fontname='Arial')
    ax2.set_title('Convergence Speed: Normalized by Total Training Steps', fontsize=14, fontname='Arial')
    ax2.set_xticks(x_pos_ratio)
    ax2.set_xticklabels(all_labels_ratio, fontsize=10, fontname='Arial')
    ax2.tick_params(axis='y', labelsize=10)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save plot
    output_file2 = os.path.join(save_dir, 'convergence_ratio.png')
    output_file2_svg = os.path.join(save_dir, 'convergence_ratio.svg')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.savefig(output_file2_svg, bbox_inches='tight')
    print(f"Convergence ratio plot saved to: {output_file2}")
    plt.show()
    
    # Create summary CSV
    summary_data = []
    for i, row_steps in ldfa_grouped_steps.iterrows():
        rank = int(row_steps['Rank'])
        row_ratio = ldfa_grouped_ratio[ldfa_grouped_ratio['Rank'] == rank].iloc[0]
        
        summary_data.append({
            'Rank': rank,
            'Mean_Steps_to_90pct': f"{row_steps['mean']:.1f}",
            'Std_Steps_to_90pct': f"{row_steps['std']:.1f}",
            'Mean_Convergence_Ratio': f"{row_ratio['mean']:.3f}",
            'Std_Convergence_Ratio': f"{row_ratio['std']:.3f}",
            'Num_Experiments': int(row_steps['count'])
        })
    
    # Add BP baseline
    if bp_count > 0:
        summary_data.append({
            'Rank': 'BP',
            'Mean_Steps_to_90pct': f"{bp_mean_steps:.1f}",
            'Std_Steps_to_90pct': f"{bp_std_steps:.1f}",
            'Mean_Convergence_Ratio': f"{bp_mean_ratio:.3f}",
            'Std_Convergence_Ratio': f"{bp_std_ratio:.3f}",
            'Num_Experiments': bp_count
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(save_dir, 'convergence_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Convergence summary saved to: {summary_file}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Plot convergence speed comparison')
    parser.add_argument('--csv_file', type=str, 
                        default='experiment_plots/training_summary_detailed.csv',
                        help='Path to training summary CSV file')
    parser.add_argument('--output_dir', type=str, 
                        default='experiment_plots',
                        help='Directory to save output plots')
    args = parser.parse_args()
    
    # Configuration
    csv_file = args.csv_file
    save_dir = args.output_dir
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        print("Please run extract_training_summary.py first to generate the CSV file.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create convergence plots
    summary_df = plot_convergence_comparison(csv_file, save_dir)
    
    if summary_df is not None:
        print("\n" + "="*60)
        print("CONVERGENCE SPEED SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
