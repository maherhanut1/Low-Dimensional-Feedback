import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import os
import argparse
from scipy.stats import ttest_rel

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
    return bool(re.search(r'_BP_|_BP$', task_name, re.IGNORECASE))

def plot_rank_vs_accuracy_bar(csv_file, save_dir='experiment_plots'):
    """
    Plot bar chart of maximum accuracy vs rank for LDFA tasks with BP baseline
    
    Args:
        csv_file: Path to the training summary CSV file
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} experiments from {csv_file}")
    
    # Convert accuracy strings to floats
    df['Max_Acc'] = df['Max_Val_Acc_Top1'].astype(float)
    df['Max_Acc_Top2'] = df['Max_Val_Acc_Top2'].apply(lambda x: float(x) if x != 'N/A' else np.nan)
    
    # Extract ranks and separate BP tasks
    df['Rank'] = df['Task'].apply(extract_rank_from_task)
    df['Is_BP'] = df['Task'].apply(is_bp_task)
    
    # Separate LDFA and BP tasks
    ldfa_df = df[df['Rank'].notna()].copy()
    bp_df = df[df['Is_BP']].copy()
    
    if len(ldfa_df) == 0:
        print("ERROR: No LDFA experiments found! Check task naming pattern.")
        return None
    
    # Group by rank and calculate mean/std for LDFA
    ldfa_grouped = ldfa_df.groupby('Rank')['Max_Acc'].agg(['mean', 'std', 'count']).reset_index()
    ldfa_grouped_top2 = ldfa_df.groupby('Rank')['Max_Acc_Top2'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error (SE = std / sqrt(n))
    ldfa_grouped['se'] = ldfa_grouped['std'] / np.sqrt(ldfa_grouped['count'])
    ldfa_grouped_top2['se'] = ldfa_grouped_top2['std'] / np.sqrt(ldfa_grouped_top2['count'])
    
    # Calculate BP mean and std
    if len(bp_df) > 0:
        bp_mean = bp_df['Max_Acc'].mean()
        bp_std = bp_df['Max_Acc'].std()
        bp_se = bp_std / np.sqrt(len(bp_df))
        bp_count = len(bp_df)
        
        bp_mean_top2 = bp_df['Max_Acc_Top2'].mean()
        bp_std_top2 = bp_df['Max_Acc_Top2'].std()
        bp_se_top2 = bp_std_top2 / np.sqrt(len(bp_df))
        
        print(f"\nBP Average Accuracy (Top1): {bp_mean:.4f} ± {bp_se:.4f} (n={bp_count})")
        print(f"BP Average Accuracy (Top2): {bp_mean_top2:.4f} ± {bp_se_top2:.4f} (n={bp_count})")
    else:
        bp_mean = bp_std = bp_se = bp_mean_top2 = bp_std_top2 = bp_se_top2 = np.nan
        bp_count = 0
        print("\nWarning: No BP experiments found")
    
    # Sort by rank (descending order)
    ldfa_grouped = ldfa_grouped.sort_values('Rank', ascending=False)
    ldfa_grouped_top2 = ldfa_grouped_top2.sort_values('Rank', ascending=False)
    
    print("\nLDFA Results by Rank (sorted high to low):")
    for _, row in ldfa_grouped.iterrows():
        print(f"  Rank {int(row['Rank'])}: {row['mean']:.4f} ± {row['se']:.4f} (n={int(row['count'])})")
    
    # ==================== PLOT 1: Top-1 Accuracy Bar Chart ====================
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Use lighter blues color scheme (lighter for smaller ranks, darker for larger ranks)
    blues = plt.cm.Blues(np.linspace(0.3, 0.7, len(ldfa_grouped)))
    
    # --- Top-1 Accuracy Bar Chart ---
    # Prepare data with BP first, then LDFA ranks (high to low)
    if not np.isnan(bp_mean):
        all_labels = ['BP'] + [str(int(r)) for r in ldfa_grouped['Rank'].values]
        all_means = [bp_mean] + ldfa_grouped['mean'].tolist()
        all_ses = [bp_se] + ldfa_grouped['se'].tolist()
        # BP is black, then blues gradient (reversed since ranks are high to low)
        colors = ['black'] + list(reversed(blues))
    else:
        all_labels = [str(int(r)) for r in ldfa_grouped['Rank'].values]
        all_means = ldfa_grouped['mean'].tolist()
        all_ses = ldfa_grouped['se'].tolist()
        colors = list(reversed(blues))
    
    x_pos = np.arange(len(all_labels))
    bars1 = ax1.bar(x_pos, all_means, yerr=all_ses, 
                    width=0.8,
                    capsize=5, 
                    color=colors, 
                    edgecolor='none',
                    error_kw={'capthick': 2})
    
    ax1.set_xlabel('Rank', fontsize=12, fontname='Arial')
    ax1.set_ylabel('Mean Accuracy', fontsize=12, fontname='Arial')
    ax1.set_title('Bar Plot of Maximum Accuracy (Top-1)', fontsize=14, fontname='Arial')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_labels, fontsize=10, fontname='Arial')
    ax1.tick_params(axis='y', labelsize=10)
    
    ax1.legend(['Mean ± SE'], fontsize=10, loc='lower right')
    ax1.set_ylim(bottom=0.85, top=0.925)
    
    plt.tight_layout()
    
    # Save Top-1 plot
    output_file1 = os.path.join(save_dir, 'rank_vs_accuracy_barplot_top1.png')
    output_file1_svg = os.path.join(save_dir, 'rank_vs_accuracy_barplot_top1.svg')
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.savefig(output_file1_svg, bbox_inches='tight')
    print(f"\nTop-1 bar plot saved to: {output_file1}")
    print(f"Top-1 bar plot saved to: {output_file1_svg}")
    plt.show()
    
    # ==================== PLOT 2: Top-2 Accuracy Bar Chart ====================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Use lighter blues color scheme (lighter for smaller ranks, darker for larger ranks)
    blues_top2 = plt.cm.Blues(np.linspace(0.3, 0.7, len(ldfa_grouped_top2)))
    
    # --- Top-2 Accuracy Bar Chart ---
    if not np.isnan(bp_mean_top2):
        all_labels_top2 = ['BP'] + [str(int(r)) for r in ldfa_grouped_top2['Rank'].values]
        all_means_top2 = [bp_mean_top2] + ldfa_grouped_top2['mean'].tolist()
        all_ses_top2 = [bp_se_top2] + ldfa_grouped_top2['se'].tolist()
        # BP is black, then blues gradient (reversed since ranks are high to low)
        colors_top2 = ['black'] + list(reversed(blues_top2))
    else:
        all_labels_top2 = [str(int(r)) for r in ldfa_grouped_top2['Rank'].values]
        all_means_top2 = ldfa_grouped_top2['mean'].tolist()
        all_ses_top2 = ldfa_grouped_top2['se'].tolist()
        colors_top2 = list(reversed(blues_top2))
    
    x_pos_top2 = np.arange(len(all_labels_top2))
    bars2 = ax2.bar(x_pos_top2, all_means_top2, yerr=all_ses_top2, 
                    width=0.8,
                    capsize=5, 
                    color=colors_top2, 
                    edgecolor='none',
                    error_kw={'capthick': 2})
    
    ax2.set_xlabel('Rank', fontsize=12, fontname='Arial')
    ax2.set_ylabel('Mean Accuracy', fontsize=12, fontname='Arial')
    ax2.set_title('Bar Plot of Maximum Accuracy (Top-2)', fontsize=14, fontname='Arial')
    ax2.set_xticks(x_pos_top2)
    ax2.set_xticklabels(all_labels_top2, fontsize=10, fontname='Arial')
    ax2.tick_params(axis='y', labelsize=10)
    
    ax2.legend(['Mean ± SE'], fontsize=10, loc='lower right')
    ax2.set_ylim(bottom=0.93, top=0.99)
    
    plt.tight_layout()
    
    # Save Top-2 plot
    output_file2 = os.path.join(save_dir, 'rank_vs_accuracy_barplot_top2.png')
    output_file2_svg = os.path.join(save_dir, 'rank_vs_accuracy_barplot_top2.svg')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.savefig(output_file2_svg, bbox_inches='tight')
    print(f"\nTop-2 bar plot saved to: {output_file2}")
    print(f"Top-2 bar plot saved to: {output_file2_svg}")
    plt.show()


def plot_rank_vs_accuracy(csv_file, save_dir='experiment_plots'):
    """
    Plot maximum accuracy vs rank for LDFA tasks with BP baseline
    
    Args:
        csv_file: Path to the training summary CSV file
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} experiments from {csv_file}")
    print(f"\nTask names in CSV:")
    for task in df['Task'].unique():
        print(f"  {task}")
    
    # Convert accuracy strings to floats
    df['Max_Acc'] = df['Max_Val_Acc_Top1'].astype(float)
    df['Max_Acc_Top2'] = df['Max_Val_Acc_Top2'].apply(lambda x: float(x) if x != 'N/A' else np.nan)
    
    # Extract ranks and separate BP tasks
    df['Rank'] = df['Task'].apply(extract_rank_from_task)
    df['Is_BP'] = df['Task'].apply(is_bp_task)
    
    print(f"\nExtracted ranks:")
    for task, rank in zip(df['Task'], df['Rank']):
        print(f"  {task} -> Rank: {rank}")
    
    # Separate LDFA and BP tasks
    ldfa_df = df[df['Rank'].notna()].copy()
    bp_df = df[df['Is_BP']].copy()
    
    print(f"\nFound {len(ldfa_df)} LDFA experiments")
    print(f"Found {len(bp_df)} BP experiments")
    
    if len(ldfa_df) == 0:
        print("ERROR: No LDFA experiments found! Check task naming pattern.")
        return None
    
    # Group by rank and calculate mean/std for LDFA
    ldfa_grouped = ldfa_df.groupby('Rank')['Max_Acc'].agg(['mean', 'std', 'count']).reset_index()
    ldfa_grouped_top2 = ldfa_df.groupby('Rank')['Max_Acc_Top2'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate BP mean and std
    if len(bp_df) > 0:
        bp_mean = bp_df['Max_Acc'].mean()
        bp_std = bp_df['Max_Acc'].std()
        bp_count = len(bp_df)
        
        bp_mean_top2 = bp_df['Max_Acc_Top2'].mean()
        bp_std_top2 = bp_df['Max_Acc_Top2'].std()
        
        print(f"\nBP Average Accuracy (Top1): {bp_mean:.4f} ± {bp_std:.4f} (n={bp_count})")
        print(f"BP Average Accuracy (Top2): {bp_mean_top2:.4f} ± {bp_std_top2:.4f} (n={bp_count})")
    else:
        bp_mean = bp_std = bp_mean_top2 = bp_std_top2 = np.nan
        bp_count = 0
        print("\nWarning: No BP experiments found")
    
    print("\nLDFA Results by Rank:")
    for _, row in ldfa_grouped.iterrows():
        print(f"  Rank {int(row['Rank'])}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")
    
    # Set style for line plots
    plt.rc('font', family='Arial', size=12)
    
    # Use blues color scheme from original code
    blues = plt.cm.Blues(np.linspace(0.4, 1, 4))
    
    # Calculate median rank for BP positioning
    ranks = ldfa_grouped['Rank'].values
    median_rank = np.median(ranks)
    
    # ==================== PLOT 1: Top-1 Accuracy ====================
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)  # Put grid behind plot elements
    
    means = ldfa_grouped['mean'].values
    stds = ldfa_grouped['std'].values
    
    # Plot LDFA data with error bars using blue color
    ax1.errorbar(ranks, means, yerr=stds, 
                fmt='-o', capsize=5,
                linewidth=2, markersize=7,
                color=blues[2],
                label='LDFA')
    
    # Plot BP baseline as horizontal line and error bar at median rank
    if not np.isnan(bp_mean):
        ax1.plot([min(ranks), max(ranks)], [bp_mean, bp_mean], 
                linestyle='-.', linewidth=2, color='black')
        
        ax1.errorbar([median_rank], [bp_mean], yerr=[bp_std], 
                    fmt='-.', capsize=5, capthick=2, 
                    marker='s', markersize=7, 
                    linestyle='none', 
                    color='black', ecolor='black',
                    label=f'BP Baseline')
    
    ax1.set_xlabel('Backward Rank', fontsize=15, fontname='Arial')
    ax1.set_ylabel('Accuracy', fontsize=15, fontname='Arial')
    ax1.set_title('LDFA: Maximum Accuracy vs Rank (Top-1)', fontsize=14, fontname='Arial')
    ax1.set_ylim(bottom=0.6)
    ax1.tick_params(labelsize=11)
    ax1.legend(fontsize=14)
    
    plt.tight_layout()
    
    # Save Top-1 plot
    output_file1 = os.path.join(save_dir, 'rank_vs_accuracy_top1.png')
    output_file1_svg = os.path.join(save_dir, 'rank_vs_accuracy_top1.svg')
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.savefig(output_file1_svg, bbox_inches='tight')
    print(f"\nTop-1 plot saved to: {output_file1}")
    print(f"Top-1 plot saved to: {output_file1_svg}")
    plt.show()
    
    # ==================== PLOT 2: Top-2 Accuracy ====================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)  # Put grid behind plot elements
    
    ranks_top2 = ldfa_grouped_top2['Rank'].values
    means_top2 = ldfa_grouped_top2['mean'].values
    stds_top2 = ldfa_grouped_top2['std'].values
    median_rank_top2 = np.median(ranks_top2)
    
    # Plot LDFA data with error bars using blue color
    ax2.errorbar(ranks_top2, means_top2, yerr=stds_top2, 
                fmt='-o', capsize=5,
                linewidth=2, markersize=7,
                color=blues[2],
                label='LDFA')
    
    # Plot BP baseline as horizontal line and error bar at median rank
    if not np.isnan(bp_mean_top2):
        ax2.plot([min(ranks_top2), max(ranks_top2)], [bp_mean_top2, bp_mean_top2], 
                linestyle='-.', linewidth=2, color='black')
        
        ax2.errorbar([median_rank_top2], [bp_mean_top2], yerr=[bp_std_top2], 
                    fmt='-.', capsize=5, capthick=2, 
                    marker='s', markersize=7, 
                    linestyle='none', 
                    color='black', ecolor='black',
                    label=f'BP Baseline')
    
    ax2.set_xlabel('Backward Rank', fontsize=15, fontname='Arial')
    ax2.set_ylabel('Accuracy', fontsize=15, fontname='Arial')
    ax2.set_title('LDFA: Maximum Accuracy vs Rank (Top-2)', fontsize=14, fontname='Arial')
    ax2.set_ylim(bottom=0.6)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=14)
    
    plt.tight_layout()
    
    # Save Top-2 plot
    output_file2 = os.path.join(save_dir, 'rank_vs_accuracy_top2.png')
    output_file2_svg = os.path.join(save_dir, 'rank_vs_accuracy_top2.svg')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.savefig(output_file2_svg, bbox_inches='tight')
    print(f"\nTop-2 plot saved to: {output_file2}")
    print(f"Top-2 plot saved to: {output_file2_svg}")
    plt.show()
    
    # Create summary table
    summary_data = []
    for _, row in ldfa_grouped.iterrows():
        summary_data.append({
            'Rank': int(row['Rank']),
            'Mean_Accuracy_Top1': f"{row['mean']:.4f}",
            'Std_Accuracy_Top1': f"{row['std']:.4f}",
            'Num_Experiments': int(row['count'])
        })
    
    # Add top-2 data
    for i, (_, row) in enumerate(ldfa_grouped_top2.iterrows()):
        if i < len(summary_data):
            summary_data[i]['Mean_Accuracy_Top2'] = f"{row['mean']:.4f}"
            summary_data[i]['Std_Accuracy_Top2'] = f"{row['std']:.4f}"
    
    # Add BP baseline
    if bp_count > 0:
        summary_data.append({
            'Rank': 'BP',
            'Mean_Accuracy_Top1': f"{bp_mean:.4f}",
            'Std_Accuracy_Top1': f"{bp_std:.4f}",
            'Mean_Accuracy_Top2': f"{bp_mean_top2:.4f}",
            'Std_Accuracy_Top2': f"{bp_std_top2:.4f}",
            'Num_Experiments': bp_count
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(save_dir, 'rank_accuracy_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Plot rank vs accuracy bar and line plots')
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
    
    # Create line plot
    summary_df = plot_rank_vs_accuracy(csv_file, save_dir)
    
    # Create bar plot
    plot_rank_vs_accuracy_bar(csv_file, save_dir)
    
    if summary_df is not None:
        print("\n" + "="*60)
        print("RANK vs ACCURACY SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()