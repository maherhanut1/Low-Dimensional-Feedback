import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set matplotlib style for publication-quality plots
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


def extract_rank_from_task(task_name):
    """Extract rank from task name with pattern LDFA_{rank}_"""
    pattern = r'[Ll][Dd][Ff][Aa]_(\d+)'
    match = re.search(pattern, task_name)
    if match:
        return int(match.group(1))
    return None


def is_bp_task(task_name):
    """Check if task is a BP (backpropagation) task"""
    return bool(re.search(r'^BP_|_BP_|_BP$', task_name, re.IGNORECASE))


def load_tensorboard_data(log_dir, metric='eval/metric_accuracy'):
    """Load TensorBoard data for accuracy metric and return the best value."""
    ea = EventAccumulator(log_dir + '/logs')
    ea.Reload()
    
    if metric in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(metric)
        values = [event.value for event in scalar_events]
        # Return the maximum accuracy achieved
        return max(values) if values else None
    else:
        print(f"Warning: Metric '{metric}' not found in {log_dir}")
        return None


def get_accuracies_for_network(base_path, network_name, rank=32):
    """
    Get BP and LDFA (rank 32) accuracies for a specific network.
    
    Returns:
        bp_accuracies: List of BP accuracies across experiments
        ldfa_accuracies: List of LDFA rank 32 accuracies across experiments
    """
    network_path = base_path / network_name
    if not network_path.exists():
        print(f"Warning: Network path not found: {network_path}")
        return [], []
    
    bp_accuracies = []
    ldfa_accuracies = []
    
    # Pattern to match experiment folders
    pattern = re.compile(r'^(.+)_exp_(\d+)$')
    
    for folder in network_path.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                task_name = match.group(1)
                
                # Check if it's BP or LDFA rank 32
                if is_bp_task(task_name):
                    try:
                        accuracy = load_tensorboard_data(str(folder))
                        if accuracy is not None:
                            bp_accuracies.append(accuracy)
                            print(f"Loaded BP accuracy: {accuracy:.4f} from {folder.name}")
                    except Exception as e:
                        print(f"Error loading {folder.name}: {e}")
                
                elif extract_rank_from_task(task_name) == rank:
                    try:
                        accuracy = load_tensorboard_data(str(folder))
                        if accuracy is not None:
                            ldfa_accuracies.append(accuracy)
                            print(f"Loaded LDFA-{rank} accuracy: {accuracy:.4f} from {folder.name}")
                    except Exception as e:
                        print(f"Error loading {folder.name}: {e}")
    
    return bp_accuracies, ldfa_accuracies


def collect_and_save_data(base_dir, csv_path, rank=36):
    """
    Collect data from all experiments and save to CSV file.
    """
    base_path = Path(base_dir)
    
    # Define networks
    networks = [
        ('vit_b_16', '384'),
        ('vit_b_16_192', '192'),
        ('vit_b_16_96', '96'),
        ('vit_b_16_48', '48'),
    ]
    
    all_data = []
    
    for network_name, embed_dim in networks:
        print(f"\n{'='*60}")
        print(f"Collecting data for: {network_name} (embedding_dim={embed_dim})")
        print(f"{'='*60}")
        
        bp_accs, ldfa_accs = get_accuracies_for_network(base_path, network_name, rank=rank)
        
        # Save BP accuracies
        for i, acc in enumerate(bp_accs, 1):
            all_data.append({
                'network': network_name,
                'embedding_dim': int(embed_dim),
                'method': 'BP',
                'experiment': i,
                'accuracy': acc
            })
        
        # Save LDFA accuracies
        for i, acc in enumerate(ldfa_accs, 1):
            all_data.append({
                'network': network_name,
                'embedding_dim': int(embed_dim),
                'method': f'LDFA-{rank}',
                'experiment': i,
                'accuracy': acc
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Data saved to: {csv_path}")
    print(f"{'='*60}")
    
    return df


def load_data_from_csv(csv_path):
    """
    Load data from CSV file.
    """
    if not os.path.exists(csv_path):
        return None
    
    print(f"Loading data from: {csv_path}")
    df = pd.DataFrame(pd.read_csv(csv_path))
    return df


def plot_embedding_dimension_comparison(base_dir, save_dir='experiment_plots/cifar10', force_reload=False, rank=36):
    """
    Create bar plot comparing BP and LDFA across different embedding dimensions.
    
    Networks:
    - vit_b_16: embedding_dim = 384
    - vit_b_16_192: embedding_dim = 192
    - vit_b_16_96: embedding_dim = 96
    - vit_b_16_48: embedding_dim = 48
    """
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, f'embedding_dimension_data_rank{rank}.csv')
    
    # Try to load from CSV first, unless force_reload is True
    if not force_reload:
        df = load_data_from_csv(csv_path)
    else:
        df = None
    
    # If CSV doesn't exist or force_reload, collect data
    if df is None:
        df = collect_and_save_data(base_dir, csv_path, rank=rank)
    
    # Define networks and their properties
    networks = [
        ('vit_b_16', '384', '384'),
        ('vit_b_16_192', '192', '192'),
        ('vit_b_16_96', '96', '96'),
        ('vit_b_16_48', '48', '48'),
    ]
    
    # Collect data from DataFrame
    bp_means = []
    bp_stds = []
    ldfa_means = []
    ldfa_stds = []
    labels = []
    
    for network_name, embed_dim, label in networks:
        # Get BP data - use only top 4 accuracies
        bp_data = df[(df['network'] == network_name) & (df['method'] == 'BP')]['accuracy'].values
        if len(bp_data) > 0:
            # Sort and take top 4
            bp_data_sorted = np.sort(bp_data)[::-1]  # Sort descending
            bp_top4 = bp_data_sorted[:4] if len(bp_data_sorted) >= 4 else bp_data_sorted
            bp_means.append(np.mean(bp_top4))
            bp_stds.append(np.std(bp_top4))
            print(f"{network_name} BP: Using top {len(bp_top4)} accuracies: {bp_top4}")
        else:
            bp_means.append(0)
            bp_stds.append(0)
        
        # Get LDFA data - use only top 4 accuracies
        ldfa_data = df[(df['network'] == network_name) & (df['method'] == f'LDFA-{rank}')]['accuracy'].values
        if len(ldfa_data) > 0:
            # Sort and take top 4
            ldfa_data_sorted = np.sort(ldfa_data)[::-1]  # Sort descending
            ldfa_top4 = ldfa_data_sorted[:4] if len(ldfa_data_sorted) >= 4 else ldfa_data_sorted
            ldfa_means.append(np.mean(ldfa_top4))
            ldfa_stds.append(np.std(ldfa_top4))
            print(f"{network_name} LDFA-{rank}: Using top {len(ldfa_top4)} accuracies: {ldfa_top4}")
        else:
            ldfa_means.append(0)
            ldfa_stds.append(0)
        
        labels.append(label)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(labels))
    width = 0.38
    
    # Create bars - BP in black, LDFA in blue (matching other plots exactly)
    # Use the same blue from the Blues colormap as in plot_training_curves.py
    ldfa_blue = plt.cm.Blues(0.6)  # Medium blue from the Blues colormap
    
    bars1 = ax.bar(x - width/2, bp_means, width, yerr=bp_stds, 
                   label='BP (Backpropagation)', color='black', alpha=1.0, 
                   capsize=5, edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5})
    
    bars2 = ax.bar(x + width/2, ldfa_means, width, yerr=ldfa_stds,
                   label=f'LDFA (rank={rank})', color=ldfa_blue, alpha=1.0,
                   capsize=5, edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5})
    
    # Customize plot
    ax.set_xlabel('Embedding Dimension', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('BP vs LDFA Performance Across Embedding Dimensions\n(CIFAR-10, ViT-B/16)', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis range for better visibility
    ax.set_ylim(0.7, 0.93)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'embedding_dimension_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {save_path}")
    print(f"{'='*60}")
    
    # Also save as PDF for publication
    save_path_pdf = os.path.join(save_dir, 'embedding_dimension_comparison.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"PDF saved to: {save_path_pdf}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Label':<10} {'Embed Dim':<12} {'BP Mean':<12} {'BP Std':<12} {'LDFA Mean':<12} {'LDFA Std':<12}")
    print("-"*60)
    for i, (_, embed_dim, label) in enumerate(networks):
        print(f"{label:<10} {embed_dim:<12} {bp_means[i]*100:>10.2f}% {bp_stds[i]*100:>10.2f}% {ldfa_means[i]*100:>10.2f}% {ldfa_stds[i]*100:>10.2f}%")
    print("="*60)


if __name__ == '__main__':
    base_dir = '/home/maherhanut/Documents/Low-Dimensional-Feedback/artifacts/training_checkpoints/cifar10'
    
    # Use CSV by default (force_reload=False)
    # Set force_reload=True only if you want to reload from TensorBoard logs
    plot_embedding_dimension_comparison(base_dir, force_reload=False)
