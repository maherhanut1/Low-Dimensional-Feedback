import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re
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


def extract_rank_from_alt_format(task_name):
    """Extract rank from alternative naming format: cifar100_lr<lr>_wd<wd>_bs_<bs>_LDFA_<rank>"""
    pattern = r'_LDFA_(\d+)$'
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


def collect_data_for_subset(base_path, subset_name, use_alt_format=False):
    """
    Collect BP and LDFA data for a specific subset.
    
    Args:
        base_path: Path to base directory
        subset_name: Name of subset directory
        use_alt_format: If True, use alternative naming format (cifar100_lr<lr>_...)
    
    Returns:
        bp_accuracies: List of BP accuracies
        ldfa_data: Dict mapping rank to list of accuracies
    """
    subset_path = base_path / subset_name
    if not subset_path.exists():
        print(f"Warning: Subset path not found: {subset_path}")
        return [], {}
    
    bp_accuracies = []
    ldfa_data = {}  # rank -> list of accuracies
    
    # Pattern to match experiment folders
    pattern = re.compile(r'^(.+)_exp_(\d+)$')
    
    for folder in subset_path.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                task_name = match.group(1)
                
                # Check if it's BP
                if is_bp_task(task_name):
                    try:
                        accuracy = load_tensorboard_data(str(folder))
                        if accuracy is not None:
                            bp_accuracies.append(accuracy)
                            print(f"  Loaded BP accuracy: {accuracy:.4f} from {folder.name}")
                    except Exception as e:
                        print(f"  Error loading {folder.name}: {e}")
                
                # Check if it's LDFA with rank
                else:
                    # Try both naming formats
                    rank = None
                    if use_alt_format:
                        rank = extract_rank_from_alt_format(task_name)
                    else:
                        rank = extract_rank_from_task(task_name)
                    
                    if rank is not None:
                        try:
                            accuracy = load_tensorboard_data(str(folder))
                            if accuracy is not None:
                                if rank not in ldfa_data:
                                    ldfa_data[rank] = []
                                ldfa_data[rank].append(accuracy)
                                print(f"  Loaded LDFA-{rank} accuracy: {accuracy:.4f} from {folder.name}")
                        except Exception as e:
                            print(f"  Error loading {folder.name}: {e}")
    
    return bp_accuracies, ldfa_data


def plot_accuracy_vs_rank(base_dir, save_dir='experiment_plots/cifar100'):
    """
    Plot accuracy as a function of rank for different CIFAR-100 subsets.
    
    Creates a single plot with:
    - Three lines for LDFA with different subsets (10, 20, 50 classes)
    - BP results as horizontal lines for each subset
    """
    base_path = Path(base_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define subsets
    subsets = [
        ('vit_b_16_subsets10', '10 classes', '#1f77b4', False),  # Blue
        ('vit_b_16_subsets50', '50 classes', '#2ca02c', False),  # Green
        ('vit_s_8_layers_384_fix', '100 classes', '#d62728', True),  # Red, use alt format
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    all_data = {}
    
    # Collect data for each subset
    for subset_name, label, color, use_alt_format in subsets:
        print(f"\n{'='*60}")
        print(f"Processing subset: {subset_name}")
        print(f"{'='*60}")
        
        bp_accs, ldfa_data = collect_data_for_subset(base_path, subset_name, use_alt_format)
        
        all_data[subset_name] = {
            'label': label,
            'color': color,
            'bp_accs': bp_accs,
            'ldfa_data': ldfa_data
        }
        
        # Calculate BP mean and std
        if bp_accs:
            bp_mean = np.mean(bp_accs)
            bp_std = np.std(bp_accs)
            print(f"BP: Mean={bp_mean:.4f}, Std={bp_std:.4f}, N={len(bp_accs)}")
        
        # Print LDFA summary
        if ldfa_data:
            print(f"LDFA ranks found: {sorted(ldfa_data.keys())}")
            for rank in sorted(ldfa_data.keys()):
                accs = ldfa_data[rank]
                print(f"  Rank {rank}: Mean={np.mean(accs):.4f}, Std={np.std(accs):.4f}, N={len(accs)}")
    
    # Plot data
    for subset_name, label, color, use_alt_format in subsets:
        data = all_data[subset_name]
        bp_accs = data['bp_accs']
        ldfa_data = data['ldfa_data']
        
        # Plot BP as horizontal line
        if bp_accs:
            bp_mean = np.mean(bp_accs)
            ax.axhline(y=bp_mean, color='black', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'BP ({label})' if subset_name == subsets[0][0] else '')
        
        # Plot LDFA accuracy vs rank
        if ldfa_data:
            ranks = sorted(ldfa_data.keys())
            means = [np.mean(ldfa_data[r]) for r in ranks]
            stds = [np.std(ldfa_data[r]) for r in ranks]
            
            # Plot line with error bars
            ax.errorbar(ranks, means, yerr=stds, marker='o', markersize=8,
                       linewidth=2.5, capsize=5, capthick=2,
                       label=f'LDFA ({label})', color=color, alpha=0.85)
    
    # Customize plot
    ax.set_xlabel('Rank', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('LDFA Performance vs Rank for CIFAR-100 Subsets', 
                 fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis limits for better visibility
    ax.set_ylim(0.5, 1.0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'cifar100_subsets_rank_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {save_path}")
    print(f"{'='*60}")
    
    # Also save as PDF for publication
    save_path_pdf = os.path.join(save_dir, 'cifar100_subsets_rank_comparison.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"PDF saved to: {save_path_pdf}")
    
    plt.show()


if __name__ == '__main__':
    base_dir = '/home/maherhanut/Documents/Low-Dimensional-Feedback/artifacts/training_checkpoints/cifar100'
    plot_accuracy_vs_rank(base_dir)
