import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import sys
import os
import yaml
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.opt_layers.LDFA_Linear import Linear as LDFA_Linear
from modules.opt_layers.BP_Linear import Linear as BP_Linear


def replace_linear(module, new_linear_cls, **kwargs):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            new_linear = new_linear_cls(in_features, out_features, **kwargs, bias=bias)
            setattr(module, name, new_linear)
        else:
            replace_linear(child, new_linear_cls, **kwargs)


def measure_flops(model, x, y, n_iters=5):
    # Forward FLOPs
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        for _ in range(n_iters):
            out = model(x)
    fwd_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None]) / n_iters
    
    # Backward FLOPs
    out = model(x)
    loss = (out * y).sum() if isinstance(out, torch.Tensor) else (out.logits * y).sum()
    model.zero_grad()
    x.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        for _ in range(n_iters):
            loss.backward(retain_graph=True)
    bwd_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None]) / n_iters
    
    return fwd_flops, bwd_flops


def get_flops_per_epoch(config_path, ranks_list):
    """Measure FLOPs for BP and each LDFA rank"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = config.get('batch_size', 32)
    image_size = config.get('image_size', 224)
    num_classes = config.get('num_classes', 1000)
    device = config.get('device', 'cpu')
    dtype_str = config.get('dtype', 'float32')
    n_iters = config.get('n_iters', 3)
    
    # Model parameters
    patch_size = config.get('patch_size', 16)
    in_chans = config.get('in_chans', 3)
    embed_dim = config.get('embed_dim', 768)
    depth = config.get('depth', 12)
    num_heads = config.get('num_heads', 12)
    mlp_ratio = config.get('mlp_ratio', 4.0)
    qkv_bias = config.get('qkv_bias', True)
    drop_rate = config.get('drop_rate', 0.0)
    attn_drop_rate = config.get('attn_drop_rate', 0.0)
    drop_path_rate = config.get('drop_path_rate', 0.0)
    
    dtype = torch.float32 if dtype_str == 'float32' else torch.float16
    
    # Prepare input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(batch_size, num_classes, device=device, dtype=dtype)
    
    flops_dict = {}
    
    # Measure BP
    print("Measuring BP FLOPs...")
    model_bp = VisionTransformer(
        img_size=image_size, patch_size=patch_size, in_chans=in_chans,
        num_classes=num_classes, embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate
    )
    replace_linear(model_bp, BP_Linear)
    model_bp = model_bp.to(device=device, dtype=dtype)
    fwd_bp, bwd_bp = measure_flops(model_bp, x, y, n_iters=n_iters)
    flops_dict['BP'] = (fwd_bp + bwd_bp) / 1e9  # Total FLOPs in GFLOPs
    print(f"BP: {flops_dict['BP']:.2f} GFLOPs per batch")
    del model_bp
    
    # Measure LDFA for each rank
    for rank in ranks_list:
        print(f"Measuring LDFA rank {rank} FLOPs...")
        model_ldfa = VisionTransformer(
            img_size=image_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        replace_linear(model_ldfa, LDFA_Linear, rank=rank)
        model_ldfa = model_ldfa.to(device=device, dtype=dtype)
        fwd_ldfa, bwd_ldfa = measure_flops(model_ldfa, x, y, n_iters=n_iters)
        flops_dict[rank] = (fwd_ldfa + bwd_ldfa) / 1e9  # Total FLOPs in GFLOPs
        print(f"LDFA rank {rank}: {flops_dict[rank]:.2f} GFLOPs per batch")
        del model_ldfa
    
    return flops_dict


def plot_flops_accuracy_combined(convergence_csv, accuracy_csv, config_path, output_dir='experiment_plots'):
    """
    Create combined plots showing FLOPs to convergence and accuracy
    Generates two versions: bars for FLOPs + line for accuracy, and vice versa
    """
    
    # Load data
    convergence_df = pd.read_csv(convergence_csv)
    accuracy_df = pd.read_csv(accuracy_csv)
    
    # Extract ranks (excluding BP for now), convert to int
    ldfa_ranks = [int(r) for r in convergence_df['Rank'].values if r != 'BP' and str(r).isdigit()]
    
    # Get FLOPs per epoch for each rank
    print("\n=== Measuring FLOPs ===")
    flops_per_batch = get_flops_per_epoch(config_path, ldfa_ranks)
    
    # Calculate batches per epoch (CIFAR-10: 50000 images / 256 batch size = 195.3125 ≈ 196 batches)
    batches_per_epoch = 196  # for CIFAR-10 with batch_size=256
    
    # Calculate FLOPs per epoch for each configuration
    flops_per_epoch = {}
    for rank, flops_batch in flops_per_batch.items():
        flops_per_epoch[rank] = flops_batch * batches_per_epoch  # GFLOPs per epoch
        print(f"{rank}: {flops_per_epoch[rank]:.2f} GFLOPs per epoch ({batches_per_epoch} batches)")
    
    # Calculate total FLOPs to convergence
    # FLOPs to convergence = epochs_to_90% * FLOPs_per_epoch
    flops_to_convergence = {}
    steps_to_convergence = {}
    accuracies_top1 = {}
    accuracies_top2 = {}
    
    for _, row in convergence_df.iterrows():
        rank_raw = row['Rank']
        # Convert rank to int if it's a digit, otherwise keep as string (for 'BP')
        rank = int(rank_raw) if str(rank_raw).isdigit() else rank_raw
        epochs = row['Mean_Steps_to_90pct']  # These are actually epochs, not batches
        steps_to_convergence[rank] = epochs
        
        # Get corresponding accuracy
        acc_row = accuracy_df[accuracy_df['Rank'] == rank_raw].iloc[0]
        accuracies_top1[rank] = acc_row['Mean_Accuracy_Top1']
        accuracies_top2[rank] = acc_row['Mean_Accuracy_Top2']
        
        # Calculate FLOPs to convergence (in TFLOPs)
        # Total FLOPs = epochs × batches_per_epoch × FLOPs_per_batch
        if rank in flops_per_batch:
            flops_to_convergence[rank] = (epochs * flops_per_epoch[rank]) / 1000  # Convert to TFLOPs
        else:
            flops_to_convergence[rank] = (epochs * flops_per_epoch['BP']) / 1000  # Use BP FLOPs
    
    # Prepare data for plotting
    # Reverse order: BP first, then 64, 36, 32, 24, 20, 16, 10
    ranks_sorted = ['BP'] + sorted([r for r in ldfa_ranks if isinstance(r, int)], reverse=True)
    flops_values = [flops_to_convergence[r] for r in ranks_sorted]
    acc_top1_values = [accuracies_top1[r] for r in ranks_sorted]
    acc_top2_values = [accuracies_top2[r] for r in ranks_sorted]
    rank_labels = [str(r) for r in ranks_sorted]
    
    # Calculate FLOPs reduction percentage (relative to BP)
    bp_flops = flops_to_convergence['BP']
    flops_reduction_pct = []
    for rank in ranks_sorted:
        if rank == 'BP':
            flops_reduction_pct.append(0.0)  # BP is baseline
        else:
            reduction = ((bp_flops - flops_to_convergence[rank]) / bp_flops) * 100
            flops_reduction_pct.append(reduction)
    
    # Save FLOPs summary to CSV
    flops_summary_data = []
    for i, rank in enumerate(ranks_sorted):
        flops_summary_data.append({
            'Rank': rank,
            'Mean_Accuracy_Top1': f"{acc_top1_values[i]:.4f}",
            'Mean_Accuracy_Top2': f"{acc_top2_values[i]:.4f}",
            'Total_FLOPs_to_Convergence_TFLOPs': f"{flops_values[i]:.2f}",
            'FLOPs_Reduction_Percentage': f"{flops_reduction_pct[i]:.2f}",
            'Epochs_to_90pct': f"{steps_to_convergence[rank]:.1f}"
        })
    
    flops_summary_df = pd.DataFrame(flops_summary_data)
    flops_summary_file = f'{output_dir}/flops_convergence_summary.csv'
    flops_summary_df.to_csv(flops_summary_file, index=False)
    print(f"\nFLOPs summary saved to: {flops_summary_file}")
    print("\nFLOPs Summary:")
    print(flops_summary_df.to_string(index=False))
    
    # Generate colors: Black for BP, blues gradient for LDFA (lighter to darker as rank decreases)
    ldfa_count = len(ldfa_ranks)
    blues = plt.cm.Blues(np.linspace(0.35, 0.85, ldfa_count))[::-1]  # Reverse so darker is for higher ranks
    bar_colors = ['#000000'] + [blues[i] for i in range(ldfa_count)]  # Pure black for BP
    
    # Version 1: Bars for FLOPs, Line for Accuracy
    print("\n=== Creating plots ===")
    
    # Top-1 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    bars = ax1.bar(rank_labels, flops_values, color=bar_colors, alpha=0.7, label='FLOPs to Convergence')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    
    # Second y-axis for accuracy
    ax2 = ax1.twinx()
    color_line = 'darkred'
    ax2.set_ylabel('Top-1 Accuracy', color=color_line, fontsize=12)
    line = ax2.plot(rank_labels, acc_top1_values, color=color_line, marker='o', 
                    linewidth=2, markersize=8, label='Top-1 Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    # Set y-axis limits to make line visible above bars
    ax2.set_ylim([0.85, 0.95])
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: FLOPs, Line: Accuracy)', fontsize=14)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flops_accuracy_bars_flops_line_acc_top1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/flops_accuracy_bars_flops_line_acc_top1.svg', bbox_inches='tight')
    print(f"Saved: {output_dir}/flops_accuracy_bars_flops_line_acc_top1.png/svg")
    plt.close()
    
    # Top-2 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    bars = ax1.bar(rank_labels, flops_values, color=bar_colors, alpha=0.7, label='FLOPs to Convergence')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Top-2 Accuracy', color=color_line, fontsize=12)
    line = ax2.plot(rank_labels, acc_top2_values, color=color_line, marker='o', 
                    linewidth=2, markersize=8, label='Top-2 Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim([0.96, 0.98])
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: FLOPs, Line: Accuracy)', fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flops_accuracy_bars_flops_line_acc_top2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/flops_accuracy_bars_flops_line_acc_top2.svg', bbox_inches='tight')
    print(f"Saved: {output_dir}/flops_accuracy_bars_flops_line_acc_top2.png/svg")
    plt.close()
    
    # Version 2: Bars for Accuracy, Line for FLOPs
    
    # Top-1 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Top-1 Accuracy', fontsize=12)
    bars = ax1.bar(rank_labels, acc_top1_values, color=bar_colors, alpha=0.7, label='Top-1 Accuracy')
    ax1.tick_params(axis='y')
    ax1.set_ylim([0.85, 0.95])
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_line = 'steelblue'
    ax2.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    line = ax2.plot(rank_labels, flops_values, color=color_line, marker='s', 
                    linewidth=2, markersize=8, label='FLOPs to Convergence')
    ax2.tick_params(axis='y')
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: Accuracy, Line: FLOPs)', fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flops_accuracy_bars_acc_line_flops_top1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/flops_accuracy_bars_acc_line_flops_top1.svg', bbox_inches='tight')
    print(f"Saved: {output_dir}/flops_accuracy_bars_acc_line_flops_top1.png/svg")
    plt.close()
    
    # Top-2 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Top-2 Accuracy', fontsize=12)
    bars = ax1.bar(rank_labels, acc_top2_values, color=bar_colors, alpha=0.7, label='Top-2 Accuracy')
    ax1.tick_params(axis='y')
    ax1.set_ylim([0.96, 0.98])
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_line = 'steelblue'
    ax2.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    line = ax2.plot(rank_labels, flops_values, color=color_line, marker='s', 
                    linewidth=2, markersize=8, label='FLOPs to Convergence')
    ax2.tick_params(axis='y')
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: Accuracy, Line: FLOPs)', fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flops_accuracy_bars_acc_line_flops_top2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/flops_accuracy_bars_acc_line_flops_top2.svg', bbox_inches='tight')
    print(f"Saved: {output_dir}/flops_accuracy_bars_acc_line_flops_top2.png/svg")
    plt.close()
    
    print("\n=== All plots created successfully ===")
    print("Generated 4 versions:")
    print("1. Bars: FLOPs, Line: Accuracy (Top-1 and Top-2)")
    print("2. Bars: Accuracy, Line: FLOPs (Top-1 and Top-2)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot combined FLOPs and accuracy analysis')
    parser.add_argument('--convergence_csv', type=str, 
                        default='experiment_plots/convergence_summary.csv',
                        help='Path to convergence summary CSV file')
    parser.add_argument('--accuracy_csv', type=str, 
                        default='experiment_plots/rank_accuracy_summary.csv',
                        help='Path to accuracy summary CSV file')
    parser.add_argument('--config', type=str, 
                        default='configs/benchmarking_configs/vit_benchmarking_configs.yaml',
                        help='Path to benchmarking config YAML file')
    parser.add_argument('--output_dir', type=str, 
                        default='experiment_plots',
                        help='Directory to save output plots')
    args = parser.parse_args()
    
    plot_flops_accuracy_combined(
        args.convergence_csv, 
        args.accuracy_csv, 
        args.config,
        args.output_dir
    )
