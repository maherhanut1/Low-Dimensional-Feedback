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
        
        # Get corresponding accuracy - ranks in accuracy CSV are strings ('10.0', 'BP', etc.)
        if isinstance(rank, int):
            # For LDFA ranks, convert to string with .0
            rank_str = f"{rank}.0"
        else:
            # For BP, keep as string
            rank_str = rank
        
        acc_matches = accuracy_df[accuracy_df['Rank'] == rank_str]
        
        if len(acc_matches) == 0:
            print(f"Warning: No accuracy data found for rank {rank} (looking for '{rank_str}')")
            continue
            
        acc_row = acc_matches.iloc[0]
        accuracies_top1[rank] = acc_row['Mean_Top1_Acc']
        accuracies_top2[rank] = acc_row['Mean_Top2_Acc']
        
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
    bar_colors = ['#000000'] + [blues[i] for i in range(ldfa_count)]  # Pure black (#000000) for BP
    
    # Version 1: Bars for FLOPs, Line for Accuracy
    print("\n=== Creating plots ===")
    
    # Top-1 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    bars = ax1.bar(rank_labels, flops_values, color=bar_colors, alpha=0.7, label='FLOPs to Convergence')
    ax1.tick_params(axis='y')
    
    # Second y-axis for accuracy
    ax2 = ax1.twinx()
    color_line = 'darkred'
    ax2.set_ylabel('Top-1 Accuracy', color=color_line, fontsize=12)
    line = ax2.plot(rank_labels, acc_top1_values, color=color_line, marker='o', 
                    linewidth=2, markersize=8, label='Top-1 Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    # Set y-axis limits to show full range up to 93.5%
    ax2.set_ylim([0.76, 0.935])
    
    # Add text annotations showing FLOPs and Accuracy for each bar
    for i, (rank_label, flops_val, acc_val) in enumerate(zip(rank_labels, flops_values, acc_top1_values)):
        # Print FLOPs above each bar
        ax1.text(i, flops_val + 200, f'{flops_val:.0f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Print Accuracy near each point on the line
        ax2.text(i, acc_val + 0.003, f'{acc_val:.2%}', 
                ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold')
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: FLOPs, Line: Accuracy)', fontsize=14)
    
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
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Top-2 Accuracy', color=color_line, fontsize=12)
    line = ax2.plot(rank_labels, acc_top2_values, color=color_line, marker='o', 
                    linewidth=2, markersize=8, label='Top-2 Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim([0.96, 0.98])
    
    # Add text annotations showing FLOPs and Accuracy for each bar
    for i, (rank_label, flops_val, acc_val) in enumerate(zip(rank_labels, flops_values, acc_top2_values)):
        # Print FLOPs above each bar
        ax1.text(i, flops_val + 200, f'{flops_val:.0f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Print Accuracy near each point on the line
        ax2.text(i, acc_val + 0.0002, f'{acc_val:.2%}', 
                ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold')
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: FLOPs, Line: Accuracy)', fontsize=14)
    
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
    # Scale accuracy axis so that 0.85 (85%) aligns visually with 8000 TFLOPs
    # We want the full accuracy range visible, going up to 0.935 (93.5%)
    # If 0.85 should align with 8000 TFLOPs at 100% of FLOPs axis:
    # Then 0.85 needs to be at: 8000/8000 = 100% height
    # If max accuracy is 0.935, and 0.85 is at 100% relative to FLOPs:
    # We need: (0.85 - acc_min) / (0.935 - acc_min) = 8000 / 8000 = 1.0
    # This means: 0.85 - acc_min = 0.935 - acc_min, which doesn't work
    # Let's think differently: we want 85% to appear at same height as 8000 TFLOPs
    # If FLOPs go 0 to 8000, and we want 85% at the 8000 position:
    # (0.85 - acc_min) / (0.935 - acc_min) = 8000 / 8000 = full height
    # Actually, let's make it proportional:
    # FLOPs span: 0 to 8000 (range = 8000)
    # Accuracy should span: such that 0.85 is at same relative height as 8000
    # If we want accuracy from 0.76 to 0.935 (range = 0.175)
    # Then: (0.85 - 0.76) / (0.935 - 0.76) = 0.09 / 0.175 ≈ 0.514
    # And: 8000 / 8000 = 1.0, so they won't align
    # Better: use range where (0.85 - min) / (max - min) = 8000 / 8000
    # So if max = 0.935, then: 0.85 - min = 0.935 - min is wrong
    # Let's make it: (0.85 - min) = (8000/8000) * (max - min)
    # 0.85 - min = max - min => 0.85 = max (wrong)
    # Actually to align 0.85 with 8000: make their relative positions equal
    # rel_pos_acc = (0.85 - min) / (max - min)
    # rel_pos_flops = 8000 / 8000 = 1.0
    # So: (0.85 - min) / (0.935 - min) = 1.0 => 0.85 - min = 0.935 - min => 0.85 = 0.935 NO
    # 
    # Different approach: What if we want 85% at the SAME absolute height as 8000?
    # If max_acc = 0.935, and we want 85% to visually align with 8000 TFLOPs:
    # (0.85 - min) / (0.935 - min) = 8000 / 8000 = 1.0 is wrong
    # 
    # Correct interpretation: 85% should appear at same pixel height as 8000 TFLOPs
    # FLOPs: 0 to 8000, so 8000 is at top (100% of axis)
    # Accuracy: min_acc to 0.935, and we want 0.85 at 100% too
    # So: min_acc = 0.85 - (0.935 - 0.85) = 0.85 - 0.085 = 0.765
    # Let's use min = 0.76, max = 0.935, and 0.85 will be at (0.85-0.76)/(0.935-0.76) = 0.09/0.175 = 51.4% height
    # But 8000 is at 100% height on FLOPs axis
    #
    # To make them align: we need (0.85 - min_acc) / (max_acc - min_acc) = 8000 / max_flops
    # If max_flops = 8000: (0.85 - min_acc) / (0.935 - min_acc) = 1.0
    # => 0.85 - min_acc = 0.935 - min_acc => impossible
    #
    # Wait, I need to make 8000 appear at same height as 0.85
    # FLOPs axis: 0 to max_flops, where 8000 is somewhere in between
    # Accuracy axis: min_acc to max_acc, where 0.85 is somewhere in between
    # For visual alignment: (8000 - 0) / (max_flops - 0) = (0.85 - min_acc) / (max_acc - min_acc)
    # If we set: max_acc = 0.935, and want to find min_acc such that 8000 aligns with 0.85
    # Assuming max_flops = 8000: 8000/8000 = (0.85 - min_acc)/(0.935 - min_acc)
    # => 1 = (0.85 - min_acc)/(0.935 - min_acc) => 0.935 - min_acc = 0.85 - min_acc => NO
    #
    # OK the issue is if max_flops = 8000 and 8000 should align with something less than max...
    # Let me set max_flops higher. If max_flops = 9500, then:
    # 8000/9500 = (0.85 - min_acc)/(0.935 - min_acc)
    # 0.842 = (0.85 - min_acc)/(0.935 - min_acc)
    # 0.842 * (0.935 - min_acc) = 0.85 - min_acc
    # 0.787 - 0.842*min_acc = 0.85 - min_acc
    # 0.158*min_acc = 0.063
    # min_acc = 0.399 (way too low)
    #
    # Let me try: min_acc = 0.76, max_acc = 0.935, max_flops such that 0.85 aligns with 8000
    # (0.85 - 0.76)/(0.935 - 0.76) = 8000/max_flops
    # 0.09/0.175 = 8000/max_flops
    # max_flops = 8000 * 0.175 / 0.09 = 15555.56
    ax1.set_ylim([0.76, 0.935])
    
    ax2 = ax1.twinx()
    color_line = '#B34700'  # Even darker orange
    ax2.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    line = ax2.plot(rank_labels, flops_values, color=color_line, marker='s', 
                    linewidth=2, markersize=8, label='FLOPs to Convergence')
    ax2.tick_params(axis='y')
    # Set FLOPs scale so 8000 aligns with 0.85 on accuracy axis
    # (0.85 - 0.76) / (0.935 - 0.76) = 8000 / max_flops
    # 0.09 / 0.175 = 8000 / max_flops => max_flops = 15556
    ax2.set_ylim([5000, 8000])
    
    # Add text annotations showing Accuracy and FLOPs for each bar
    for i, (rank_label, acc_val, flops_val) in enumerate(zip(rank_labels, acc_top1_values, flops_values)):
        # Print Accuracy above each bar
        ax1.text(i, acc_val + 0.003, f'{acc_val:.2%}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Print FLOPs beside each point on the line (to the right and slightly below to avoid overlap)
        # Adjust horizontal and vertical offsets based on rank
        if i == 0:  # BP - move more to the right
            horizontal_offset = 0.15
            vertical_offset = 150
        elif i == 1:  # rank 64 - lower
            horizontal_offset = 0.1
            vertical_offset = 190
        elif i == 2:  # rank 36 - lower
            horizontal_offset = 0.1
            vertical_offset = 180
        else:  # 32, 24, 20, 16, 10
            horizontal_offset = 0.1
            vertical_offset = 100
        ax2.text(i + horizontal_offset, flops_val - vertical_offset, f'{flops_val:.0f}', 
                ha='left', va='top', fontsize=10, color='#B34700', fontweight='bold')
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: Accuracy, Line: FLOPs)', fontsize=14)
    
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
    # Scale accuracy axis: show full range up to 98%
    # Using same scaling principle: (0.85 - min) / (max - min) = 8000 / max_flops
    # But for top-2, let's use a natural range showing the data
    ax1.set_ylim([0.96, 0.98])
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_line = '#B34700'  # Even darker orange
    ax2.set_ylabel('Total FLOPs to 90% Convergence (TFLOPs)', fontsize=12)
    line = ax2.plot(rank_labels, flops_values, color=color_line, marker='s', 
                    linewidth=2, markersize=8, label='FLOPs to Convergence')
    ax2.tick_params(axis='y')
    # For consistency, use the same FLOPs scale as top-1
    ax2.set_ylim([0, 15556])
    
    # Add text annotations showing Accuracy and FLOPs for each bar
    for i, (rank_label, acc_val, flops_val) in enumerate(zip(rank_labels, acc_top2_values, flops_values)):
        # Print Accuracy above each bar
        ax1.text(i, acc_val + 0.0002, f'{acc_val:.2%}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Print FLOPs beside each point on the line (to the right and slightly below to avoid overlap)
        # Adjust horizontal and vertical offsets based on rank
        if i == 0:  # BP - move more to the right
            horizontal_offset = 0.15
            vertical_offset = 150
        elif i == 1:  # rank 64 - lower
            horizontal_offset = 0.1
            vertical_offset = 190
        elif i == 2:  # rank 36 - lower
            horizontal_offset = 0.1
            vertical_offset = 180
        else:  # 32, 24, 20, 16, 10
            horizontal_offset = 0.1
            vertical_offset = 100
        ax2.text(i + horizontal_offset, flops_val - vertical_offset, f'{flops_val:.0f}', 
                ha='left', va='top', fontsize=10, color='#B34700', fontweight='bold')
    
    plt.title('FLOPs to Convergence vs Accuracy (Bars: Accuracy, Line: FLOPs)', fontsize=14)
    
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
