import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import sys
import yaml
import argparse
import json
import hashlib

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
            curr_kwargs = dict(kwargs)
            if 'rank' in curr_kwargs and 'qkv' in name:
                curr_kwargs['rank'] = curr_kwargs['rank'] * 3  # Triple rank for QKV layers (matches training)
            new_linear = new_linear_cls(in_features, out_features, **curr_kwargs, bias=bias)
            setattr(module, name, new_linear)
        else:
            replace_linear(child, new_linear_cls, **kwargs)


def measure_flops(model, x, y, n_iters=5):
    # Forward FLOPs
    with torch.no_grad():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    with_flops=True) as prof:
            for _ in range(n_iters):
                _ = model(x)
    fwd_flops = sum([evt.flops for evt in prof.key_averages() if hasattr(evt, 'flops') and evt.flops is not None]) / n_iters

    # Backward FLOPs — recompute graph each iteration to avoid retain_graph memory accumulation
    x_fp32 = x.float().requires_grad_(True)  # bfloat16 doesn't support grad on input; use float32 proxy
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                with_flops=True) as prof:
        for _ in range(n_iters):
            out = model(x_fp32.to(x.dtype))
            loss = (out * y).sum() if isinstance(out, torch.Tensor) else (out.logits * y).sum()
            loss.backward()
            model.zero_grad()
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
    
    dtype = torch.float32 if dtype_str == 'float32' else (torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16)
    
    # Prepare input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
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


def plot_flops_accuracy_combined(convergence_csv, accuracy_csv, config_path, output_dir='experiment_plots', suffix='top1'):
    """
    Create combined plots showing FLOPs to convergence and accuracy
    Generates two versions: bars for FLOPs + line for accuracy, and vice versa
    """
    
    # Load data
    convergence_df = pd.read_csv(convergence_csv)
    accuracy_df = pd.read_csv(accuracy_csv)
    
    # Extract ranks (excluding BP for now), convert to int
    ldfa_ranks = [int(r) for r in convergence_df['Rank'].values if r != 'BP' and str(r).isdigit()]
    
    # Get FLOPs per epoch for each rank — use cache if available
    print("\n=== Measuring FLOPs ===")
    # Cache key: hash of config file contents + ranks list
    with open(config_path, 'r') as f:
        config_text = f.read()
    cache_key = hashlib.md5((config_text + str(sorted(ldfa_ranks))).encode()).hexdigest()[:12]
    cache_file = os.path.join(output_dir, f'flops_cache_{cache_key}.json')

    if os.path.exists(cache_file):
        print(f"Loading cached FLOPs from {cache_file}")
        with open(cache_file, 'r') as f:
            flops_per_batch = {int(k) if k.isdigit() else k: v for k, v in json.load(f).items()}
    else:
        flops_per_batch = get_flops_per_epoch(config_path, ldfa_ranks)
        with open(cache_file, 'w') as f:
            json.dump(flops_per_batch, f, indent=2)
        print(f"FLOPs cached to {cache_file}")
    
    # Load config to get batches_per_epoch and actual training batch size
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # batches_per_epoch is based on the real training batch size (not the benchmarking batch size)
    batches_per_epoch  = config.get('batches_per_epoch', 196)
    bench_batch_size   = config.get('batch_size', 256)
    train_batch_size   = config.get('train_batch_size', bench_batch_size)
    batch_scale        = train_batch_size / bench_batch_size  # scale FLOPs to training batch size

    # Calculate FLOPs per epoch for each configuration (scaled to actual training batch size)
    flops_per_epoch = {}
    for rank, flops_batch in flops_per_batch.items():
        flops_per_epoch[rank] = flops_batch * batch_scale * batches_per_epoch  # GFLOPs per epoch
        print(f"{rank}: {flops_per_epoch[rank]:.2f} GFLOPs per epoch ({batches_per_epoch} batches)")
    
    # Calculate total FLOPs to convergence
    # FLOPs to convergence = epochs_to_90% * FLOPs_per_epoch
    flops_to_convergence = {}
    flops_to_convergence_std = {}
    flops_to_convergence_sem = {}
    steps_to_convergence = {}
    steps_to_convergence_std = {}
    accuracies_top1 = {}
    accuracies_top1_std = {}
    accuracies_top2 = {}
    accuracies_top2_std = {}
    
    for _, row in convergence_df.iterrows():
        rank_raw = row['Rank']
        # Convert rank to int if it's a digit, otherwise keep as string (for 'BP')
        rank = int(rank_raw) if str(rank_raw).isdigit() else rank_raw
        epochs = row['Mean_Steps_to_90pct']  # These are actually epochs, not batches
        epochs_std = row.get('Std_Steps_to_90pct', 0.0)  # Get std if available
        steps_to_convergence[rank] = epochs
        steps_to_convergence_std[rank] = epochs_std
        
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
        accuracies_top1_std[rank] = acc_row.get('Std_Top1_Acc', 0.0)  # Get std if available
        accuracies_top2[rank] = acc_row['Mean_Top2_Acc']
        accuracies_top2_std[rank] = acc_row.get('Std_Top2_Acc', 0.0)  # Get std if available
        
        # Calculate FLOPs to convergence (in TFLOPs)
        # Total FLOPs = epochs × batches_per_epoch × FLOPs_per_batch
        # Std FLOPs = epochs_std × batches_per_epoch × FLOPs_per_batch
        # SEM FLOPs = Std FLOPs / sqrt(n_experiments)
        if rank in flops_per_batch:
            flops_to_convergence[rank] = (epochs * flops_per_epoch[rank]) / 1e6  # Convert to PFLOPs
            flops_to_convergence_std[rank] = (epochs_std * flops_per_epoch[rank]) / 1e6
            flops_to_convergence_sem[rank] = flops_to_convergence_std[rank] / np.sqrt(5)
        else:
            flops_to_convergence[rank] = (epochs * flops_per_epoch['BP']) / 1e6
            flops_to_convergence_std[rank] = (epochs_std * flops_per_epoch['BP']) / 1e6
            flops_to_convergence_sem[rank] = flops_to_convergence_std[rank] / np.sqrt(5)
    
    # Prepare data for plotting
    # Reverse order: BP first, then 64, 36, 32, 24, 20, 16, 10
    ranks_sorted = ['BP'] + sorted([r for r in ldfa_ranks if isinstance(r, int)], reverse=True)
    flops_values = [flops_to_convergence[r] for r in ranks_sorted]
    flops_std_values = [flops_to_convergence_std[r] for r in ranks_sorted]
    flops_sem_values = [flops_to_convergence_sem[r] for r in ranks_sorted]
    acc_top1_values = [accuracies_top1[r] for r in ranks_sorted]
    acc_top1_std_values = [accuracies_top1_std[r] for r in ranks_sorted]
    acc_top2_values = [accuracies_top2[r] for r in ranks_sorted]
    acc_top2_std_values = [accuracies_top2_std[r] for r in ranks_sorted]
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
            'Std_Accuracy_Top1': f"{accuracies_top1_std[rank]:.4f}",
            'Mean_Accuracy_Top2': f"{acc_top2_values[i]:.4f}",
            'Std_Accuracy_Top2': f"{accuracies_top2_std[rank]:.4f}",
            'Mean_Epochs_to_90pct': f"{steps_to_convergence[rank]:.1f}",
            'Std_Epochs_to_90pct': f"{steps_to_convergence_std[rank]:.1f}",
            'Total_FLOPs_to_Convergence_PFLOPs': f"{flops_values[i]:.2f}",
            'FLOPs_Reduction_Percentage': f"{flops_reduction_pct[i]:.2f}"
        })
    
    flops_summary_df = pd.DataFrame(flops_summary_data)
    flops_summary_file = f'{output_dir}/flops_convergence_summary.csv'
    flops_summary_df.to_csv(flops_summary_file, index=False)
    print(f"\nFLOPs summary saved to: {flops_summary_file}")
    print("\nFLOPs Summary:")
    print(flops_summary_df.to_string(index=False))
    
    # Auto-compute axis limits from data
    acc_margin_bottom = 0.07
    acc_margin_top    = 0.01   # extra room for text annotations
    flops_margin      = 2.5   # 10% padding each side
    acc_ylim = [
        min(acc_top1_values) - acc_margin_bottom,
        max(acc_top1_values) + acc_margin_top,
    ]
    flops_span   = max(flops_values) - min(flops_values)
    flops_pad    = max(flops_span * flops_margin, max(flops_values) * 0.05)
    flops_ylim   = [min(flops_values) - flops_pad, max(flops_values) + flops_pad]

    # Generate colors: Black for BP, blues gradient for LDFA (lighter to darker as rank decreases)
    ldfa_count = len(ldfa_ranks)
    blues = plt.cm.Blues(np.linspace(0.35, 0.85, ldfa_count))[::-1]
    bar_colors = ["#000000"] + [blues[i] for i in range(ldfa_count)]

    print("\n=== Creating plot ===")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Rank', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Top-1 Accuracy', fontsize=16, fontweight='bold')
    ax1.bar(rank_labels, acc_top1_values, yerr=acc_top1_std_values,
            color=bar_colors, alpha=0.7, capsize=5,
            error_kw={'elinewidth': 2, 'capthick': 2},
            label='Top-1 Accuracy')
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.set_ylim(acc_ylim)

    ax2 = ax1.twinx()
    color_line = '#B34700'
    ax2.set_ylabel('Computational Cost (PFLOPs)', fontsize=16, fontweight='bold', color=color_line)
    ax2.errorbar(rank_labels, flops_values, yerr=flops_sem_values,
                 color=color_line, marker='s', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, elinewidth=1.5,
                 label='FLOPs to Convergence')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_ylim(flops_ylim)

    # Annotate accuracy value above each bar only
    for i, (acc_val, acc_std) in enumerate(zip(acc_top1_values, acc_top1_std_values)):
        ax1.text(i, acc_val + acc_std + (acc_ylim[1] - acc_ylim[0]) * 0.01, f'{acc_val:.2%}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['png', 'svg', 'pdf']:
        plt.savefig(f'{output_dir}/flops_accuracy_{suffix}.{ext}',
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    print(f"Saved: {output_dir}/flops_accuracy_{suffix}.png/svg/pdf")
    plt.close()


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
    parser.add_argument('--suffix', type=str,
                        default='top1',
                        help='Suffix for output filenames (e.g. 90pct_last or to_max)')
    args = parser.parse_args()

    plot_flops_accuracy_combined(
        args.convergence_csv,
        args.accuracy_csv,
        args.config,
        args.output_dir,
        args.suffix,
    )
