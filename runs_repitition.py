import os
import yaml
import subprocess
import argparse
from pathlib import Path
import tempfile

def run_config_multiple_times(config_path, num_runs=5):
    """Run a single config multiple times with different log names."""
    
    # Load the original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    original_log_name = config.get('log_name', 'default_run')
    
    for i in range(num_runs):
        # Modify log name for this run
        config['log_name'] = f"{original_log_name}_exp_{i+1}"
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config, tmp_file, default_flow_style=False)
            temp_config_path = tmp_file.name
        
        try:
            print(f"Running experiment {i+1}/5 for {config_path}")
            print(f"Log name: {config['log_name']}")
            
            # Run the training script
            result = subprocess.run([
                'python', 'vit_training/train_resnet18_2.py',
                '--config', temp_config_path
            ])
            
            print(f"Experiment {i+1} completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"Experiment {i+1} failed with error:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_config_path)

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments from config files')
    parser.add_argument('--config_dir', type=str, required=True, 
                       help='Directory containing config files')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of times to run each config')
    parser.add_argument('--pattern', type=str, default='*.yaml',
                       help='Pattern to match config files')
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Find all config files
    config_files = list(config_dir.glob(args.pattern))
    
    if not config_files:
        print(f"No config files found in {config_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(config_files)} config files")
    print(f"Each will be run {args.num_runs} times")
    
    for config_file in sorted(config_files):
        print(f"\n{'='*60}")
        print(f"Processing: {config_file.name}")
        print(f"{'='*60}")
        
        run_config_multiple_times(config_file, args.num_runs)
    
    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()