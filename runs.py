from pathlib import Path
import subprocess


if __name__ == "__main__":
    config_dirs = [Path("configs/cifar10_configs_set2")]
    configs = []
    for config_dir in config_dirs:
        configs.extend(sorted(list(config_dir.glob("*.yaml"))))
    for config_file in configs: 
        print(f"Running training with config: {config_file}")
        subprocess.run(["python", "vit_training/train_vit.py", "--config", str(config_file)])