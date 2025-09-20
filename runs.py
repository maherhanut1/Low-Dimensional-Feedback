from pathlib import Path
import subprocess


if __name__ == "__main__":
    config_dir = Path("configs/expirements_batch")
    configs_tmp = sorted(list(config_dir.glob("*.yaml")))
    configs = [configs_tmp[1], configs_tmp[0], configs_tmp[2]]  # specify the order of configs to run
    for config_file in configs:
        print(f"Running training with config: {config_file}")
        subprocess.run(["python", "vit_training/train_vit.py", "--config", str(config_file)])