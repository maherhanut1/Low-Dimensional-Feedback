from pathlib import Path
import subprocess


if __name__ == "__main__":
    config_dir = Path("configs/tests")
    for config_file in config_dir.glob("*.yaml"):
        print(f"Running training with config: {config_file}")
        subprocess.run(["python", "vit_training/train_vit.py", "--config", str(config_file)])