#!/bin/bash
# This is a shell script to run a Python file with different configurations.
# It executes each command sequentially.

echo "--- Starting batch run ---"

# --- Run 1 ---
echo "=> Running with configuration 'config_A.json'..."
python vit_training/train_vit.py --config configs/train_vit_config.yaml

# --- Run 2 ---
echo "=> Running with configuration 'config_B.json'..."
python vit_training/train_vit.py --config configs/train_vit_LDFA_config.yaml

echo "--- Batch run finished ---"