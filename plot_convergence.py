import os
import yaml
import time
import subprocess
import shutil
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set the ranks to test and other config
RANKS = [4, 8, 16, 32, 64]
USE_LDFA = True
RUNS_DIR = 'artifacts/training_checkpoints/cifar10/vit16b/'
CONFIG_PATH = 'configs/train_vit_config.yaml'
TRAIN_SCRIPT = 'vit_training/train_vit.py'

# Helper to update YAML config
def update_config(rank, use_ldfa, log_name):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['ldfa_rank'] = rank
    config['use_ldfa_linear'] = use_ldfa
    config['log_name'] = log_name
    with open(CONFIG_PATH, 'w') as f:
        yaml.safe_dump(config, f)

def run_training(rank, use_ldfa, log_name):
    update_config(rank, use_ldfa, log_name)
    start = time.time()
    result = subprocess.run(['python', TRAIN_SCRIPT], capture_output=True)
    end = time.time()
    return end - start, result.returncode, result.stdout.decode(), result.stderr.decode()

def get_tensorboard_scalars(log_dir, tag='eval/metric_accuracy'):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        return [], []
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def main():
    results = []
    for rank in RANKS:
        log_name = f'ldfa_rank{rank}'
        print(f'Running LDFA with rank={rank}...')
        elapsed, code, out, err = run_training(rank, True, log_name)
        log_dir = os.path.join(RUNS_DIR, log_name, 'logs')
        steps, accs = get_tensorboard_scalars(log_dir)
        results.append({'type': 'LDFA', 'rank': rank, 'time': elapsed, 'steps': steps, 'accs': accs})
    # Standard BP
    log_name = 'bp_standard'
    print('Running standard BP...')
    elapsed, code, out, err = run_training(0, False, log_name)
    log_dir = os.path.join(RUNS_DIR, log_name, 'logs')
    steps, accs = get_tensorboard_scalars(log_dir)
    results.append({'type': 'BP', 'rank': None, 'time': elapsed, 'steps': steps, 'accs': accs})

    # Plotting
    plt.figure(figsize=(10,6))
    for r in results:
        label = f"LDFA r={r['rank']}" if r['type']=='LDFA' else 'BP'
        plt.plot(r['steps'], r['accs'], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Convergence: LDFA (various ranks) vs BP')
    plt.legend()
    plt.grid()
    plt.savefig('convergence_plot.png')
    plt.show()

    # Print timing
    for r in results:
        print(f"{r['type']} (rank={r['rank']}): {r['time']:.1f} seconds")

if __name__ == '__main__':
    main()
