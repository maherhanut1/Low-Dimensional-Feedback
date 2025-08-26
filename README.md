

# Refactored LDFA Project

## Project Overview

This repository is dedicated to research and development of efficient low-rank linear layers (LDFA_Linear) and their integration into modern deep learning architectures, such as Vision Transformers (ViT). The project aims to provide novel modules, tools, and experiments for advancing scalable and memory-efficient neural network training.

## Benchmarking (Component)

As part of this project, a benchmarking suite is provided to evaluate and compare the computational cost (FLOPs) of LDFA_Linear and standard BP_Linear layers within ViT models. This enables rigorous analysis of efficiency and scalability for different configurations.

### How to Run the ViT Benchmark

The benchmarking script is located at:

	benchmarking/plot_vit_flops.py

This script supports configuration via a YAML file, allowing for reproducible and easily modifiable experiment setups.

#### Usage Example


To run the ViT benchmarking script with a configuration file, make sure your current directory is the project root and set the `PYTHONPATH` so Python can find all modules:

```bash
export PYTHONPATH=.
python benchmarking/plot_vit_flops.py --config configs/benchmarking_configs/vit_benchmarking_configs.yaml
```

#### Configuration Parameters

The YAML configuration file supports the following parameters:

| Parameter     | Type         | Description                                                      | Default                        |
|-------------- |--------------|------------------------------------------------------------------|--------------------------------|
| `model`       | `str`        | ViT model variant (`vit_b_16` or `vit_h_14`)                     | `vit_b_16`                     |
| `batch_size`  | `int`        | Batch size for benchmarking                                      | `32`                           |
| `image_size`  | `int`        | Input image size (pixels)                                        | `224`                          |
| `num_classes` | `int`        | Number of output classes                                         | `1000`                         |
| `ranks`       | `list[int]`  | List of ranks to benchmark for LDFA_Linear                       | `[4,8,16,32,64,128,256,512]`   |
| `device`      | `str`        | Device to run on (`cpu` or `cuda`)                               | `cpu`                          |
| `dtype`       | `str`        | Data type (`float32` or `float16`)                               | `float32`                      |
| `n_iters`     | `int`        | Number of iterations to average FLOPs measurements               | `3`                            |

##### Example YAML Configuration

```yaml
model: vit_b_16
batch_size: 32
image_size: 224
num_classes: 1000
ranks: [4, 8, 16, 32, 64, 128, 256, 512]
```

You may create multiple YAML config files for different experiments and specify the desired one with the `--config` argument.

#### Output

The script will print the measured forward and backward FLOPs for both BP_Linear and LDFA_Linear layers, and generate plots visualizing the results and efficiency gains.

---

For further details, refer to the source code in `benchmarking/plot_vit_flops.py` or contact the maintainers.
