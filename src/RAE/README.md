# Diffusion Transformers with Representation Autoencoders (RAE)

## Installation

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# Install PyTorch with CUDA (adjust cu129 to match your CUDA version)
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install other dependencies
uv pip install -r requirements.txt
```

## Weights Download

Run from the `src/RAE/` directory:

```bash
pip install huggingface_hub
hf download nyu-visionx/RAE-collections --local-dir models
```

To download a specific model only:

```bash
hf download nyu-visionx/RAE-collections <remote_model_path> --local-dir models
```

The default eval config uses `configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml`, which references models from the `models/` directory.

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `CONFIG` | Path to sampling config YAML (default: `configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml`) |
| `NUM_GPUS` | Number of GPUs to use (default: 4) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k) using torchrun multi-GPU. Run from `src/RAE/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference.sh
```

Runs 1-step and 25-step generation with CFG 7.0 and natural guidance (CFG 1.42).

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/RAE/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelationNet benchmark. Run from `src/RAE/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
