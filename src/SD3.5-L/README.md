# Stable Diffusion 3.5 Large

## Installation

```bash
conda create -n sd35 python=3.10 -y
conda activate sd35

# Install PyTorch (adjust cu124 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install SD 3.5 dependencies
pip install diffusers transformers accelerate safetensors sentencepiece protobuf

# Install eval dependencies
pip install torchmetrics torch-fidelity open_clip_torch image-reward pillow scipy

# HuggingFace login (requires access to stabilityai/stable-diffusion-3.5-large)
pip install huggingface_hub
export HF_TOKEN="your_token_here"
```

## Weights Download

Weights are downloaded automatically from HuggingFace on first run via `diffusers`. Ensure `HF_TOKEN` is set with access to `stabilityai/stable-diffusion-3.5-large`.

## Evaluation

Before running any script, set the following variable at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `NUM_SHARDS` | Number of parallel GPU shards (default: 4) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k) using 25-step generation with guidance 3.5. Run from `src/SD3.5-L/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference.sh
```

The script generates images in parallel shards, combines them, then computes metrics. Outputs go to `outputs/sd35_natural_25steps/`.

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/SD3.5-L/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelAIONet benchmark. Run from `src/SD3.5-L/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
