# Flux.1-dev

## Installation

```bash
conda create -n flux2 python=3.10 -y
conda activate flux2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate sentencepiece protobuf
pip install torchmetrics torch-fidelity open_clip_torch image-reward pillow scipy huggingface_hub
```

## Weights Download

Weights are downloaded automatically from HuggingFace on first run via `diffusers`. The model requires access to `black-forest-labs/FLUX.1-dev`:

```bash
export HF_TOKEN="your_token_here"
```

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `NUM_SHARDS` | Number of parallel GPU shards (default: 4) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k) using parallel GPU shards. Run from `src/flux1/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference.sh
```

Runs 25-step generation with CFG 7.0. The script generates images in parallel shards then computes metrics.

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/flux1/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelAIONet benchmark. Run from `src/flux1/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
