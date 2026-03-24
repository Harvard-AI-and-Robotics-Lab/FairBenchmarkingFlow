# Scale-RAE: Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders

## Installation

```bash
conda create -n scale_rae python=3.10 -y
conda activate scale_rae
pip install -e .
pip install torchmetrics clean-fid image-reward torch-fidelity pick-score
```

## Weights Download

Models and decoders are downloaded automatically from HuggingFace on first run. The default model is `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B`.

To pre-download manually, run from `src/Scale-RAE/`:

```bash
# Download model weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B', local_dir='models')"

# Download decoder
python -c "from huggingface_hub import snapshot_download; snapshot_download('nyu-visionx/siglip2_decoder', local_dir='decoder')"
```

The eval scripts expect:
- `MODEL_PATH=models` — path to downloaded model directory
- `DECODER_PATH=decoder/model.pt` — path to decoder weights

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `MODEL_PATH` | Path to downloaded model directory (e.g. `models`) |
| `DECODER_PATH` | Path to decoder weights (e.g. `decoder/model.pt`) |
| `NUM_GPUS` | Number of GPUs to use (default: 4) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k) using parallel GPU shards. Run from `src/Scale-RAE/`:

```bash
# Set DATA_PATH, MODEL_PATH, DECODER_PATH in the script, then:
bash run_inference.sh
```

Runs 25-step generation with natural guidance (1.42).

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/Scale-RAE/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelationNet benchmark. Run from `src/Scale-RAE/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
