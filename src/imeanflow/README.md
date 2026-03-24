# Improved Mean Flows (iMeanFlow)

## Installation

```bash
conda create -n imeanflow python=3.10
conda activate imeanflow
bash scripts/install.sh
```

## Weights Download

Run from the `src/imeanflow/` directory. Create the `weights/` folder and download the desired model:

**B-2 weights**
```bash
mkdir -p weights && cd weights
wget "https://huggingface.co/Lyy0725/iMF/resolve/main/iMF-B-2.zip?download=true" -O iMF-B-2.zip
unzip iMF-B-2.zip && rm iMF-B-2.zip
```

**M-2 weights**
```bash
mkdir -p weights && cd weights
wget "https://huggingface.co/Lyy0725/iMF/resolve/main/iMF-M-2.zip?download=true" -O iMF-M-2.zip
unzip iMF-M-2.zip && rm iMF-M-2.zip
```

**XL-2 weights** (default for eval scripts)
```bash
mkdir -p weights && cd weights
wget "https://huggingface.co/Lyy0725/iMF/resolve/main/iMF-XL-2.zip?download=true" -O iMF-XL-2.zip
unzip iMF-XL-2.zip && rm iMF-XL-2.zip
```

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `CHECKPOINT_PATH` | Path to downloaded weights folder (e.g. `weights/iMF-XL-2`) |
| `NUM_GPUS` | Number of GPUs to use (default: 4) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k). Run from `src/imeanflow/`:

```bash
# Set DATA_PATH and CHECKPOINT_PATH in the script, then:
bash run_inference.sh
```

Runs 1-step generation with the XL-2 model.

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/imeanflow/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelationNet benchmark. Run from `src/imeanflow/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
