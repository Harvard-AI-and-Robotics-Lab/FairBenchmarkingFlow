# MeanFlow

## Installation

```bash
conda create -n meanflow python=3.10
conda activate meanflow
bash source/install.sh
```

## Weights Download

Run from the `src/meanflow/` directory:

```bash
mkdir -p weights && cd weights
gdown --fuzzy https://drive.google.com/file/d/19MR1WLycyqc627gsOJF4yzR4ZkU8fvOu/view
```

Then unzip the downloaded archive. The expected checkpoint path is `weights/checkpoint_1200960/`.

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `CHECKPOINT_PATH` | Path to weights folder (e.g. `weights/checkpoint_1200960/`) |
| `NUM_GPUS` | Number of GPUs to use (default: 4) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k). Run from `src/meanflow/`:

```bash
# Set DATA_PATH and CHECKPOINT_PATH in the script, then:
bash run_inference.sh
```

Runs 1-step generation with natural guidance (omega=1.0).

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/meanflow/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelationNet benchmark. Run from `src/meanflow/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
