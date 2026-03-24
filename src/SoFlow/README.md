# SoFlow: Solution Flow Models

## Installation

```bash
conda create -n soflow python=3.10 -y
conda activate soflow
pip install -r requirements_soflow.txt
pip install "transformers>=4.38"
conda install -c conda-forge git-lfs
```

**Note:** If GPU parallel inference fails due to missing cufile library, set:
```bash
# export LD_LIBRARY_PATH=/path/to/nvidia/cufile/lib:$LD_LIBRARY_PATH  # set if needed
```

## Weights Download

Run from the `src/SoFlow/` directory:

```bash
mkdir -p weights && cd weights
git clone https://huggingface.co/zlab-princeton/SoFlow
cd SoFlow

# Pull specific model (e.g. XL-2-cond)
git lfs pull --include="XL-2-cond/**"
```

Verify download:
```bash
ls -lh XL-2-cond/ckpts/
```

The expected checkpoint path is `weights/SoFlow/XL-2-cond`.

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `CHECKPOINT_PATH` | Path to downloaded weights folder (e.g. `weights/SoFlow/XL-2-cond`) |

The script auto-discovers the YAML config and `ckpts/*.pt` checkpoint from the weights directory.

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k). Run from `src/SoFlow/`:

```bash
# Set DATA_PATH and CHECKPOINT_PATH in the script, then:
bash run_inference.sh
```

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/SoFlow/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelationNet benchmark. Run from `src/SoFlow/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
