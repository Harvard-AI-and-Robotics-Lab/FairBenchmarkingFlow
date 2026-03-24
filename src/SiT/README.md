# Scalable Interpolant Transformers (SiT)

## Installation

```bash
conda env create -f environment.yml
conda activate SiT
```

## Weights Download

Run from the `src/SiT/` directory:

```bash
mkdir -p pre_trained && cd pre_trained
wget "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0" -O SiT-XL-2-256x256.pt
```

The expected checkpoint path is `pre_trained/SiT-XL-2-256x256.pt`.

## Evaluation

Before running any script, set the following variables at the top of each script:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to ImageNet or ImageNetV2 dataset root |
| `CKPT_PATH` | Path to downloaded checkpoint (e.g. `pre_trained/SiT-XL-2-256x256.pt`) |

### ImageNet (`run_inference.sh`)

Evaluates on ImageNet validation set (FID-50k). Run from `src/SiT/`:

```bash
# Set DATA_PATH and CKPT_PATH in the script, then:
bash run_inference.sh
```

Runs 25-step generation with CFG 7.0 and natural guidance.

### ImageNetV2 (`run_inference_imagenetv2.sh`)

Evaluates on ImageNetV2 (~10k images). Run from `src/SiT/`:

```bash
# Set DATA_PATH=/path/to/ImageNetV2 in the script, then:
bash run_inference_imagenetv2.sh
```

### RelationNet (`run_inference_relaionet.sh`)

Evaluates on the RelationNet benchmark. Run from `src/SiT/`:

```bash
# Set DATA_PATH in the script, then:
bash run_inference_relaionet.sh
```
