# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained SiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os
from pathlib import Path


pretrained_models = {'SiT-XL-2-256x256.pt'}
_SIT_DIR = Path(__file__).resolve().parent


def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  
        return download_model(model_name)
    else:
        model_path = Path(model_name).expanduser()
        if not model_path.is_absolute():
            cwd_path = (Path.cwd() / model_path).resolve()
            sit_path = (_SIT_DIR / model_path).resolve()
            if cwd_path.exists():
                model_path = cwd_path
            elif sit_path.exists():
                model_path = sit_path
            else:
                model_path = cwd_path

        if model_path.is_dir():
            candidates = sorted(
                list(model_path.glob("*.pt")) + list(model_path.glob("*.pth")) + list(model_path.glob("*.ckpt")),
                key=lambda p: p.stat().st_mtime,
            )
            assert candidates, f'Could not find SiT checkpoint file in directory: {model_name}'
            model_path = candidates[-1]
            print(f"[find_model] Using checkpoint: {model_path}")

        assert model_path.is_file(), f'Could not find SiT checkpoint at {model_name}'
        checkpoint = torch.load(str(model_path), map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    model_dir = _SIT_DIR / 'pretrained_models'
    local_path = model_dir / model_name
    if not local_path.is_file():
        model_dir.mkdir(parents=True, exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
        download_url(web_path, str(model_dir), filename=model_name)
    model = torch.load(str(local_path), map_location=lambda storage, loc: storage)
    return model
