"""reLAIONet DataLoader (CLIP-filtered web-scraped ImageNet subset).

Returns (image, imagenet_class_idx) pairs. Class indices are derived from
the folder structure: each subdirectory under images/ is an ImageNet class
name (or synset ID), mapped to the standard 0-999 integer index.

Expected directory structure:
    data_path/
      reLAIONet-cleaned-v1/
        images/
          {class_name}/        ← folder name is the ImageNet class name
            *.png / *.jpg / ...

Usage:
    from utils.dataloader import ReLAIONetDataset

    dataset = ReLAIONetDataset(
        data_path='/path/to/data',  # contains reLAIONet-cleaned-v1/
    )
    image, class_idx = dataset[0]
"""

import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from torch.utils.data import DataLoader, Dataset

# Some PNG files in reLAIONet have oversized iCCP (ICC profile) chunks that
# exceed PIL's default MAX_TEXT_CHUNK limit, causing a ValueError on open.
# Raise the limit to accommodate these files.
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)  # 100 MB
from torchvision import transforms

# Reuse the class-name list and synset mapping from the ImageNet dataloader
from .imagenet import _CLASS_NAMES, _SYNSET_TO_IDX, IMAGE_EXTENSIONS, FIXED_SAMPLE_SEED, NUM_FIXED_SAMPLES

RELAIONET_SUBDIR = "reLAIONet-cleaned-v1"


class ReLAIONetDataset(Dataset):
    """reLAIONet matched-frequency dataset.

    Args:
        data_path: Path to the root directory containing reLAIONet-cleaned-v1/.
        image_size: Target image size (images are resized to image_size x image_size).
            Default: 256.
        transform: Optional custom transform. If None, uses the default pipeline:
            Resize(256x256) -> ToTensor -> Normalize to [-1, 1].
        return_uint8: If True, return uint8 HWC numpy arrays instead of
            normalized tensors. Useful for FID reference images.
        return_class_name: If True, return the ImageNet class name string instead
            of the integer class index. Default: False.
    """

    def __init__(self, data_path, image_size=256,
                 transform=None, return_uint8=False, return_class_name=False):
        self.data_path = data_path
        self.image_size = image_size
        self.return_uint8 = return_uint8
        self.return_class_name = return_class_name
        self._class_names = list(_CLASS_NAMES)
        self._num_classes = len(_CLASS_NAMES)

        # Accept both: a path that already ends with reLAIONet-cleaned-v1/
        # and a parent directory that contains reLAIONet-cleaned-v1/.
        if os.path.basename(os.path.normpath(data_path)) == RELAIONET_SUBDIR:
            root_dir = data_path
        else:
            root_dir = os.path.join(data_path, RELAIONET_SUBDIR)
        if not os.path.isdir(root_dir):
            raise RuntimeError(
                f"Expected '{RELAIONET_SUBDIR}/' under {data_path}. "
                "Please point --data_path at the directory containing reLAIONet-cleaned-v1/."
            )

        # Build name -> class_idx lookup.
        # Supports both human-readable class names (e.g. "golden_retriever")
        # and synset IDs (e.g. "n02099601") as folder names.
        # Normalise: lowercase, hyphens -> underscores, apostrophes -> underscores
        # so that e.g. "carpenter's_kit" matches folder "carpenter_s_kit".
        def _norm(s):
            return s.lower().replace('-', '_').replace("'", '_')

        _name_to_idx = {_norm(name): idx for idx, name in enumerate(_CLASS_NAMES) if name}
        _name_to_idx.update({_norm(k): v for k, v in _SYNSET_TO_IDX.items()})

        # Images live under root_dir/images/ or directly under root_dir.
        images_dir = os.path.join(root_dir, "images")
        if not os.path.isdir(images_dir):
            images_dir = root_dir  # fall back to scanning root directly

        # Walk one level of class folders and collect image files.
        self.samples = []
        unrecognized = []
        for class_dir in sorted(Path(images_dir).iterdir()):
            if not class_dir.is_dir():
                continue
            folder_name = class_dir.name
            class_idx = _name_to_idx.get(_norm(folder_name))
            if class_idx is None:
                unrecognized.append(folder_name)
                continue
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(img_file), class_idx))

        if unrecognized:
            print(f"[ReLAIONet] {len(unrecognized)} folder(s) could not be matched to an ImageNet class:")
            for name in unrecognized:
                print(f"  {name}")

        if not self.samples:
            raise RuntimeError(
                f"No valid images found under {images_dir}. "
                "Ensure subdirectory names match ImageNet class names or synset IDs."
            )

        # Default transform
        if transform is not None:
            self.transform = transform
        elif return_uint8:
            self.transform = None
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        # Pre-select fixed sample indices for reproducible visualization
        rng = np.random.RandomState(FIXED_SAMPLE_SEED)
        self._fixed_indices = sorted(
            rng.choice(len(self.samples),
                       size=min(NUM_FIXED_SAMPLES, len(self.samples)),
                       replace=False)
        )

    @property
    def fixed_sample_indices(self):
        """Return 10 deterministic sample indices for reproducible visualization."""
        return list(self._fixed_indices)

    @property
    def class_names(self):
        """Return standard ImageNet class names in index order (0–999)."""
        return list(self._class_names)

    @property
    def num_classes(self):
        return self._num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.return_class_name:
            label = self._class_names[label]

        if self.return_uint8:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            return np.array(img, dtype=np.uint8), label

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def create_relaionet_dataloader(data_path, batch_size=32, image_size=256,
                                 num_workers=4, shuffle=False,
                                 return_uint8=False, return_class_name=False):
    """Create a PyTorch DataLoader for reLAIONet.

    Args:
        data_path: Path to directory containing reLAIONet-cleaned-v1/.
        batch_size: Batch size.
        image_size: Target image size (default: 256).
        num_workers: DataLoader workers (default: 4).
        shuffle: Whether to shuffle (default: False).
        return_uint8: If True, returns uint8 HWC numpy arrays.
        return_class_name: If True, return class name instead of index.

    Returns:
        (dataloader, dataset) tuple.
    """
    dataset = ReLAIONetDataset(
        data_path=data_path,
        image_size=image_size,
        return_uint8=return_uint8,
        return_class_name=return_class_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )

    return dataloader, dataset
