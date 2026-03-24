"""ImageNet DataLoader using standard folder structure.

Returns (image, class_index) pairs where class indices are derived from the
synset folder names (e.g. n01440764/) mapped via imagenet_classes.json.
Images are resized to 256x256.

Expected directory structure (either convention works):
    imagenet/
      train/            (or imagenet_train/)
        n01440764/
          ILSVRC2012_train_00000001.JPEG
          ...
        n01443537/
          ...
      val/              (or imagenet_val/)
        n01440764/
          ...

Usage:
    from utils.dataloader import ImageNetDataset

    dataset = ImageNetDataset(
        data_path='/path/to/imagenet',
        split='val',
    )
    image, class_idx = dataset[0]
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

FIXED_SAMPLE_SEED = 42
NUM_FIXED_SAMPLES = 10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Load synset -> index mapping from the shared imagenet_classes.json
_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "classifiers" / "siglip_classifier" / "imagenet_classes.json"
)


def _load_synset_to_idx() -> tuple[dict[str, int], list[str]]:
    """Build synset_id -> class_index mapping and ordered class names list."""
    with open(_JSON_PATH, "r") as f:
        raw: dict[str, list[str]] = json.load(f)
    synset_to_idx = {}
    class_names = [""] * len(raw)
    for idx_str, (synset_id, class_name) in raw.items():
        idx = int(idx_str)
        synset_to_idx[synset_id] = idx
        class_names[idx] = class_name
    return synset_to_idx, class_names


_SYNSET_TO_IDX, _CLASS_NAMES = _load_synset_to_idx()


class ImageNetDataset(Dataset):
    """ImageNet dataset reading from the standard synset folder structure.

    Args:
        data_path: Path to the ImageNet root directory (containing train/ and/or val/).
        split: Which split to load -- "train", "val", or "all" (default: "val").
        image_size: Target image size (images are resized to image_size x image_size).
            Default: 256.
        transform: Optional custom transform. If None, uses the default pipeline:
            Resize(256x256) -> ToTensor -> Normalize to [-1, 1].
        return_uint8: If True, return uint8 HWC numpy arrays instead of
            normalized tensors. Useful for FID reference images.
        return_class_name: If True, return the class name string instead of the
            integer class index. Default: False.
    """

    def __init__(self, data_path, split="val", image_size=256,
                 transform=None, return_uint8=False, return_class_name=False):
        self.data_path = data_path
        self.image_size = image_size
        self.return_uint8 = return_uint8
        self.return_class_name = return_class_name
        self._class_names = list(_CLASS_NAMES)
        self._num_classes = len(_CLASS_NAMES)

        # Determine which split directories to scan
        if split == "all":
            splits = ["train", "val"]
        else:
            splits = [split]

        # Collect (image_path, class_index) samples
        self.samples = []
        for s in splits:
            split_dir = os.path.join(data_path, s)
            if not os.path.isdir(split_dir):
                # Also check imagenet_train / imagenet_val naming convention
                alt_dir = os.path.join(data_path, f"imagenet_{s}")
                if os.path.isdir(alt_dir):
                    split_dir = alt_dir
                else:
                    continue

            # Check if split_dir has synset subdirectories or flat images
            entries = sorted(os.listdir(split_dir))
            has_synset_dirs = any(
                os.path.isdir(os.path.join(split_dir, e)) and e in _SYNSET_TO_IDX
                for e in entries[:50]  # check first 50 entries
            )

            if has_synset_dirs:
                # Standard structure: split_dir/synset_id/image.JPEG
                for synset_id in entries:
                    synset_dir = os.path.join(split_dir, synset_id)
                    if not os.path.isdir(synset_dir):
                        continue
                    if synset_id not in _SYNSET_TO_IDX:
                        continue
                    class_idx = _SYNSET_TO_IDX[synset_id]
                    for fname in sorted(os.listdir(synset_dir)):
                        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                            self.samples.append(
                                (os.path.join(synset_dir, fname), class_idx)
                            )
            else:
                # Flat structure: split_dir/image.JPEG (no class labels)
                for fname in entries:
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                        self.samples.append(
                            (os.path.join(split_dir, fname), -1)
                        )

        if not self.samples:
            raise RuntimeError(
                f"No images found under {data_path} for split '{split}'. "
                "Expected structure: data_path/{train,val}/synset_id/image.JPEG "
                "or data_path/{train,val}/image.JPEG"
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
        """Return class names in index order."""
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


def create_imagenet_dataloader(data_path, split="val", batch_size=32,
                               image_size=256, num_workers=4, shuffle=False,
                               return_uint8=False, return_class_name=False):
    """Create a PyTorch DataLoader for ImageNet.

    Args:
        data_path: Path to ImageNet root directory.
        split: "train", "val", or "all" (default: "val").
        batch_size: Batch size.
        image_size: Target image size (default: 256).
        num_workers: DataLoader workers (default: 4).
        shuffle: Whether to shuffle (default: False).
        return_uint8: If True, returns uint8 HWC numpy arrays.
        return_class_name: If True, return class name instead of index.

    Returns:
        (dataloader, dataset) tuple.
    """
    dataset = ImageNetDataset(
        data_path=data_path,
        split=split,
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
        pin_memory=True,
    )

    return dataloader, dataset
