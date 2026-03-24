"""ImageNetV2 DataLoader (matched-frequency variant).

Returns (image, class_index) pairs where class indices are the integer folder
names (0–999) directly — no synset mapping required.

Expected directory structure:
    data_path/
      imagenetv2-matched-frequency-format-val/
        0/
          <uuid>.jpeg
          ...
        1/
          ...
        999/
          ...

Usage:
    from utils.dataloader import ImageNetV2Dataset

    dataset = ImageNetV2Dataset(
        data_path='/path/to/ImageNetV2',
    )
    image, class_idx = dataset[0]
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Reuse the class-name list from the ImageNet dataloader
from .imagenet import _CLASS_NAMES, IMAGE_EXTENSIONS, FIXED_SAMPLE_SEED, NUM_FIXED_SAMPLES

V2_SUBDIR = "imagenetv2-matched-frequency-format-val"


class ImageNetV2Dataset(Dataset):
    """ImageNetV2 matched-frequency dataset.

    Args:
        data_path: Path to the ImageNetV2 root directory (containing
            imagenetv2-matched-frequency-format-val/).
        image_size: Target image size (images are resized to image_size x image_size).
            Default: 256.
        transform: Optional custom transform. If None, uses the default pipeline:
            Resize(256x256) -> ToTensor -> Normalize to [-1, 1].
        return_uint8: If True, return uint8 HWC numpy arrays instead of
            normalized tensors. Useful for FID reference images.
        return_class_name: If True, return the class name string instead of the
            integer class index. Default: False.
    """

    def __init__(self, data_path, image_size=256,
                 transform=None, return_uint8=False, return_class_name=False):
        self.data_path = data_path
        self.image_size = image_size
        self.return_uint8 = return_uint8
        self.return_class_name = return_class_name
        self._class_names = list(_CLASS_NAMES)
        self._num_classes = len(_CLASS_NAMES)

        v2_dir = os.path.join(data_path, V2_SUBDIR)
        if not os.path.isdir(v2_dir):
            raise RuntimeError(
                f"Expected '{V2_SUBDIR}/' under {data_path}. "
                "Please point --data_path at the ImageNetV2 root directory."
            )

        # Collect (image_path, class_index) samples
        self.samples = []
        for entry in sorted(os.listdir(v2_dir)):
            class_dir = os.path.join(v2_dir, entry)
            if not os.path.isdir(class_dir):
                continue
            if not entry.isdigit():
                continue
            class_idx = int(entry)
            if class_idx < 0 or class_idx >= 1000:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {v2_dir}. "
                "Expected structure: imagenetv2-matched-frequency-format-val/<class_idx>/<image>.jpeg"
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


def create_imagenetv2_dataloader(data_path, batch_size=32, image_size=256,
                                  num_workers=4, shuffle=False,
                                  return_uint8=False, return_class_name=False):
    """Create a PyTorch DataLoader for ImageNetV2 matched-frequency.

    Args:
        data_path: Path to ImageNetV2 root directory.
        batch_size: Batch size.
        image_size: Target image size (default: 256).
        num_workers: DataLoader workers (default: 4).
        shuffle: Whether to shuffle (default: False).
        return_uint8: If True, returns uint8 HWC numpy arrays.
        return_class_name: If True, return class name instead of index.

    Returns:
        (dataloader, dataset) tuple.
    """
    dataset = ImageNetV2Dataset(
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
        pin_memory=True,
    )

    return dataloader, dataset
