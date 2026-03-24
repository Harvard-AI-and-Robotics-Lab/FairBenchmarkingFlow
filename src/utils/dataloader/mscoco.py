"""MS-COCO DataLoader using precomputed class mappings from compute_map.py.

Returns (image, class_index) pairs where class indices come from the mapping JSON
produced by the SigLIP2 classifier. Images are resized to 256x256.

The mapping JSON has the structure:
    {
        "data_path": "/absolute/path/to/images",
        "num_classes": 1000,
        "class_names": ["tench", "goldfish", ...],
        "mapping": {
            "train2017/000000000009.jpg": 42,
            "val2017/000000000139.jpg": 15,
            ...
        }
    }

Usage:
    from utils.dataloader import MSCOCODataset

    dataset = MSCOCODataset(
        data_path='/path/to/coco/images',
        data_mapping='data/mappings/coco.json',
    )
    image, class_idx = dataset[0]
"""

import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

FIXED_SAMPLE_SEED = 42
NUM_FIXED_SAMPLES = 10


class MSCOCODataset(Dataset):
    """MS-COCO dataset with class labels from a precomputed mapping file.

    Args:
        data_path: Path to the images base directory. Image keys from the
            mapping file are resolved relative to this path.
        data_mapping: Path to the mapping JSON file produced by compute_map.py.
            Required -- raises ValueError if None.
        image_size: Target image size (images are resized to image_size x image_size).
            Default: 256.
        transform: Optional custom transform. If None, uses the default pipeline:
            Resize(256x256) -> ToTensor -> Normalize to [-1, 1].
        return_uint8: If True, return uint8 HWC numpy arrays instead of
            normalized tensors. Useful for FID reference images.
        return_class_name: If True, return the class name string instead of
            the integer class index. Default: False.
    """

    def __init__(self, data_path, data_mapping=None, image_size=256,
                 transform=None, return_uint8=False, return_class_name=False):
        if data_mapping is None:
            raise ValueError(
                "data_mapping file path is required. "
                "Run compute_map.py first to generate the mapping JSON."
            )

        with open(data_mapping, "r") as f:
            config = json.load(f)

        raw_mapping = config["mapping"]
        self._class_names = config.get("class_names", None)
        self._num_classes = config.get("num_classes", 1000)

        self.data_path = data_path
        self.image_size = image_size
        self.return_uint8 = return_uint8
        self.return_class_name = return_class_name

        # Build samples: only include images that exist on disk
        self.samples = []
        for key in sorted(raw_mapping.keys()):
            img_path = os.path.join(data_path, key)
            if os.path.isfile(img_path):
                self.samples.append((img_path, int(raw_mapping[key])))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {data_path} matching keys in {data_mapping}. "
                "Check that data_path is the correct base directory for the mapping keys."
            )

        # Default transform: resize to target size -> tensor -> normalize to [-1, 1]
        if transform is not None:
            self.transform = transform
        elif return_uint8:
            self.transform = None  # handled in __getitem__
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
        """Return class names in index order (from the mapping JSON)."""
        return list(self._class_names) if self._class_names else None

    @property
    def num_classes(self):
        return self._num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.return_class_name and self._class_names:
            label = self._class_names[label]

        if self.return_uint8:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            return np.array(img, dtype=np.uint8), label

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def create_coco_dataloader(data_path, data_mapping, batch_size, image_size=256,
                           num_workers=4, shuffle=False, return_uint8=False,
                           return_class_name=False):
    """Create a PyTorch DataLoader for MS-COCO using a mapping file.

    Args:
        data_path: Path to images base directory.
        data_mapping: Path to mapping JSON from compute_map.py.
        batch_size: Batch size.
        image_size: Target image size (default: 256).
        num_workers: DataLoader workers (default: 4).
        shuffle: Whether to shuffle (default: False for eval).
        return_uint8: If True, returns uint8 HWC numpy arrays.
        return_class_name: If True, return class name instead of index.

    Returns:
        (dataloader, dataset) tuple.
    """
    dataset = MSCOCODataset(
        data_path=data_path,
        data_mapping=data_mapping,
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
