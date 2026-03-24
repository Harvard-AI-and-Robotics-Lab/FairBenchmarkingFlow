"""Sanity check: load a dataloader, pick 10 random images, and plot with class names.

Usage:
    # ImageNet
    python src/utils/sanity_check.py --dataset imagenet --data_path /path/to/imagenet --split val

    # MS-COCO
    python src/utils/sanity_check.py --dataset coco --data_path /path/to/coco/images --data_mapping data/mappings/coco.json
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running as: python src/utils/sanity_check.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.dataloader.imagenet import ImageNetDataset
from src.utils.dataloader.mscoco import MSCOCODataset

NUM_SAMPLES = 10


def load_dataset(args):
    """Instantiate the requested dataset with return_class_name=True and uint8 output."""
    if args.dataset == "imagenet":
        return ImageNetDataset(
            data_path=args.data_path,
            split=args.split,
            image_size=256,
            return_uint8=True,
            return_class_name=True,
        )
    elif args.dataset == "coco":
        if args.data_mapping is None:
            raise ValueError("--data_mapping is required for the coco dataset.")
        return MSCOCODataset(
            data_path=args.data_path,
            data_mapping=args.data_mapping,
            image_size=256,
            return_uint8=True,
            return_class_name=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def plot_samples(dataset):
    """Select 10 random images from the dataset and plot them with class names."""
    rng = np.random.RandomState()
    indices = rng.choice(len(dataset), size=min(NUM_SAMPLES, len(dataset)), replace=False)

    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 7))
    fig.suptitle(f"Sanity Check — {len(dataset)} images in dataset", fontsize=14)

    for i, idx in enumerate(indices):
        img, class_name = dataset[idx]
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.set_title(class_name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Dataloader sanity check.")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["imagenet", "coco"],
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--data_mapping", type=str, default=None,
        help="Path to the mapping JSON (required for coco).",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="ImageNet split to use: train, val, or all (default: val).",
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} dataset from {args.data_path} ...")
    dataset = load_dataset(args)
    print(f"Loaded {len(dataset)} images, {dataset.num_classes} classes.")

    plot_samples(dataset)


if __name__ == "__main__":
    main()
