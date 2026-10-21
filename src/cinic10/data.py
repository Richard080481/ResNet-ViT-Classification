"""CINIC-10 dataset validation, transforms, and data loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)


@dataclass(frozen=True)
class DataBundle:
    train: DataLoader
    valid: DataLoader
    test: DataLoader
    class_names: list[str]


def build_transform(model_name: str, augment: bool = False) -> transforms.Compose:
    """Return preprocessing appropriate for ResNet or pretrained ViT."""
    operations: list[object] = []
    if model_name == "vit":
        operations.append(transforms.Resize((224, 224), antialias=True))
    if augment:
        operations.extend(
            [
                transforms.RandomCrop(224 if model_name == "vit" else 32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
        ]
    )
    return transforms.Compose(operations)


def make_dataloaders(
    data_dir: str | Path,
    model_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = False,
    pin_memory: bool = True,
) -> DataBundle:
    root = Path(data_dir)
    missing = [str(root / split) for split in ("train", "valid", "test") if not (root / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            "CINIC-10 is not in the expected ImageFolder layout. Missing: "
            + ", ".join(missing)
        )

    eval_transform = build_transform(model_name, augment=False)
    train_transform = build_transform(model_name, augment=augment)
    datasets_by_split = {
        "train": datasets.ImageFolder(root / "train", transform=train_transform),
        "valid": datasets.ImageFolder(root / "valid", transform=eval_transform),
        "test": datasets.ImageFolder(root / "test", transform=eval_transform),
    }

    expected_mapping = datasets_by_split["train"].class_to_idx
    for split in ("valid", "test"):
        if datasets_by_split[split].class_to_idx != expected_mapping:
            raise ValueError(f"Class-to-index mapping for {split} does not match train")
    if len(expected_mapping) != 10:
        raise ValueError(f"Expected 10 classes, found {len(expected_mapping)}")

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return DataBundle(
        train=DataLoader(datasets_by_split["train"], shuffle=True, **common),
        valid=DataLoader(datasets_by_split["valid"], shuffle=False, **common),
        test=DataLoader(datasets_by_split["test"], shuffle=False, **common),
        class_names=datasets_by_split["train"].classes,
    )

