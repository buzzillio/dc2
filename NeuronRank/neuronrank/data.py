"""Dataset loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .utils.seed import worker_init_fn


@dataclass
class DatasetBundle:
    """Container for loaders used in experiments."""

    train: DataLoader
    calibration: DataLoader
    eval: DataLoader
    num_classes: int


class HFImageDataset(Dataset):
    """Wrap a Hugging Face vision dataset in a PyTorch Dataset."""

    def __init__(self, hf_dataset, transform) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __getitem__(self, idx: int):  # type: ignore[override]
        item = self.dataset[idx]
        image = item["img"] if "img" in item else item["image"]
        if self.transform is not None:
            image = self.transform(image)
        label = item["label"]
        return image, label


def build_cifar10_loaders(
    batch_size: int,
    calib_size: int,
    num_workers: int,
    seed: int,
) -> DatasetBundle:
    """Return loaders for CIFAR-10 using HF datasets."""

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = load_dataset("cifar10")
    train_dataset = HFImageDataset(dataset["train"], transform_train)
    test_dataset = HFImageDataset(dataset["test"], transform_test)

    generator = torch.Generator()
    generator.manual_seed(seed)
    calib_indices = torch.randperm(len(train_dataset), generator=generator)[:calib_size]
    calib_dataset = Subset(train_dataset, calib_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return DatasetBundle(train_loader, calib_loader, eval_loader, num_classes=10)


def build_imagenet_loaders(
    val_dir: str,
    batch_size: int,
    calib_size: int,
    num_workers: int,
    seed: int,
) -> DatasetBundle:
    """ImageNet loaders based on torchvision dataset API."""

    from torchvision.datasets import ImageFolder

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )

    val_dataset = ImageFolder(val_dir, transform=transform)

    if calib_size > len(val_dataset):
        raise ValueError("Calibration size cannot exceed number of validation examples")

    generator = torch.Generator()
    generator.manual_seed(seed)
    calib_indices = torch.randperm(len(val_dataset), generator=generator)[:calib_size]
    calib_dataset = Subset(val_dataset, calib_indices.tolist())

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    calib_loader = DataLoader(calib_dataset, shuffle=False, **loader_kwargs)
    eval_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    # For simplicity we fine-tune on calibration subset when ImageNet val only
    train_loader = DataLoader(calib_dataset, shuffle=True, **loader_kwargs)
    return DatasetBundle(train_loader, calib_loader, eval_loader, num_classes=1000)


def get_dataset(
    dataset: str,
    batch_size: int,
    calib_size: int,
    num_workers: int,
    seed: int,
    imagenet_val: str | None = None,
) -> DatasetBundle:
    """Factory dispatch for dataset loaders."""

    dataset = dataset.lower()
    if dataset == "cifar10":
        return build_cifar10_loaders(batch_size, calib_size, num_workers, seed)
    if dataset == "imagenet":
        if not imagenet_val:
            raise ValueError("--imagenet-val must be provided for ImageNet")

        val_path = Path(imagenet_val).expanduser()
        if not val_path.exists():
            raise FileNotFoundError(
                f"ImageNet validation directory not found: {val_path}"
            )
        if not any(val_path.iterdir()):
            raise FileNotFoundError(
                f"ImageNet validation directory is empty: {val_path}"
            )

        return build_imagenet_loaders(
            str(val_path), batch_size, calib_size, num_workers, seed
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
