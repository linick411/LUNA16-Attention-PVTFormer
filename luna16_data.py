import os
import csv
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


DEFAULT_DATA_DIR = Path(os.environ.get("LUNA16_TASK_DIR", "data/Task03_lung"))
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _slice_index(path):
    stem = Path(path).stem
    try:
        return (0, int(stem.rsplit("_", 1)[-1]))
    except ValueError:
        return (1, stem)


def _media_by_stem(directory):
    files = [path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    result = {}
    for path in files:
        if path.stem in result:
            raise ValueError(f"Duplicate slice stem under {directory}: {path.stem}")
        result[path.stem] = str(path)
    return result


def _nodule_index(path):
    name = Path(path).name
    try:
        return (0, int(name.split("_", 1)[1]))
    except (IndexError, ValueError):
        return (1, name)


def build_slice_lookup(paths):
    """Index slices by case directory and numeric z order, independent of list order."""
    lookup = {}
    case_numbers = {}
    for position, path in enumerate(paths):
        path = Path(path)
        try:
            number = int(path.stem.rsplit("_", 1)[-1])
        except ValueError as exc:
            raise ValueError(f"2.5D slice name needs a numeric index: {path}") from exc
        key = (path.parent, number)
        if key in lookup:
            raise ValueError(f"Duplicate 2.5D slice index: {path}")
        lookup[key] = position
        case_numbers.setdefault(path.parent, []).append(number)
    for directory, numbers in case_numbers.items():
        ordered = sorted(numbers)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError(f"Missing 2.5D slice index under {directory}")
    return lookup


def neighbor_indices(index, paths, lookup):
    """Return true z-1, z, z+1 within one case; repeat center at boundaries."""
    center = Path(paths[index])
    number = int(center.stem.rsplit("_", 1)[-1])
    return (
        lookup.get((center.parent, number - 1), index),
        index,
        lookup.get((center.parent, number + 1), index),
    )


def load_data(path=DEFAULT_DATA_DIR):
    """Load Task03_lung folders and keep the original split policy."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(
            f"LUNA16 Task03_lung directory not found: {root}. "
            "Set LUNA16_TASK_DIR or place data under data/Task03_lung."
        )

    def get_data(name):
        nodule_dir = root / name
        image_dir = nodule_dir / "images"
        mask_dir = nodule_dir / "masks" / "nodule"
        if not image_dir.exists() or not mask_dir.exists():
            return [], []

        images = _media_by_stem(image_dir)
        masks = _media_by_stem(mask_dir)
        if images.keys() != masks.keys():
            missing_masks = sorted(images.keys() - masks.keys())[:5]
            missing_images = sorted(masks.keys() - images.keys())[:5]
            raise ValueError(
                f"Image/mask filename mismatch under {nodule_dir}. "
                f"Missing masks: {missing_masks}; missing images: {missing_images}"
            )

        stems = sorted(images, key=lambda stem: _slice_index(stem))
        return [images[stem] for stem in stems], [masks[stem] for stem in stems]

    dirs = sorted(
        [item.name for item in root.iterdir() if item.is_dir() and item.name.startswith("nodule_")],
        key=_nodule_index,
    )
    manifest_path = root / "case_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            rows = list(csv.DictReader(manifest_file))
        manifest_names = [row["case_name"] for row in rows]
        if manifest_names != dirs:
            raise ValueError("case_manifest.csv does not exactly match Task03_lung folders")
        invalid_splits = [row for row in rows if row.get("split") not in {"train", "validation", "test"}]
        if invalid_splits:
            raise ValueError("case_manifest.csv contains a missing or invalid split")
        train_names = [row["case_name"] for row in rows if row["split"] == "train"]
        valid_names = [row["case_name"] for row in rows if row["split"] == "validation"]
        test_names = [row["case_name"] for row in rows if row["split"] == "test"]
    else:
        test_names = [f"nodule_{i}" for i in range(0, 30)]
        valid_names = [f"nodule_{i}" for i in range(30, 60)]
        train_names = [item for item in dirs if item not in set(test_names + valid_names)]

    splits = []
    for names in (train_names, valid_names, test_names):
        split_x, split_y = [], []
        for name in names:
            x, y = get_data(name)
            split_x += x
            split_y += y
        splits.append((split_x, split_y))

    return tuple(splits)


class LUNA16SliceDataset(Dataset):
    """2.5D dataset: previous, center, and next CT slices predict the center mask."""

    def __init__(self, images_path, masks_path, size, transform=None, use_25d=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.use_25d = use_25d
        self.n_samples = len(images_path)
        self.slice_lookup = build_slice_lookup(images_path) if use_25d else None

    def _get_neighbor_indices(self, index):
        return neighbor_indices(index, self.images_path, self.slice_lookup)

    @staticmethod
    def _read_gray(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return image

    def __getitem__(self, index):
        if self.use_25d:
            prev_idx, center_idx, next_idx = self._get_neighbor_indices(index)
            slices = [self._read_gray(self.images_path[idx]) for idx in (prev_idx, center_idx, next_idx)]
            image = np.stack(slices, axis=-1)
        else:
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {self.images_path[index]}")

        mask = self._read_gray(self.masks_path[index])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.use_25d:
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=1)
        else:
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.transpose(image, (2, 0, 1))

        image = image.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        return image, mask

    def __len__(self):
        return self.n_samples
