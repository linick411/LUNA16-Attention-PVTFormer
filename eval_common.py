import os
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from luna16_data import DEFAULT_DATA_DIR, build_slice_lookup, load_data, neighbor_indices
from metrics import confusion_counts, hd_dist, metrics_from_confusion
from utils import create_dir, seeding


@dataclass
class EvaluationConfig:
    model_name: str
    checkpoint_name: str
    output_dir: Path
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoints_dir: Path = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    image_size: int = int(os.environ.get("IMAGE_SIZE", "192"))
    use_25d: bool = os.environ.get("USE_25D", "1") != "0"
    use_amp: bool = os.environ.get("USE_AMP", "1") != "0"
    seed: int = int(os.environ.get("SEED", "42"))


def _read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _process_prediction(y_pred):
    y_pred = y_pred[0].detach().cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = (y_pred > 0.5).astype(np.uint8) * 255
    y_pred = np.expand_dims(y_pred, axis=-1)
    return np.concatenate([y_pred, y_pred, y_pred], axis=2)


def _print_score(summary):
    print(
        f"Jaccard: {summary['jaccard']:1.4f} - F1: {summary['f1']:1.4f} - "
        f"Recall: {summary['recall']:1.4f} - Precision: {summary['precision']:1.4f} - "
        f"Acc: {summary['accuracy']:1.4f} - F2: {summary['f2']:1.4f} - "
        f"HD: {summary['hausdorff_pixels'] if summary['hausdorff_pixels'] is not None else 'N/A'} - "
        f"AUC: {summary['auc'] if summary['auc'] is not None else 'N/A'}"
    )


def _update_diagnostics(y_true, y_score, valid, diagnostics):
    y_true = np.asarray(y_true) > 0.5
    y_score = np.asarray(y_score, dtype=np.float32)
    valid = np.asarray(valid) > 0.5
    y_pred = y_score > 0.5

    diagnostics["counts"] += confusion_counts(y_true, y_pred, valid_mask=valid)
    true_valid = y_true & valid
    pred_valid = y_pred & valid
    has_true = bool(np.any(true_valid))
    has_pred = bool(np.any(pred_valid))

    if has_true:
        diagnostics["positive_slices"] += 1
    else:
        diagnostics["negative_slices"] += 1
        if has_pred:
            diagnostics["negative_slices_with_false_positive"] += 1

    if has_true or has_pred:
        diagnostics["hausdorff_sum"] += hd_dist(true_valid, pred_valid)
        diagnostics["hausdorff_count"] += 1

    true_flat = y_true[valid]
    score_flat = y_score[valid]
    if true_flat.size and np.unique(true_flat).size == 2:
        diagnostics["auc_sum"] += roc_auc_score(true_flat.astype(np.uint8), score_flat)
        diagnostics["auc_count"] += 1


def _load_ignore_mask(mask_path, size):
    path = Path(mask_path)
    ignore_path = path.parent.parent / "ignore" / path.name
    if not ignore_path.exists():
        return None

    ignore = _read_gray(str(ignore_path))
    ignore = cv2.resize(ignore, size, interpolation=cv2.INTER_NEAREST)
    ignore = (ignore > 127).astype(np.float32)
    return torch.from_numpy(ignore).unsqueeze(0).unsqueeze(0)


def run_evaluation(model_factory, config):
    seeding(config.seed)
    size = (config.image_size, config.image_size)
    checkpoint_path = config.checkpoints_dir / config.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run the matching train script first.")

    for item in ("mask", "joint"):
        create_dir(config.output_dir / item)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    (_, _), (_, _), (test_x, test_y) = load_data(config.data_dir)
    if len(test_x) == 0:
        raise RuntimeError(f"Test split is empty. Check data directory: {config.data_dir}")
    slice_lookup = build_slice_lookup(test_x) if config.use_25d else None

    diagnostics = {
        "counts": np.zeros(4, dtype=np.float64),
        "hausdorff_sum": 0.0,
        "hausdorff_count": 0,
        "auc_sum": 0.0,
        "auc_count": 0,
        "positive_slices": 0,
        "negative_slices": 0,
        "negative_slices_with_false_positive": 0,
    }
    time_taken = []

    for index, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        if config.use_25d:
            prev_idx, center_idx, next_idx = neighbor_indices(index, test_x, slice_lookup)
            slices = []
            for slice_idx in (prev_idx, center_idx, next_idx):
                img = _read_gray(test_x[slice_idx])
                img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                slices.append(img)

            image_np = np.stack(slices, axis=0)
            image_np = np.expand_dims(image_np, axis=1).astype(np.float32) / 255.0
            save_img = cv2.cvtColor(slices[1], cv2.COLOR_GRAY2BGR)
        else:
            image_np = cv2.imread(x_path, cv2.IMREAD_COLOR)
            if image_np is None:
                raise RuntimeError(f"Failed to read image: {x_path}")
            image_np = cv2.resize(image_np, size, interpolation=cv2.INTER_LINEAR)
            save_img = image_np.copy()
            image_np = np.transpose(image_np, (2, 0, 1)).astype(np.float32) / 255.0

        image = torch.from_numpy(image_np).unsqueeze(0).to(device)

        mask_np = _read_gray(y_path)
        mask_np = cv2.resize(mask_np, size, interpolation=cv2.INTER_NEAREST)
        mask_np = (mask_np > 127).astype(np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        save_mask = (mask_np * 255).astype(np.uint8)
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

        ignore = _load_ignore_mask(y_path, size)
        if ignore is not None:
            ignore = ignore.to(device)

        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            amp_enabled = config.use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                y_pred = torch.sigmoid(model(image))
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)

            valid_mask = (ignore < 0.5).float() if ignore is not None else torch.ones_like(mask)
            _update_diagnostics(
                mask[0, 0].detach().cpu().numpy(),
                y_pred[0, 0].detach().cpu().numpy(),
                valid_mask[0, 0].detach().cpu().numpy(),
                diagnostics,
            )
            y_pred_vis = _process_prediction(y_pred)

        name = f"{Path(x_path).parent.parent.name}_{Path(x_path).name}"
        line = np.ones((size[0], 10, 3), dtype=np.uint8) * 255
        joint = np.concatenate([save_img, line, save_mask, line, y_pred_vis], axis=1)
        cv2.imwrite(str(config.output_dir / "joint" / name), joint)
        cv2.imwrite(str(config.output_dir / "mask" / name), y_pred_vis)

    summary = metrics_from_confusion(diagnostics["counts"])
    summary["hausdorff_pixels"] = (
        diagnostics["hausdorff_sum"] / diagnostics["hausdorff_count"]
        if diagnostics["hausdorff_count"]
        else None
    )
    summary["hausdorff_valid_samples"] = diagnostics["hausdorff_count"]
    summary["auc"] = diagnostics["auc_sum"] / diagnostics["auc_count"] if diagnostics["auc_count"] else None
    summary["auc_valid_samples"] = diagnostics["auc_count"]
    summary["positive_slices"] = diagnostics["positive_slices"]
    summary["negative_slices"] = diagnostics["negative_slices"]
    summary["negative_slice_false_positive_rate"] = (
        diagnostics["negative_slices_with_false_positive"] / diagnostics["negative_slices"]
        if diagnostics["negative_slices"]
        else None
    )
    _print_score(summary)
    mean_time_taken = float(np.mean(time_taken))
    summary.update(
        {
            "model": config.model_name,
            "dataset_version": os.environ.get("DATASET_VERSION"),
            "label_type": os.environ.get("LABEL_TYPE"),
            "test_samples": len(test_x),
            "mean_inference_seconds": mean_time_taken,
            "mean_fps": 1 / mean_time_taken if mean_time_taken > 0 else 0.0,
            "image_size": config.image_size,
            "use_25d": config.use_25d,
            "use_amp": config.use_amp and device.type == "cuda",
            "seed": config.seed,
            "checkpoint": str(checkpoint_path),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    )
    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Mean FPS: {summary['mean_fps']:.4f}")
    print(f"Saved metrics: {metrics_path}")
