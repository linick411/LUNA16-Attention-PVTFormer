import datetime
import csv
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from luna16_data import DEFAULT_DATA_DIR, LUNA16SliceDataset, load_data
from metrics import DiceBCELoss, metrics_from_confusion
from utils import create_dir, epoch_time, print_and_save, seeding, shuffling


@dataclass
class TrainingConfig:
    model_name: str
    checkpoint_name: str
    train_log_name: str
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoints_dir: Path = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    logs_dir: Path = Path(os.environ.get("LOG_DIR", "logs"))
    image_size: int = int(os.environ.get("IMAGE_SIZE", "192"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "16"))
    num_epochs: int = int(os.environ.get("NUM_EPOCHS", "500"))
    lr: float = float(os.environ.get("LR", "1e-4"))
    early_stopping_patience: int = 50
    num_workers: int = int(os.environ.get("NUM_WORKERS", "2"))
    use_25d: bool = os.environ.get("USE_25D", "1") != "0"
    use_amp: bool = os.environ.get("USE_AMP", "1") != "0"
    seed: int = int(os.environ.get("SEED", "42"))


def _batch_counts(y, y_pred):
    true = y > 0.5
    predicted = y_pred > 0.5
    counts = torch.stack(
        [
            torch.count_nonzero(true & predicted),
            torch.count_nonzero(~true & predicted),
            torch.count_nonzero(~true & ~predicted),
            torch.count_nonzero(true & ~predicted),
        ]
    )
    return counts.detach().cpu().numpy().astype(np.float64)


def _four_metrics(counts):
    summary = metrics_from_confusion(counts)
    return [summary[name] for name in ("jaccard", "f1", "recall", "precision")]


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    model.train()
    epoch_loss = 0.0
    epoch_counts = np.zeros(4, dtype=np.float64)
    sample_count = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss detected: {loss.detach().item()}")
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        y_pred = torch.sigmoid(logits)
        batch_counts = _batch_counts(y, y_pred)
        batch_metrics = _four_metrics(batch_counts)

        batch_size = x.shape[0]
        epoch_loss += loss.item() * batch_size
        epoch_counts += batch_counts
        sample_count += batch_size
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "F1": f"{batch_metrics[1]:.4f}"})

    return epoch_loss / sample_count, _four_metrics(epoch_counts)


def evaluate_one_epoch(model, loader, loss_fn, device, use_amp=False):
    model.eval()
    epoch_loss = 0.0
    epoch_counts = np.zeros(4, dtype=np.float64)
    sample_count = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss detected: {loss.detach().item()}")
            y_pred = torch.sigmoid(logits)
            batch_counts = _batch_counts(y, y_pred)
            batch_metrics = _four_metrics(batch_counts)

            batch_size = x.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_counts += batch_counts
            sample_count += batch_size
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "F1": f"{batch_metrics[1]:.4f}"})

    return epoch_loss / sample_count, _four_metrics(epoch_counts)


def run_training(model_factory, config):
    seeding(config.seed)
    create_dir(config.checkpoints_dir)
    create_dir(config.logs_dir)

    checkpoint_path = config.checkpoints_dir / config.checkpoint_name
    train_log_path = config.logs_dir / config.train_log_name
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    history_path = config.logs_dir / f"history_{config.model_name}_{run_id}.csv"
    metadata_path = config.logs_dir / f"run_{config.model_name}_{run_id}.json"
    if not train_log_path.exists():
        train_log_path.write_text("", encoding="utf-8")

    with history_path.open("x", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_jaccard",
                "train_f1",
                "train_recall",
                "train_precision",
                "valid_loss",
                "valid_jaccard",
                "valid_f1",
                "valid_recall",
                "valid_precision",
                "learning_rate",
            ]
        )

    size = (config.image_size, config.image_size)
    print_and_save(str(train_log_path), str(datetime.datetime.now()))
    print_and_save(
        str(train_log_path),
        (
            f"Model: {config.model_name}\n"
            f"Data: {config.data_dir}\n"
            f"Image Size: {size}\n"
            f"Batch Size: {config.batch_size}\n"
            f"LR: {config.lr}\n"
            f"Epochs: {config.num_epochs}\n"
            f"Early Stopping Patience: {config.early_stopping_patience}\n"
            f"Use 2.5D: {config.use_25d}\n"
            f"Use AMP: {config.use_amp and torch.cuda.is_available()}"
        ),
    )

    metadata = {
        "run_id": run_id,
        "model": config.model_name,
        "data_dir": str(config.data_dir),
        "dataset_version": os.environ.get("DATASET_VERSION"),
        "label_type": os.environ.get("LABEL_TYPE"),
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "learning_rate": config.lr,
        "early_stopping_patience": config.early_stopping_patience,
        "num_workers": config.num_workers,
        "use_25d": config.use_25d,
        "use_amp": config.use_amp and torch.cuda.is_available(),
        "seed": config.seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(config.data_dir)
    if len(train_x) == 0 or len(valid_x) == 0:
        raise RuntimeError(
            f"Dataset split is empty. train={len(train_x)}, valid={len(valid_x)}. "
            "Check LUNA16_TASK_DIR and Task03_lung structure."
        )

    train_x, train_y = shuffling(train_x, train_y)
    print_and_save(
        str(train_log_path),
        f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}",
    )

    transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
        ]
    )

    train_dataset = LUNA16SliceDataset(train_x, train_y, size, transform=transform, use_25d=config.use_25d)
    valid_dataset = LUNA16SliceDataset(valid_x, valid_y, size, transform=None, use_25d=config.use_25d)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    loss_fn = DiceBCELoss()
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    print_and_save(str(train_log_path), "Optimizer: Adam\nLoss: BCE Dice Loss")

    best_valid_f1 = float("-inf")
    early_stopping_count = 0

    start_epoch = 0
    resume_path = os.environ.get("TRAIN_RESUME_PATH")
    state_path = checkpoint_path.with_suffix(".last.pt")
    completion_path = checkpoint_path.with_suffix(".complete.json")
    if resume_path:
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            if scaler is not None and payload["scaler"] is not None:
                scaler.load_state_dict(payload["scaler"])
            best_valid_f1 = payload["best_valid_f1"]
            early_stopping_count = payload["early_stopping_count"]
            start_epoch = payload["epoch"]
            resume_kind = "training_state_resume"
        else:
            model.load_state_dict(payload, strict=True)
            _, initial_metrics = evaluate_one_epoch(model, valid_loader, loss_fn, device, use_amp=amp_enabled)
            best_valid_f1 = initial_metrics[1]
            resume_kind = "weights_only_warm_restart"
            if not checkpoint_path.exists():
                _atomic_torch_save(model.state_dict(), checkpoint_path)
        metadata.update(resume_source=str(resume_path), resume_kind=resume_kind,
                        start_epoch=start_epoch, initial_best_valid_f1=best_valid_f1,
                        exact_random_stream_resume=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print_and_save(str(train_log_path), f"Resume: {resume_kind}; source={resume_path}; initial best F1={best_valid_f1:.8f}")
    epoch_pbar = tqdm(range(start_epoch, config.num_epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_pbar:
        start_time = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler=scaler, use_amp=amp_enabled
        )
        valid_loss, valid_metrics = evaluate_one_epoch(
            model, valid_loader, loss_fn, device, use_amp=amp_enabled
        )
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if valid_metrics[1] > best_valid_f1:
            print_and_save(
                str(train_log_path),
                f"Valid F1 improved from {best_valid_f1:2.4f} to {valid_metrics[1]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}",
            )
            best_valid_f1 = valid_metrics[1]
            _atomic_torch_save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        epoch_pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Train F1": f"{train_metrics[1]:.4f}",
                "Valid Loss": f"{valid_loss:.4f}",
                "Valid F1": f"{valid_metrics[1]:.4f}",
                "Best F1": f"{best_valid_f1:.4f}",
                "LR": f"{current_lr:.2e}",
            }
        )

        log = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        log += (
            f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - "
            f"F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - "
            f"Precision: {train_metrics[3]:.4f}\n"
        )
        log += (
            f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - "
            f"F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - "
            f"Precision: {valid_metrics[3]:.4f}\n"
        )
        log += f"\tLearning Rate: {current_lr:.2e}\n"
        print_and_save(str(train_log_path), log)

        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            csv.writer(history_file).writerow(
                [
                    epoch + 1,
                    train_loss,
                    *train_metrics,
                    valid_loss,
                    *valid_metrics,
                    current_lr,
                ]
            )


        _atomic_torch_save({
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch + 1, "best_valid_f1": best_valid_f1,
            "early_stopping_count": early_stopping_count,
            "model_name": config.model_name,
        }, state_path)
        if early_stopping_count >= config.early_stopping_patience:
            print_and_save(
                str(train_log_path),
                f"Early stopping: validation F1 did not improve for {config.early_stopping_patience} epochs.",
            )
            break


    completion_path.write_text(json.dumps({"model": config.model_name, "best_valid_f1": best_valid_f1,
                                          "history": str(history_path), "resume_source": resume_path}), encoding="utf-8")
    print_and_save(str(train_log_path), f"History CSV: {history_path}")
    print_and_save(str(train_log_path), f"Run metadata: {metadata_path}")
