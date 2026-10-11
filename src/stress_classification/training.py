"""Reproducible PyTorch training on the fixed subject-independent split."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .deep_data import ARRAY_FILES, CachedMultimodalDataset
from .evaluation import save_evaluation
from .models import create_model, parameter_counts
from .splits import SubjectSplit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(
    dataset: CachedMultimodalDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for eeg, ecg, hrv, target, _ in loader:
        eeg, ecg, hrv, target = (
            eeg.to(device, non_blocking=True),
            ecg.to(device, non_blocking=True),
            hrv.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(eeg, ecg, hrv)
                loss = criterion(logits, target)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        running_loss += float(loss.detach()) * len(target)
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    average_loss = running_loss / len(loader.dataset)
    accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    return average_loss, accuracy, macro_f1


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    with torch.no_grad():
        for eeg, ecg, hrv, _, source_index in loader:
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(eeg.to(device), ecg.to(device), hrv.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            indices.append(source_index.numpy())
            weights = getattr(model, "last_attention_weights", None)
            if weights is not None:
                attention.append(weights.cpu().numpy())
    return (
        np.concatenate(indices),
        np.concatenate(probabilities),
        np.concatenate(attention) if attention else None,
    )


def _plot_history(history: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title in zip(
        axes,
        ("loss", "accuracy", "f1_macro"),
        ("Loss", "Accuracy", "Macro F1"),
    ):
        axis.plot(history.epoch, history[f"train_{metric}"], label="train")
        axis.plot(history.epoch, history[f"validation_{metric}"], label="validation")
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def train_experiment(
    config: dict,
    *,
    model_name: str,
    cache_dir: str | Path,
    window_index: pd.DataFrame,
    output_dir: str | Path,
    device_name: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    split = SubjectSplit.from_mapping(config["split"])
    index = window_index.reset_index(drop=True)
    masks = split.masks(index.subject_id.to_numpy())
    positions = {name: np.flatnonzero(mask) for name, mask in masks.items()}
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    settings = config["deep"]
    batch_size = int(settings["batch_size"])
    workers = int(settings["num_workers"])
    datasets = {
        "train": CachedMultimodalDataset(cache_dir, positions["train"], augment=True, seed=seed),
        "validation": CachedMultimodalDataset(cache_dir, positions["validation"]),
        "test": CachedMultimodalDataset(cache_dir, positions["test"]),
    }
    loaders = {
        name: _loader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            workers=workers,
            seed=seed,
        )
        for name, dataset in datasets.items()
    }
    model = create_model(model_name).to(device)
    labels = np.load(Path(cache_dir) / ARRAY_FILES["labels"], mmap_mode="r")
    train_counts = np.bincount(np.asarray(labels[positions["train"]]), minlength=3)
    class_weights = train_counts.sum() / np.maximum(train_counts * len(train_counts), 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=float(settings["label_smoothing"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["epochs"]), eta_min=1e-6
    )
    use_amp = bool(settings["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
    (destination / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    best_f1 = -np.inf
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    log_path = destination / "training_log.jsonl"
    checkpoint_path = destination / "best_checkpoint.pt"
    for epoch in range(1, int(settings["epochs"]) + 1):
        train_loss, train_accuracy, train_f1 = _epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        validation_loss, validation_accuracy, validation_f1 = _epoch(
            model, loaders["validation"], criterion, device, use_amp=use_amp
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_f1_macro": train_f1,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_f1_macro": validation_f1,
        }
        history_rows.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if validation_f1 > best_f1 + 1e-6:
            best_f1 = validation_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_f1_macro": validation_f1,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    history = pd.DataFrame(history_rows)
    history.to_csv(destination / "training_history.csv", index=False)
    _plot_history(history, destination / "training_curves.png")
    source_indices, probabilities, attention = predict(
        model, loaders["test"], device, use_amp
    )
    order = np.argsort(source_indices)
    source_indices = source_indices[order]
    probabilities = probabilities[order]
    expected = positions["test"]
    if not np.array_equal(source_indices, expected):
        raise RuntimeError("test predictions are not aligned with the window index")
    test_metadata = index.iloc[source_indices].reset_index(drop=True)
    metrics = save_evaluation(
        destination,
        test_metadata,
        np.asarray(labels[source_indices]),
        probabilities,
    )
    if attention is not None:
        attention = attention[order]
        pd.DataFrame(
            {
                "sample_id": test_metadata.sample_id,
                "eeg_weight": attention[:, 0],
                "ecg_weight": attention[:, 1],
            }
        ).to_csv(destination / "attention_weights.csv", index=False)
    runtime = time.perf_counter() - started
    metadata = {
        "model": model_name,
        "parameter_counts": parameter_counts(model),
        "best_epoch": best_epoch,
        "best_validation_f1_macro": float(best_f1),
        "runtime_seconds": runtime,
        "execution_environment": os.environ.get("EXECUTION_ENVIRONMENT", "local"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "mixed_precision": use_amp,
        "train_class_counts": train_counts.tolist(),
        "class_weights": class_weights.tolist(),
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {"metrics": metrics, "metadata": metadata}
