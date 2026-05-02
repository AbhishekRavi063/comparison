"""ShallowConvNet (Schirrmeister et al. 2017) for EEG classification.

Inputs: (batch, n_channels, n_times) float32, with per-fold channel
standardization.  Temporal conv → spatial depthwise conv → square/log
nonlinearity → average pool → FC head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .transformer_eeg import (
    _build_loaders,
    _downsample_trials,
    _split_train_val,
    _standardize_by_channel,
)


@dataclass
class ShallowConvNetResult:
    fold_accuracies: List[float]
    pooled_test_correct: int = 0
    pooled_test_total: int = 0
    fold_aucs: List[float] = field(default_factory=list)


class ShallowConvNet(nn.Module):
    """ShallowConvNet: temporal conv → spatial depthwise → square-log pool → FC."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        n_temporal_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            1, n_temporal_filters, (1, temporal_kernel), bias=False
        )
        self.spatial_conv = nn.Conv2d(
            n_temporal_filters, n_temporal_filters, (n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_temporal_filters, eps=1e-5, momentum=0.1)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            t = torch.zeros(1, 1, n_channels, n_samples)
            t = self.temporal_conv(t)
            t = self.spatial_conv(t)
            t = self.bn(t)
            t = t ** 2
            t = self.pool(t)
            t = torch.clamp(t, min=1e-7)
            t = torch.log(t)
        n_feat = int(t.numel())
        self.fc = nn.Linear(n_feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, C, T); got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.clamp(x, min=1e-7)
        x = torch.log(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def _fit_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    target_sfreq: Optional[float],
    random_state: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    n_temporal_filters: int,
    temporal_kernel: int,
    pool_size: int,
    pool_stride: int,
    val_fraction: float,
    patience: int,
    device: str = "cpu",
    train_groups: Optional[np.ndarray] = None,
    paper_exact: bool = False,
) -> Tuple[float, float, int, int]:
    X_train_ds = _downsample_trials(X_train, sfreq, target_sfreq)
    X_test_ds = _downsample_trials(X_test, sfreq, target_sfreq)
    X_tr, y_tr, X_val, y_val = _split_train_val(
        X_train_ds,
        y_train,
        val_fraction=val_fraction,
        random_state=random_state,
        groups=train_groups,
    )
    if X_val.shape[0] == 0:
        X_val = X_tr[: min(len(X_tr), batch_size)]
        y_val = y_tr[: min(len(y_tr), batch_size)]
    X_tr, X_test_ds, mean, std = _standardize_by_channel(X_tr, X_test_ds)
    X_val = ((X_val - mean) / std).astype(np.float32, copy=False)

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    try:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    except Exception:
        pass

    n_classes = int(len(np.unique(y_train)))
    model = ShallowConvNet(
        n_channels=int(X_tr.shape[1]),
        n_samples=int(X_tr.shape[2]),
        n_classes=n_classes,
        n_temporal_filters=n_temporal_filters,
        temporal_kernel=temporal_kernel,
        pool_size=pool_size,
        pool_stride=pool_stride,
        dropout=dropout,
    ).to(device)

    if paper_exact:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs))
        )
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = _build_loaders(X_tr, y_tr, X_val, y_val, batch_size)

    best_state = None
    best_val = -np.inf
    stale_epochs = 0

    rng_aug = np.random.RandomState(int(random_state) + 53)
    aug_time_shift = max(2, int(round(0.05 * X_tr.shape[2])))
    use_aug = (not paper_exact) and (epochs > 5)

    for _epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if use_aug:
                shift = int(rng_aug.randint(-aug_time_shift, aug_time_shift + 1))
                if shift != 0:
                    xb = torch.roll(xb, shifts=shift, dims=-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if not paper_exact:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += int(yb.numel())
        val_acc = (val_correct / val_total) if val_total > 0 else 0.0
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, int(patience)):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    X_test_t = torch.from_numpy(X_test_ds.astype(np.float32, copy=False))
    test_loader = DataLoader(
        TensorDataset(X_test_t, torch.from_numpy(y_test.astype(np.int64, copy=False))),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(np.argmax(p, axis=1))
            probs.append(p)
    pred_arr = np.concatenate(preds, axis=0)
    prob_arr = np.concatenate(probs, axis=0)
    acc = float(np.mean(pred_arr == y_test))
    auc = 0.5
    try:
        if prob_arr.shape[1] == 2:
            auc = float(roc_auc_score(y_test, prob_arr[:, 1]))
    except Exception:
        auc = 0.5
    return acc, auc, int(np.sum(pred_arr == y_test)), int(len(y_test))


def run_shallow_convnet_cv_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    sfreq: float,
    target_sfreq: Optional[float] = None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    n_temporal_filters: int = 40,
    temporal_kernel: int = 25,
    pool_size: int = 75,
    pool_stride: int = 15,
    val_fraction: float = 0.111,
    patience: int = 10,
    device: str = "cpu",
    random_state: int = 42,
    groups: Optional[np.ndarray] = None,
    paper_exact: bool = False,
) -> ShallowConvNetResult:
    y = np.asarray(y, dtype=np.int64)
    fold_accuracies: List[float] = []
    fold_aucs: List[float] = []
    pooled_correct = 0
    pooled_total = 0
    for fold_i, (train_idx, test_idx) in enumerate(cv_splits):
        train_groups = None
        if groups is not None:
            train_groups = np.asarray(groups)[train_idx]
        acc, auc, corr, total = _fit_one_fold(
            X_proc[train_idx],
            y[train_idx],
            X_proc[test_idx],
            y[test_idx],
            sfreq=sfreq,
            target_sfreq=target_sfreq,
            random_state=int(random_state) + fold_i,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            n_temporal_filters=n_temporal_filters,
            temporal_kernel=temporal_kernel,
            pool_size=pool_size,
            pool_stride=pool_stride,
            val_fraction=val_fraction,
            patience=patience,
            device=device,
            train_groups=train_groups,
            paper_exact=paper_exact,
        )
        fold_accuracies.append(acc)
        fold_aucs.append(auc)
        pooled_correct += corr
        pooled_total += total
    return ShallowConvNetResult(
        fold_accuracies=fold_accuracies,
        pooled_test_correct=pooled_correct,
        pooled_test_total=pooled_total,
        fold_aucs=fold_aucs,
    )
