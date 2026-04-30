from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TransformerResult:
    fold_accuracies: List[float]
    pooled_test_correct: int = 0
    pooled_test_total: int = 0
    fold_aucs: List[float] = field(default_factory=list)


def _downsample_trials(
    X: np.ndarray,
    sfreq: float,
    target_sfreq: Optional[float] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if target_sfreq is None or target_sfreq <= 0 or sfreq <= target_sfreq:
        return X
    sfreq_i = int(round(float(sfreq)))
    target_i = int(round(float(target_sfreq)))
    if sfreq_i <= 0 or target_i <= 0:
        return X
    common = gcd(sfreq_i, target_i)
    up = target_i // common
    down = sfreq_i // common
    return resample_poly(X, up=up, down=down, axis=-1).astype(np.float32, copy=False)


class EEGTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_times, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)


def _build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32, copy=False)),
        torch.from_numpy(y_train.astype(np.int64, copy=False)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32, copy=False)),
        torch.from_numpy(y_val.astype(np.int64, copy=False)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _standardize_by_channel(
    X_train: np.ndarray,
    X_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32, copy=False)
    std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32, copy=False)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    return (
        ((X_train - mean) / std).astype(np.float32, copy=False),
        ((X_other - mean) / std).astype(np.float32, copy=False),
        mean,
        std,
    )


def _split_train_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float,
    random_state: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(np.unique(y_train)) < 2 or X_train.shape[0] < 20 or val_fraction <= 0:
        return X_train, y_train, X_train[:0], y_train[:0]
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        if unique_groups.size >= 4:
            group_labels = np.array(
                [
                    int(np.bincount(y_train[groups == g].astype(int)).argmax())
                    for g in unique_groups
                ],
                dtype=np.int64,
            )
            if len(np.unique(group_labels)) >= 2:
                n_val_groups = max(1, int(round(unique_groups.size * val_fraction)))
                n_val_groups = min(n_val_groups, unique_groups.size - 1)
                try:
                    sss = StratifiedShuffleSplit(
                        n_splits=1,
                        test_size=n_val_groups,
                        random_state=random_state,
                    )
                    tr_g_idx, val_g_idx = next(
                        sss.split(np.zeros((unique_groups.size, 1)), group_labels)
                    )
                    train_groups = set(unique_groups[tr_g_idx].tolist())
                    val_groups = set(unique_groups[val_g_idx].tolist())
                    train_idx = np.array(
                        [i for i, g in enumerate(groups) if g in train_groups],
                        dtype=int,
                    )
                    val_idx = np.array(
                        [i for i, g in enumerate(groups) if g in val_groups],
                        dtype=int,
                    )
                    if train_idx.size > 0 and val_idx.size > 0:
                        return (
                            X_train[train_idx],
                            y_train[train_idx],
                            X_train[val_idx],
                            y_train[val_idx],
                        )
                except ValueError:
                    pass
    val_size = max(1, int(round(X_train.shape[0] * val_fraction)))
    if val_size >= X_train.shape[0]:
        return X_train, y_train, X_train[:0], y_train[:0]
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )
    train_idx, val_idx = next(sss.split(np.zeros((X_train.shape[0], 1)), y_train))
    return X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx]


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
    d_model: int,
    n_heads: int,
    n_layers: int,
    ff_dim: int,
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
    model = EEGTransformer(
        n_channels=int(X_tr.shape[1]),
        n_times=int(X_tr.shape[2]),
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
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

    rng_aug = np.random.RandomState(int(random_state) + 17)
    aug_time_shift = max(2, int(round(0.05 * X_tr.shape[2])))
    aug_chan_drop_p = 0.05
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
                if rng_aug.rand() < 0.5:
                    n_drop = max(1, int(round(aug_chan_drop_p * xb.shape[1])))
                    drop_ch = rng_aug.choice(xb.shape[1], size=n_drop, replace=False)
                    xb = xb.clone()
                    xb[:, drop_ch, :] = 0.0
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
    test_loader = DataLoader(TensorDataset(X_test_t, torch.from_numpy(y_test.astype(np.int64, copy=False))), batch_size=batch_size, shuffle=False)
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


def run_transformer_cv_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    sfreq: float,
    target_sfreq: Optional[float] = None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    ff_dim: int = 256,
    val_fraction: float = 0.125,
    patience: int = 5,
    device: str = "cpu",
    random_state: int = 42,
    groups: Optional[np.ndarray] = None,
    paper_exact: bool = False,
) -> TransformerResult:
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
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
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
    return TransformerResult(
        fold_accuracies=fold_accuracies,
        pooled_test_correct=pooled_correct,
        pooled_test_total=pooled_total,
        fold_aucs=fold_aucs,
    )


def fit_transformer_model_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
    target_sfreq: Optional[float] = None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    ff_dim: int = 256,
    device: str = "cpu",
    random_state: int = 42,
) -> Dict:
    X_ds = _downsample_trials(X_proc, sfreq, target_sfreq)
    X_std, _unused, mean, std = _standardize_by_channel(X_ds, X_ds.copy())
    torch.manual_seed(int(random_state))
    model = EEGTransformer(
        n_channels=int(X_std.shape[1]),
        n_times=int(X_std.shape[2]),
        n_classes=int(len(np.unique(y))),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_std.astype(np.float32, copy=False)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _epoch in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {
        "backbone": "transformer",
        "state_dict": state,
        "denoising": denoising,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "feature_sfreq": (float(target_sfreq) if target_sfreq and target_sfreq > 0 else float(sfreq)),
        "l_freq": l_freq,
        "h_freq": h_freq,
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
        "params": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "dropout": float(dropout),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "ff_dim": int(ff_dim),
        },
    }
