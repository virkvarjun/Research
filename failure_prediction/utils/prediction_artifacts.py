"""Utilities for saving and loading prediction artifacts.

Supports val/test predictions with logits, probs, labels, episode_ids, timesteps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_predictions(
    out_path: str | Path,
    logits: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray | None = None,
    **extra: np.ndarray,
) -> None:
    """Save prediction artifacts to .npz."""
    data = {
        "logits": np.asarray(logits, dtype=np.float64),
        "probs": np.asarray(probs, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.float64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
    }
    if timesteps is not None:
        data["timesteps"] = np.asarray(timesteps, dtype=np.int64)
    for k, v in extra.items():
        data[k] = np.asarray(v)
    np.savez(Path(out_path), **data)


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    """Load prediction artifacts from .npz."""
    data = dict(np.load(Path(path), allow_pickle=True))
    return {k: np.asarray(v) for k, v in data.items()}
