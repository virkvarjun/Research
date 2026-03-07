"""Conformal-style threshold calibration using successful rollouts only."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calibrate_thresholds(
    rnd_scores: np.ndarray,
    ace_scores: np.ndarray,
    success_mask: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Compute empirical quantile thresholds from successful episodes only.

    For target miscoverage alpha, threshold = quantile(1 - alpha) of scores on success.
    Higher scores = more OOD/uncertain; we want to alarm when score > threshold.

    Args:
        rnd_scores: (N,) RND scores.
        ace_scores: (N,) ACE scores.
        success_mask: (N,) bool, True for successful episode timesteps.
        alpha: Target miscoverage (e.g. 0.1 -> 90% of success scores below threshold).

    Returns:
        Dict with rnd_threshold, ace_threshold, n_calibration, alpha.
    """
    rnd_scores = np.asarray(rnd_scores, dtype=np.float64).ravel()
    ace_scores = np.asarray(ace_scores, dtype=np.float64).ravel()
    success_mask = np.asarray(success_mask, dtype=bool).ravel()

    n = len(success_mask)
    if n == 0:
        raise ValueError("Empty dataset for calibration")
    n_success = int(success_mask.sum())
    if n_success == 0:
        raise ValueError(
            "Calibration set has no successful episodes. "
            "FIPER requires successful rollouts for conformal calibration."
        )

    rnd_cal = rnd_scores[success_mask]
    ace_cal = ace_scores[success_mask]

    q = 1.0 - alpha
    rnd_threshold = float(np.quantile(rnd_cal, q))
    ace_threshold = float(np.quantile(ace_cal, q))

    result = {
        "rnd_threshold": rnd_threshold,
        "ace_threshold": ace_threshold,
        "n_calibration": n_success,
        "alpha": alpha,
    }
    logger.info(
        f"Calibrated: rnd_threshold={rnd_threshold:.4f}, ace_threshold={ace_threshold:.4f} "
        f"(n_success={n_success}, alpha={alpha})"
    )
    return result


def normalize_scores(
    scores: np.ndarray,
    shift: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Simple normalization: (scores - shift) / scale. Use for interpretability."""
    if scale <= 0 or not np.isfinite(scale):
        return np.clip(scores - shift, 0, None).astype(np.float32)
    return ((scores - shift) / scale).astype(np.float32)
