"""Windowed alarm aggregation for FIPER-style runtime failure prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WindowedAlarmAggregator:
    """Aggregate RND and ACE scores over a rolling window, trigger alarm when both exceed thresholds."""

    def __init__(
        self,
        rnd_threshold: float,
        ace_threshold: float,
        window_size: int = 3,
        agg_mode: str = "mean",
        rule: str = "and",
    ):
        """
        Args:
            rnd_threshold: Threshold for RND score.
            ace_threshold: Threshold for ACE score.
            window_size: Rolling window length.
            agg_mode: "mean" or "consecutive" (count of consecutive exceedances in window).
            rule: "and" (default) = alarm when both aggregated scores exceed thresholds.
        """
        self.rnd_threshold = rnd_threshold
        self.ace_threshold = ace_threshold
        self.window_size = window_size
        self.agg_mode = agg_mode
        self.rule = rule

    def compute_alarms(
        self,
        rnd_scores: np.ndarray,
        ace_scores: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute per-timestep alarms.

        Returns:
            alarms: (N,) binary, 1 = alarm triggered.
            info: dict with rnd_agg, ace_agg, etc.
        """
        rnd_scores = np.asarray(rnd_scores, dtype=np.float64).ravel()
        ace_scores = np.asarray(ace_scores, dtype=np.float64).ravel()
        n = len(rnd_scores)
        if len(ace_scores) != n:
            raise ValueError(f"Mismatched lengths: rnd={n}, ace={len(ace_scores)}")

        rnd_agg = self._aggregate(rnd_scores, self.rnd_threshold)
        ace_agg = self._aggregate(ace_scores, self.ace_threshold)

        rnd_exceed = rnd_agg >= self.rnd_threshold
        ace_exceed = ace_agg >= self.ace_threshold

        if self.rule == "and":
            alarms = (rnd_exceed & ace_exceed).astype(np.int64)
        elif self.rule == "or":
            alarms = (rnd_exceed | ace_exceed).astype(np.int64)
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

        info = {
            "rnd_agg": rnd_agg,
            "ace_agg": ace_agg,
            "rnd_exceed": rnd_exceed,
            "ace_exceed": ace_exceed,
        }
        return alarms, info

    def _aggregate(self, scores: np.ndarray, threshold: float | None = None) -> np.ndarray:
        if self.agg_mode == "mean":
            return _rolling_mean(scores, self.window_size)
        if self.agg_mode == "consecutive":
            t = threshold if threshold is not None else (self.rnd_threshold + self.ace_threshold) / 2
            return _rolling_consecutive_exceedances(scores, t, self.window_size)
        raise ValueError(f"Unknown agg_mode: {self.agg_mode}")


def _rolling_mean(scores: np.ndarray, window: int) -> np.ndarray:
    n = len(scores)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.mean(scores[start : i + 1])
    return out


def _rolling_consecutive_exceedances(scores: np.ndarray, threshold: float, window: int) -> np.ndarray:
    exceed = scores >= threshold
    n = len(scores)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.sum(exceed[start : i + 1]) / max(1, i - start + 1)
    return out
