"""Action-Chunk Entropy (ACE) style uncertainty scoring.

Supports:
- Mode A: Sample-based chunk dispersion (when multiple samples available)
- Mode B: Logged chunk proxy (chunk norm variability, chunk-to-chunk change)
- Mode C: Interface for online multi-sample ACE (placeholder)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_ace_scores(
    action_chunks: np.ndarray | None = None,
    chunk_field: str | None = None,
    dataset: dict | None = None,
    mode: str = "chunk_change",
    window: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute ACE-style uncertainty scores.

    Args:
        action_chunks: (N, H, A) or (N, H*A) - predicted action chunks per timestep.
        chunk_field: Key in dataset dict to use for chunks (if dataset provided).
        dataset: Full dataset dict (used with chunk_field).
        mode: "chunk_change" (B proxy), "chunk_norm_var" (B proxy), "sample_dispersion" (A, needs multiple samples).
        window: Rolling window for variability (chunk_change / chunk_norm_var).
        seed: For reproducibility.

    Returns:
        (N,) ACE scores. Higher = more uncertain.
    """
    if dataset is not None:
        n = len(dataset.get("episode_id", []))
        if n == 0:
            raise ValueError("Dataset has no episode_id; cannot infer length")
        if chunk_field is None or chunk_field not in dataset:
            if chunk_field:
                logger.warning(
                    f"ACE: chunk field '{chunk_field}' not in dataset. "
                    "Using zeros."
                )
            return np.zeros(n, dtype=np.float32)
        chunks = np.asarray(dataset[chunk_field], dtype=np.float32)
    elif action_chunks is not None:
        chunks = np.asarray(action_chunks, dtype=np.float32)
    else:
        raise ValueError("Provide action_chunks or dataset")

    if chunks.ndim == 2:
        chunks = chunks.reshape(chunks.shape[0], -1)
    n = chunks.shape[0]

    if mode == "chunk_change":
        return _ace_chunk_change(chunks, window)
    if mode == "chunk_norm_var":
        return _ace_chunk_norm_var(chunks, window)
    if mode == "sample_dispersion":
        # Mode A: would need (N, num_samples, H*A) - not available from single logged chunk
        logger.warning(
            "ACE sample_dispersion requires multiple chunk samples per timestep. "
            "Falling back to chunk_change proxy."
        )
        return _ace_chunk_change(chunks, window)
    raise ValueError(f"Unknown ACE mode: {mode}")


def _ace_chunk_change(chunks: np.ndarray, window: int) -> np.ndarray:
    """Proxy: magnitude of chunk-to-chunk change over rolling window."""
    n = chunks.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        start = max(0, i - window)
        diffs = chunks[i] - chunks[start:i]
        scores[i] = np.mean(np.linalg.norm(diffs, axis=1))
    return scores


def _ace_chunk_norm_var(chunks: np.ndarray, window: int) -> np.ndarray:
    """Proxy: variance of chunk L2 norm over rolling window."""
    n = chunks.shape[0]
    norms = np.linalg.norm(chunks, axis=1)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        scores[i] = np.var(norms[start : i + 1]) if start <= i else 0.0
    return scores


def compute_ace_from_samples(chunk_samples: np.ndarray) -> np.ndarray:
    """Mode A: Compute dispersion across N sampled chunks for same observation.

    chunk_samples: (N_samples, H, A) or (N_samples, H*A)
    Returns: scalar score (e.g. mean variance across positions).
    """
    if chunk_samples.ndim == 2:
        chunk_samples = chunk_samples.reshape(1, -1)
    flat = chunk_samples.reshape(chunk_samples.shape[0], -1)
    return float(np.mean(np.var(flat, axis=0)))


# --- Interface for later online integration ---
# TODO: When ACT supports multi-sample chunk generation at runtime,
# implement CandidateChunkScorer.score_candidates(chunk_samples) -> float
