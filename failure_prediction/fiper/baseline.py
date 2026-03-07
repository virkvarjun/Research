"""FIPER baseline: RND + ACE + conformal calibration + windowed alarm."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from failure_prediction.data.failure_dataset import load_processed_dataset
from failure_prediction.fiper.ace import compute_ace_scores
from failure_prediction.fiper.alarm import WindowedAlarmAggregator
from failure_prediction.fiper.conformal import calibrate_thresholds
from failure_prediction.models.rnd import RNDModule, compute_rnd_scores

logger = logging.getLogger(__name__)


def load_fiper_data(
    processed_dir: str | Path,
    feature_field: str = "feat_decoder_mean",
    action_chunk_field: str | None = "predicted_action_chunk",
) -> dict[str, Any]:
    """Load processed dataset for FIPER with success mask and optional action chunks."""
    data, meta = load_processed_dataset(processed_dir)
    n = len(data["episode_id"])

    if np.any(~np.isfinite(data[feature_field])):
        raise ValueError(f"Feature field '{feature_field}' contains NaN or Inf")

    success = np.asarray(data["success"], dtype=np.float64).ravel()
    success_mask = success > 0.5
    episode_failed = np.asarray(data["episode_failed"], dtype=bool).ravel()
    failure_within_k = np.asarray(data["failure_within_k"], dtype=np.float32).ravel()

    result = {
        "embeddings": np.asarray(data[feature_field], dtype=np.float32),
        "episode_id": np.asarray(data["episode_id"], dtype=np.int64),
        "timestep": np.asarray(data["timestep"], dtype=np.int64) if "timestep" in data else np.arange(n),
        "success_mask": success_mask,
        "episode_failed": episode_failed,
        "failure_within_k": failure_within_k,
        "n_samples": n,
        "feature_field": feature_field,
    }
    if action_chunk_field and action_chunk_field in data:
        result["action_chunks"] = np.asarray(data[action_chunk_field], dtype=np.float32)
        result["action_chunk_field"] = action_chunk_field
    else:
        result["action_chunks"] = None
        result["action_chunk_field"] = None
    return result


def create_mock_fiper_data(
    num_episodes: int = 50,
    timesteps_per_episode: int = 40,
    feature_dim: int = 256,
    success_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Create synthetic FIPER data for mock mode."""
    rng = np.random.default_rng(seed)
    total = num_episodes * timesteps_per_episode
    n_success_eps = int(num_episodes * success_ratio)
    success_ep_ids = set(rng.choice(num_episodes, size=n_success_eps, replace=False))

    episode_ids = np.repeat(np.arange(num_episodes), timesteps_per_episode)
    timesteps = np.tile(np.arange(timesteps_per_episode), num_episodes)

    success_mask = np.array([e in success_ep_ids for e in episode_ids])
    episode_failed = ~success_mask
    failure_within_k = np.zeros(total, dtype=np.float32)
    for ep in range(num_episodes):
        mask = episode_ids == ep
        if ep not in success_ep_ids:
            fail_step = timesteps_per_episode - 1
            k = 10
            fail_window = (timesteps[mask] >= fail_step - k) & (timesteps[mask] < fail_step)
            failure_within_k[mask] = np.where(fail_window, 1.0, 0.0)

    embeddings = rng.standard_normal((total, feature_dim)).astype(np.float32)
    embeddings[~success_mask] += 0.3

    return {
        "embeddings": embeddings,
        "episode_id": episode_ids,
        "timestep": timesteps,
        "success_mask": success_mask,
        "episode_failed": episode_failed,
        "failure_within_k": failure_within_k,
        "n_samples": total,
        "feature_field": "feat_decoder_mean",
        "action_chunks": None,
        "action_chunk_field": None,
    }


class FIPERBaseline:
    """End-to-end FIPER baseline: RND + ACE + calibration + alarm."""

    def __init__(
        self,
        rnd_model: RNDModule | None = None,
        rnd_threshold: float = 0.0,
        ace_threshold: float = 0.0,
        window_size: int = 3,
        alpha: float = 0.1,
        device: str = "cpu",
    ):
        self.rnd_model = rnd_model
        self.rnd_threshold = rnd_threshold
        self.ace_threshold = ace_threshold
        self.window_size = window_size
        self.alpha = alpha
        self.device = device
        self.alarm_agg: WindowedAlarmAggregator | None = None

    def fit_calibration(
        self,
        embeddings: np.ndarray,
        success_mask: np.ndarray,
        action_chunks: np.ndarray | None = None,
        action_chunk_field: str | None = None,
        dataset: dict | None = None,
    ) -> dict[str, Any]:
        """Calibrate thresholds on successful episodes only."""
        if self.rnd_model is None:
            raise ValueError("RND model must be set before calibration")

        rnd_scores = compute_rnd_scores(self.rnd_model, embeddings, self.device)
        ace_scores = compute_ace_scores(
            action_chunks=action_chunks,
            chunk_field=action_chunk_field,
            dataset=dataset,
            mode="chunk_change",
            window=self.window_size,
        )
        if len(ace_scores) != len(rnd_scores):
            ace_scores = np.zeros_like(rnd_scores, dtype=np.float32)

        cal = calibrate_thresholds(rnd_scores, ace_scores, success_mask, alpha=self.alpha)
        self.rnd_threshold = cal["rnd_threshold"]
        self.ace_threshold = cal["ace_threshold"]
        self.alarm_agg = WindowedAlarmAggregator(
            rnd_threshold=self.rnd_threshold,
            ace_threshold=self.ace_threshold,
            window_size=self.window_size,
        )
        return cal

    def compute_alarms(
        self,
        embeddings: np.ndarray,
        action_chunks: np.ndarray | None = None,
        dataset: dict | None = None,
        action_chunk_field: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute per-timestep alarms."""
        if self.rnd_model is None or self.alarm_agg is None:
            raise ValueError("Call fit_calibration first")

        rnd_scores = compute_rnd_scores(self.rnd_model, embeddings, self.device)
        ace_scores = compute_ace_scores(
            action_chunks=action_chunks,
            chunk_field=action_chunk_field,
            dataset=dataset,
            mode="chunk_change",
            window=self.window_size,
        )
        if len(ace_scores) != len(rnd_scores):
            ace_scores = np.zeros_like(rnd_scores, dtype=np.float32)

        alarms, info = self.alarm_agg.compute_alarms(rnd_scores, ace_scores)
        info["rnd_scores"] = rnd_scores
        info["ace_scores"] = ace_scores
        return alarms, info


def run_fiper_baseline(
    processed_dir: str | Path | None = None,
    feature_field: str = "feat_decoder_mean",
    action_chunk_field: str = "predicted_action_chunk",
    rnd_checkpoint: str | Path | None = None,
    mock: bool = False,
    mock_feature_dim: int = 256,
    mock_num_episodes: int = 50,
    mock_timesteps_per_episode: int = 40,
    mock_seed: int = 42,
    window_size: int = 3,
    alpha: float = 0.1,
    train_frac: float = 0.6,
    calibration_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run full FIPER baseline: load data, train/load RND, calibrate, evaluate."""
    from failure_prediction.data.splits import create_episode_splits
    from failure_prediction.models.rnd import RNDModule, compute_rnd_scores

    if mock:
        data = create_mock_fiper_data(
            num_episodes=mock_num_episodes,
            timesteps_per_episode=mock_timesteps_per_episode,
            feature_dim=mock_feature_dim,
            success_ratio=0.5,
            seed=mock_seed,
        )
    else:
        if processed_dir is None:
            raise ValueError("processed_dir required when mock=False")
        data = load_fiper_data(
            processed_dir,
            feature_field=feature_field,
            action_chunk_field=action_chunk_field,
        )

    episode_ids = data["episode_id"]
    embeddings = data["embeddings"]
    success_mask = data["success_mask"]
    episode_failed = data["episode_failed"]
    failure_within_k = data["failure_within_k"]
    timesteps = data["timestep"]

    train_mask, cal_mask, test_mask = create_episode_splits(
        episode_ids,
        train_frac=train_frac,
        val_frac=calibration_frac,
        test_frac=test_frac,
        seed=seed,
    )

    input_dim = embeddings.shape[1]
    rnd = RNDModule(input_dim=input_dim, hidden_dims=[256, 256], seed=seed).to(device)

    if rnd_checkpoint and Path(rnd_checkpoint).exists():
        rnd.load_state_dict(torch.load(rnd_checkpoint, map_location=device))
        logger.info(f"Loaded RND from {rnd_checkpoint}")
    else:
        train_emb = embeddings[train_mask]
        train_success = success_mask[train_mask]
        nominal_emb = train_emb[train_success]
        if len(nominal_emb) == 0:
            nominal_emb = train_emb
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.from_numpy(nominal_emb).float())
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
        for _ in range(10):
            for (xb,) in loader:
                xb = xb.to(device)
                opt.zero_grad()
                loss = rnd.loss(xb)
                loss.backward()
                opt.step()
        logger.info("Trained RND on nominal data")

    cal_emb = embeddings[cal_mask]
    cal_success = success_mask[cal_mask]
    if cal_success.sum() == 0:
        raise ValueError("No successful episodes in calibration set")
    fiper = FIPERBaseline(rnd_model=rnd, window_size=window_size, alpha=alpha, device=device)
    dataset_for_ace = None
    if data.get("action_chunks") is not None:
        dataset_for_ace = {"episode_id": episode_ids, data["action_chunk_field"]: data["action_chunks"]}
    dataset_cal = {"episode_id": episode_ids[cal_mask]}
    chunk_field = data.get("action_chunk_field")
    if chunk_field and data.get("action_chunks") is not None:
        dataset_cal[chunk_field] = data["action_chunks"][cal_mask]
    else:
        chunk_field = "predicted_action_chunk"
    fiper.fit_calibration(
        cal_emb,
        cal_success,
        dataset=dataset_cal,
        action_chunk_field=chunk_field,
    )

    test_emb = embeddings[test_mask]
    test_ep_ids = episode_ids[test_mask]
    test_ts = timesteps[test_mask]
    test_failed = episode_failed[test_mask]
    test_fwk = failure_within_k[test_mask]

    dataset_test = None
    if data.get("action_chunk_field") and data.get("action_chunks") is not None:
        dataset_test = {
            "episode_id": episode_ids[test_mask],
            data["action_chunk_field"]: data["action_chunks"][test_mask],
        }
    else:
        dataset_test = {"episode_id": episode_ids[test_mask]}
    alarms, info = fiper.compute_alarms(
        test_emb,
        dataset=dataset_test,
        action_chunk_field=data.get("action_chunk_field"),
    )
    rnd_scores = info["rnd_scores"]
    ace_scores = info["ace_scores"]

    n_test = len(test_ep_ids)
    unique_test_eps = np.unique(test_ep_ids)
    failed_eps = [e for e in unique_test_eps if test_failed[test_ep_ids == e][0]]
    success_eps = [e for e in unique_test_eps if e not in failed_eps]

    alarm_positives = (alarms > 0.5).sum()
    true_positives = ((alarms > 0.5) & (test_fwk > 0.5)).sum()
    false_positives = ((alarms > 0.5) & (test_fwk <= 0.5)).sum()
    alarm_precision = true_positives / max(alarm_positives, 1)
    alarm_recall = true_positives / max((test_fwk > 0.5).sum(), 1)
    false_alarm_rate = false_positives / max(n_test - (test_fwk > 0.5).sum(), 1)

    failed_eps_with_alarm = 0
    for ep in failed_eps:
        ep_mask = test_ep_ids == ep
        if (alarms[ep_mask] > 0.5).any():
            failed_eps_with_alarm += 1
    pct_failed_eps_with_alarm = 100 * failed_eps_with_alarm / max(len(failed_eps), 1)

    success_eps_with_false_alarm = 0
    for ep in success_eps:
        ep_mask = test_ep_ids == ep
        if (alarms[ep_mask] > 0.5).any():
            success_eps_with_false_alarm += 1
    pct_success_eps_false_alarm = 100 * success_eps_with_false_alarm / max(len(success_eps), 1)

    lead_times = []
    for ep in failed_eps:
        ep_mask = test_ep_ids == ep
        ep_alarms = np.where(alarms[ep_mask] > 0.5)[0]
        ep_ts = test_ts[ep_mask]
        ep_fwk = test_fwk[ep_mask]
        fail_idx = np.where(ep_fwk > 0.5)[0]
        if len(fail_idx) > 0 and len(ep_alarms) > 0:
            t_fail = ep_ts[fail_idx[0]]
            t_first_alarm = ep_ts[ep_alarms[0]]
            if t_first_alarm < t_fail:
                lead_times.append(t_fail - t_first_alarm)

    results = {
        "alarm_precision": float(alarm_precision),
        "alarm_recall": float(alarm_recall),
        "false_alarm_rate": float(false_alarm_rate),
        "pct_failed_eps_with_alarm": float(pct_failed_eps_with_alarm),
        "pct_success_eps_false_alarm": float(pct_success_eps_false_alarm),
        "n_test": n_test,
        "n_failed_eps": len(failed_eps),
        "n_success_eps": len(success_eps),
        "lead_time_mean": float(np.mean(lead_times)) if lead_times else 0.0,
        "lead_time_median": float(np.median(lead_times)) if lead_times else 0.0,
        "test_rnd_scores": rnd_scores,
        "test_ace_scores": ace_scores,
        "test_alarms": alarms,
        "test_episode_ids": test_ep_ids,
        "test_timesteps": test_ts,
        "test_episode_failed": test_failed,
        "test_failure_within_k": test_fwk,
    }
    return results
