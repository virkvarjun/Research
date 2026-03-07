#!/usr/bin/env python
"""Run FIPER baseline offline evaluation.

Computes RND + ACE scores, calibrates on successful episodes, evaluates alarms.
Example mock:
    python -m failure_prediction.scripts.run_fiper_offline_eval \
        --mock_data --output_dir failure_prediction_runs/mock_fiper --window_size 3 --alpha 0.1

Example real:
    python -m failure_prediction.scripts.run_fiper_offline_eval \
        --processed_dir failure_dataset/transfer_cube/processed \
        --feature_field feat_decoder_mean \
        --rnd_checkpoint failure_prediction_runs/transfer_cube_fiper_rnd/rnd_model.pt \
        --output_dir failure_prediction_runs/transfer_cube_fiper \
        --window_size 3 --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.data.failure_dataset import load_processed_dataset, load_failure_dataset, get_available_feature_fields
from failure_prediction.data.splits import create_episode_splits
from failure_prediction.fiper.ace import compute_ace_scores
from failure_prediction.fiper.alarm import WindowedAlarmAggregator
from failure_prediction.fiper.baseline import run_fiper_baseline
from failure_prediction.models.rnd import RNDPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run FIPER offline evaluation")
    p.add_argument("--mock_data", action="store_true")
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--feature_field", type=str, default="feat_decoder_mean")
    p.add_argument("--action_chunk_field", type=str, default="predicted_action_chunk")
    p.add_argument("--rnd_checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/fiper_eval")
    p.add_argument("--window_size", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--calibration_frac", type=float, default=0.2)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--num_mock_episodes", type=int, default=50)
    p.add_argument("--timesteps_per_episode", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mock_data and not args.processed_dir:
        logger.error("Either --mock_data or --processed_dir is required")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = run_fiper_baseline(
        processed_dir=args.processed_dir,
        feature_field=args.feature_field,
        action_chunk_field=args.action_chunk_field,
        rnd_checkpoint=args.rnd_checkpoint,
        mock=args.mock_data,
        mock_feature_dim=args.feature_dim,
        mock_num_episodes=args.num_mock_episodes,
        mock_timesteps_per_episode=args.timesteps_per_episode,
        mock_seed=args.seed,
        window_size=args.window_size,
        alpha=args.alpha,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "fiper_results.json", "w") as f:
        def _convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump({k: _convert(v) for k, v in results.items() if not k.endswith("_arr")}, f, indent=2)

    np.savez(
        out_dir / "fiper_artifacts.npz",
        test_rnd_scores=results.get("test_rnd_scores", np.array([])),
        test_ace_scores=results.get("test_ace_scores", np.array([])),
        test_alarms=results.get("test_alarms", np.array([])),
        test_episode_ids=results.get("test_episode_ids", np.array([])),
        test_timesteps=results.get("test_timesteps", np.array([])),
        test_episode_failed=results.get("test_episode_failed", np.array([])),
        test_failure_within_k=results.get("test_failure_within_k", np.array([])),
    )

    logger.info(
        f"FIPER eval done. Alarm precision={results.get('alarm_precision', 0):.4f} "
        f"recall={results.get('alarm_recall', 0):.4f} "
        f"failed_eps_with_alarm={results.get('pct_failed_eps_with_alarm', 0):.1f}%"
    )


if __name__ == "__main__":
    main()
