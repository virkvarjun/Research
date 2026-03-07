#!/usr/bin/env python
"""Plot failure prediction and FIPER evaluation results.

Supports supervised predictor outputs and FIPER baseline artifacts.
Example:
    python -m failure_prediction.scripts.plot_failure_results \
        --run_dir failure_prediction_runs/transfer_cube_fiper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="Plot failure prediction results")
    p.add_argument("--run_dir", type=str, required=True, help="Run directory with predictions or fiper artifacts")
    p.add_argument("--output_dir", type=str, default=None, help="Where to save plots (default: run_dir/plots)")
    p.add_argument("--episode_id", type=int, default=None, help="Plot score-over-time for specific episode")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        sys.exit(0)

    # Try supervised predictor format
    test_npz = run_dir / "test_predictions.npz"
    fiper_npz = run_dir / "fiper_artifacts.npz"

    if test_npz.exists():
        data = dict(np.load(test_npz, allow_pickle=True))
        probs = data.get("probs", data.get("logits", None))
        if probs is not None:
            if "logits" in data and "probs" not in data:
                probs = 1.0 / (1.0 + np.exp(-np.clip(data["logits"], -500, 500)))
            labels = data.get("labels", np.zeros(len(probs)))
            _plot_supervised(probs, labels, out_dir, args.episode_id, data)

    if fiper_npz.exists():
        data = dict(np.load(fiper_npz, allow_pickle=True))
        rnd = data.get("test_rnd_scores", np.array([]))
        ace = data.get("test_ace_scores", np.array([]))
        if len(rnd) > 0:
            _plot_fiper(rnd, ace, data, out_dir, args.episode_id)

    print(f"Plots saved to {out_dir}")


def _plot_supervised(probs, labels, out_dir, episode_id, data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.hist(probs[labels <= 0.5], bins=30, alpha=0.6, label="Negative", density=True)
    plt.hist(probs[labels > 0.5], bins=30, alpha=0.6, label="Positive", density=True)
    plt.xlabel("Risk score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Supervised risk score distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_histogram.png", dpi=100)
    plt.close()

    if episode_id is not None and "episode_ids" in data:
        ep_ids = np.asarray(data["episode_ids"]).ravel()
        mask = ep_ids == episode_id
        if mask.any():
            t = np.arange(mask.sum())
            plt.figure(figsize=(8, 4))
            plt.plot(t, probs[mask], label="Risk")
            plt.axhline(0.5, color="gray", linestyle="--")
            plt.xlabel("Timestep")
            plt.ylabel("Risk score")
            plt.title(f"Episode {episode_id} risk over time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"risk_episode_{episode_id}.png", dpi=100)
            plt.close()


def _plot_fiper(rnd_scores, ace_scores, data, out_dir, episode_id):
    import matplotlib.pyplot as plt
    ep_failed = data.get("test_episode_failed", np.zeros(len(rnd_scores)))
    ep_ids = data.get("test_episode_ids", np.arange(len(rnd_scores)))

    success_mask = ~(np.asarray(ep_failed).astype(bool))
    fail_mask = ~success_mask

    if success_mask.any() and fail_mask.any():
        plt.figure(figsize=(6, 4))
        plt.hist(rnd_scores[success_mask], bins=30, alpha=0.6, label="Success", density=True)
        plt.hist(rnd_scores[fail_mask], bins=30, alpha=0.6, label="Failure", density=True)
        plt.xlabel("RND score")
        plt.ylabel("Density")
        plt.legend()
        plt.title("RND score: success vs failure")
        plt.tight_layout()
        plt.savefig(out_dir / "rnd_histogram.png", dpi=100)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(ace_scores[success_mask], bins=30, alpha=0.6, label="Success", density=True)
        plt.hist(ace_scores[fail_mask], bins=30, alpha=0.6, label="Failure", density=True)
        plt.xlabel("ACE score")
        plt.ylabel("Density")
        plt.legend()
        plt.title("ACE score: success vs failure")
        plt.tight_layout()
        plt.savefig(out_dir / "ace_histogram.png", dpi=100)
        plt.close()

    if episode_id is not None:
        mask = np.asarray(ep_ids).ravel() == episode_id
        if mask.any():
            t = np.arange(mask.sum())
            plt.figure(figsize=(8, 4))
            plt.plot(t, rnd_scores[mask], label="RND")
            plt.plot(t, ace_scores[mask], label="ACE")
            plt.xlabel("Timestep")
            plt.ylabel("Score")
            plt.title(f"Episode {episode_id} scores over time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"fiper_episode_{episode_id}.png", dpi=100)
            plt.close()


if __name__ == "__main__":
    main()
