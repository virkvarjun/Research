#!/usr/bin/env python
"""Inspect feature fields in a processed failure dataset.

Lists available fields, shapes, dtypes, NaN/Inf stats, and suggests defaults.
Example:
    python -m failure_prediction.scripts.inspect_feature_fields \
        --processed_dir failure_dataset/transfer_cube/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.data.failure_dataset import load_processed_dataset, get_available_feature_fields
from failure_prediction.utils.feature_stats import inspect_dataset_features


def parse_args():
    p = argparse.ArgumentParser(description="Inspect processed dataset feature fields")
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--mock_data", action="store_true", help="Use synthetic mock dataset")
    p.add_argument("--mock_feature_dim", type=int, default=256)
    p.add_argument("--mock_num_episodes", type=int, default=10)
    p.add_argument("--mock_timesteps", type=int, default=40)
    p.add_argument("--json_out", type=str, default=None, help="Save report to JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mock_data:
        import numpy as np
        rng = np.random.default_rng(42)
        n = args.mock_num_episodes * args.mock_timesteps
        data = {
            "episode_id": np.repeat(np.arange(args.mock_num_episodes), args.mock_timesteps),
            "timestep": np.tile(np.arange(args.mock_timesteps), args.mock_num_episodes),
            "failure_within_k": rng.integers(0, 2, n).astype(np.float32),
            "success": np.ones(n),
            "episode_failed": np.zeros(n),
            "feat_decoder_mean": rng.standard_normal((n, args.mock_feature_dim)).astype(np.float32),
        }
        meta = {"mock": True}
    elif args.processed_dir:
        data, meta = load_processed_dataset(args.processed_dir)
    else:
        print("Either --processed_dir or --mock_data is required")
        sys.exit(1)
    report = inspect_dataset_features(data)
    report["metadata"] = meta

    print("\n=== Feature field inspection ===\n")
    print(f"Available feature fields: {report['feature_fields']}")
    print(f"Embedding candidates (feat_*): {report['embedding_candidates']}")
    print("\nPer-field stats:")
    for key, stats in report["field_stats"].items():
        print(f"  {key}: shape={stats['shape']} dtype={stats['dtype']} "
              f"nan={stats.get('has_nan', False)} inf={stats.get('has_inf', False)} "
              f"dim={stats.get('dim', 'N/A')}")

    if report["embedding_candidates"]:
        print(f"\nSuggested default: {report['embedding_candidates'][0]}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.json_out}")


if __name__ == "__main__":
    main()
