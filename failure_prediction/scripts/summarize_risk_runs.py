#!/usr/bin/env python
"""Summarize and rank trained failure-prediction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Summarize trained risk-model runs")
    p.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more training run directories containing config.json and metrics.json",
    )
    p.add_argument(
        "--sort_by",
        type=str,
        default="test_auroc",
        choices=["test_auroc", "test_f1", "best_val_score", "test_auprc"],
    )
    p.add_argument("--output_path", type=str, default="", help="Optional JSON output path")
    return p.parse_args()


def _load_summary(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return {
        "run_dir": str(run_dir),
        "feature_field": config.get("feature_field"),
        "feature_fields": config.get("feature_fields", []),
        "decision_only": config.get("decision_only", False),
        "selection_metric": metrics.get("best_val_metric_name", config.get("selection_metric", "auroc")),
        "best_val_score": metrics.get("best_val_score", metrics.get("best_val_auroc", 0.0)),
        "best_val_auroc": metrics.get("best_val_auroc", 0.0),
        "best_val_auprc": metrics.get("best_val_auprc", 0.0),
        "best_val_f1": metrics.get("best_val_f1", 0.0),
        "test_auroc": metrics["test_metrics"]["auroc"],
        "test_auprc": metrics["test_metrics"]["auprc"],
        "test_f1": metrics["test_metrics"]["f1"],
        "test_ece": summary.get("test_ece"),
        "test_brier_score": summary.get("test_brier_score"),
    }


def main():
    args = parse_args()
    rows = [_load_summary(Path(run_dir)) for run_dir in args.run_dirs]
    rows = sorted(rows, key=lambda row: (row.get(args.sort_by, 0.0), row["best_val_score"]), reverse=True)
    output = {"sort_by": args.sort_by, "runs": rows}
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
