#!/usr/bin/env python
"""Record video samples of failure, intervention, and success episodes.

Runs episodes until it collects: 2 failure, 2 intervention, 1 success.
Saves 5 mp4 videos to the output directory.

Example:
    PYTHONPATH=Research MUJOCO_GL=egl python -m failure_prediction.scripts.record_eval_videos \\
        --checkpoint outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \\
        --risk_model_ckpt failure_prediction_runs/transfer_cube_supervised \\
        --risk_threshold 0.5 \\
        --output_dir failure_prediction_runs/videos \\
        --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))
sys.path.insert(0, str(RESEARCH_DIR / "faact"))

from failure_prediction.scripts.run_failure_aware_eval import (
    load_risk_model,
)
from failure_prediction.runtime_components import ThresholdInterventionPolicy
from failure_prediction.scripts.collect_failure_dataset import make_single_env
from faact.backbone.factory import make_backbone_wrapper
from faact.evaluation.online_runner import EpisodeRunnerConfig, run_episode as run_wrapper_episode


def get_frame(env, obs=None):
    """Get current frame from env (rgb array, HWC uint8)."""
    try:
        frame = env.render()
        if frame is not None and frame.size > 0:
            arr = np.asarray(frame)
            if arr.ndim == 3:
                return arr
            if arr.ndim == 4:
                return arr[0]
    except Exception:
        pass
    if obs is not None and "pixels" in obs:
        pix = obs["pixels"]
        if isinstance(pix, dict):
            v = next(iter(pix.values()))
        else:
            v = pix
        arr = np.asarray(v).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return arr[..., :3]
        if arr.ndim == 4:
            return arr[0, ..., :3]
    return np.zeros((480, 640, 3), dtype=np.uint8)


def run_episode_with_frames(
    env,
    backbone,
    risk_scorer,
    intervention_policy,
    risk_threshold: float,
    num_candidate_chunks: int,
    obs_noise_std: float,
    seed: int,
    task_desc: str | None = None,
) -> tuple[dict, list[np.ndarray]]:
    """Run one episode in intervention mode, return result and frames."""
    rng = np.random.default_rng(seed)
    config = EpisodeRunnerConfig(
        mode="intervention",
        num_candidate_chunks=num_candidate_chunks,
        obs_noise_std=obs_noise_std,
        task_desc=task_desc,
    )
    result, frames = run_wrapper_episode(
        env=env,
        backbone=backbone,
        rng=rng,
        risk_scorer=risk_scorer,
        intervention_policy=intervention_policy,
        config=config,
        capture_frames=True,
        frame_fn=get_frame,
    )
    return {
        "success": result["success"],
        "n_interventions": result["n_interventions"],
    }, frames or []


def _draw_label_on_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw label text (SUCCESS / INTERVENTION / FAILURE) on frame. Returns RGB copy."""
    frame = np.asarray(frame).copy()
    try:
        import cv2
        # cv2 uses BGR; frame is RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(frame.shape[0], frame.shape[1]) / 400.0
        thick = max(1, int(scale * 2))
        color_bgr = (0, 200, 0) if label == "success" else (0, 140, 255) if label == "intervention" else (0, 0, 255)
        cv2.putText(bgr, label.upper(), (20, 40), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(bgr, label.upper(), (20, 40), font, scale, color_bgr, thick, cv2.LINE_AA)
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return frame


def _write_videos_readme(out_dir: Path, collected: dict[str, int]) -> None:
    """Write README listing videos with their labels."""
    readme = out_dir / "README.md"
    videos = sorted(out_dir.glob("*.mp4"))
    lines = [
        "# Eval Videos",
        "",
        "| Label | Count | Description |",
        "|-------|-------|-------------|",
        "| **Success** | {} | Episode completed successfully |".format(collected.get("success", 0)),
        "| **Intervention** | {} | Risk-triggered chunk re-selection prevented failure |".format(collected.get("intervention", 0)),
        "| **Failure** | {} | Episode failed (no recovery) |".format(collected.get("failure", 0)),
        "",
        "## Files",
        "",
    ]
    for v in videos:
        # Parse label from filename: success_0_ep12.mp4 -> success
        parts = v.stem.split("_")
        lb = parts[0] if parts else "?"
        lines.append(f"- `{v.name}` — **{lb.upper()}**")
    lines.extend([
        "",
        "Videos are named `{label}_{idx}_ep{ep}.mp4`. The label is also overlaid on each frame.",
    ])
    with open(readme, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {readme}")


def save_video(frames: list[np.ndarray], path: Path, fps: int = 30, label: str | None = None):
    """Save frames to mp4. If label is given, overlay it on every frame."""
    if label:
        frames = [_draw_label_on_frame(f, label) for f in frames]
    try:
        import imageio
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(np.asarray(f))
        writer.close()
        return True
    except ImportError:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            return True
        except Exception:
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--risk_model_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/videos")
    p.add_argument("--task", type=str, default="AlohaTransferCube-v0")
    p.add_argument("--env_type", type=str, default="aloha")
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--num_candidate_chunks", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_episodes", type=int, default=100, help="Max episodes to run before giving up")
    p.add_argument("--n_failure", type=int, default=2)
    p.add_argument("--n_intervention", type=int, default=2)
    p.add_argument("--n_success", type=int, default=1)
    args = p.parse_args()

    importlib.import_module(f"gym_{args.env_type}")
    backbone = make_backbone_wrapper("act", args.checkpoint, device=args.device)
    risk_scorer, _risk_key = load_risk_model(Path(args.risk_model_ckpt), args.device)
    intervention_policy = ThresholdInterventionPolicy(args.risk_threshold)
    env = make_single_env(args.task, args.env_type)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = {"failure": 0, "intervention": 0, "success": 0}
    rng = np.random.default_rng(args.seed)

    for ep in range(args.max_episodes):
        seed = int(rng.integers(0, 2**31))
        result, frames = run_episode_with_frames(
            env=env,
            backbone=backbone,
            risk_scorer=risk_scorer,
            intervention_policy=intervention_policy,
            risk_threshold=args.risk_threshold,
            num_candidate_chunks=args.num_candidate_chunks,
            obs_noise_std=0.03,
            seed=seed,
        )
        success = result["success"]
        n_int = result["n_interventions"]

        label = None
        if not success and collected["failure"] < args.n_failure:
            label = "failure"
        elif n_int > 0 and collected["intervention"] < args.n_intervention:
            label = "intervention"
        elif success and collected["success"] < args.n_success:
            label = "success"

        if label:
            idx = collected[label]
            collected[label] += 1
            out_path = out_dir / f"{label}_{idx}_ep{ep}.mp4"
            if save_video(frames, out_path, label=label):
                print(f"Saved {out_path} ({len(frames)} frames)")

        if all(
            collected[k] >= v
            for k, v in [("failure", args.n_failure), ("intervention", args.n_intervention), ("success", args.n_success)]
        ):
            break

    env.close()
    _write_videos_readme(out_dir, collected)
    print(f"Done. Collected: {collected}")


if __name__ == "__main__":
    main()
