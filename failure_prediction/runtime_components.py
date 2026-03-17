"""Concrete risk scoring and intervention policies for online FAACT runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from failure_prediction.interfaces import InterventionDecision, InterventionPolicy, RiskScore, RiskScorer
from failure_prediction.models.failure_predictor import FailurePredictorMLP


def _logit_to_prob(logit: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))


FIELD_MAP = {
    "feat_decoder_mean": "decoder_mean",
    "feat_encoder_latent_token": "encoder_latent_token",
    "feat_latent_sample": "latent_sample",
    "feat_action_chunk_mean": "action_chunk_mean",
    "feat_action_first": "action_first",
    "feat_action_prefix_mean_10": "action_prefix_mean_10",
    "feat_action_prefix_flat_10": "action_prefix_flat_10",
    "feat_action_remaining_first": "action_remaining_first",
    "feat_action_remaining_prefix_mean_10": "action_remaining_prefix_mean_10",
    "feat_action_remaining_prefix_flat_10": "action_remaining_prefix_flat_10",
}


def resolve_feature_keys(feature_field: str | list[str] | None, config: dict[str, Any] | None = None) -> list[str]:
    """Resolve feat_* field names to runtime feature keys."""
    if isinstance(feature_field, str):
        requested_fields = [field.strip() for field in feature_field.split(",") if field.strip()]
    elif feature_field:
        requested_fields = [str(field).strip() for field in feature_field if str(field).strip()]
    else:
        requested_fields = []

    if not requested_fields and config:
        requested_fields = list(config.get("feature_fields") or [])
        if not requested_fields and config.get("feature_field"):
            requested_fields = [config["feature_field"]]
    if not requested_fields:
        requested_fields = ["feat_decoder_mean"]
    return [FIELD_MAP.get(field, field.replace("feat_", "", 1)) for field in requested_fields]


@dataclass
class TorchMLPRiskScorer(RiskScorer):
    """Risk scorer backed by the failure predictor MLP."""

    model: torch.nn.Module
    feature_keys: list[str]
    device: str

    def predict_step(self, features: Any) -> RiskScore | None:
        if features is None:
            return None
        if isinstance(features, dict):
            parts = []
            for key in self.feature_keys:
                value = features.get(key)
                if value is None:
                    return None
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                parts.append(arr)
            feat_vec = np.concatenate(parts, axis=0) if parts else None
        else:
            feat_vec = features
        if feat_vec is None:
            return None
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(feat_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
            logit = float(self.model(x).cpu().item())
        return RiskScore(logit=logit, prob=_logit_to_prob(logit), raw_score=logit)


@dataclass
class ThresholdInterventionPolicy(InterventionPolicy):
    """Interrupt when risk exceeds a fixed threshold."""

    threshold: float
    cooldown_steps: int = 0
    max_interventions_per_episode: int | None = None
    boundary_only: bool = False

    def should_interrupt(
        self,
        risk_score: RiskScore | None = None,
        **kwargs: Any,
    ) -> InterventionDecision:
        if risk_score is None:
            return InterventionDecision(
                should_interrupt=False,
                reason="no_risk_score",
                confidence=0.0,
                details={"threshold": self.threshold, "risk_prob": None, "threshold_gap": None},
            )
        step = int(kwargs.get("step", 0))
        need_new_chunk = bool(kwargs.get("need_new_chunk", False))
        accepted_so_far = int(kwargs.get("accepted_interventions_so_far", 0))
        last_intervention_step = kwargs.get("last_intervention_step")
        if self.boundary_only and not need_new_chunk:
            return InterventionDecision(
                should_interrupt=False,
                reason="boundary_only_skip",
                confidence=0.0,
                details={
                    "threshold": self.threshold,
                    "risk_prob": risk_score.prob,
                    "threshold_gap": risk_score.prob - self.threshold,
                },
            )
        if self.max_interventions_per_episode is not None and accepted_so_far >= self.max_interventions_per_episode:
            return InterventionDecision(
                should_interrupt=False,
                reason="budget_exhausted",
                confidence=0.0,
                details={
                    "threshold": self.threshold,
                    "risk_prob": risk_score.prob,
                    "threshold_gap": risk_score.prob - self.threshold,
                    "accepted_interventions_so_far": accepted_so_far,
                },
            )
        if (
            self.cooldown_steps > 0
            and last_intervention_step is not None
            and (step - int(last_intervention_step)) < self.cooldown_steps
        ):
            return InterventionDecision(
                should_interrupt=False,
                reason="cooldown_active",
                confidence=0.0,
                details={
                    "threshold": self.threshold,
                    "risk_prob": risk_score.prob,
                    "threshold_gap": risk_score.prob - self.threshold,
                    "steps_since_last_intervention": step - int(last_intervention_step),
                    "cooldown_steps": self.cooldown_steps,
                },
            )
        should_interrupt = risk_score.prob >= self.threshold
        return InterventionDecision(
            should_interrupt=should_interrupt,
            reason="risk_above_threshold" if should_interrupt else "risk_below_threshold",
            confidence=abs(risk_score.prob - self.threshold),
            details={
                "threshold": self.threshold,
                "risk_prob": risk_score.prob,
                "threshold_gap": risk_score.prob - self.threshold,
                "accepted_interventions_so_far": accepted_so_far,
                "step": step,
            },
        )


def load_supervised_risk_runtime(
    ckpt_dir: str | Path,
    device: str,
    feature_field: str | list[str] | None = None,
) -> tuple[TorchMLPRiskScorer, dict[str, Any]]:
    """Load a trained MLP risk scorer and return runtime metadata."""
    ckpt_dir = Path(ckpt_dir)
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Risk model config not found: {cfg_path}")

    with open(cfg_path) as f:
        config = json.load(f)

    model = FailurePredictorMLP(
        input_dim=config.get("input_dim", 512),
        hidden_dims=config.get("hidden_dims", [256, 128]),
        dropout=config.get("dropout", 0.1),
    )
    weight_path = ckpt_dir / "best_model.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"Risk model weights not found: {weight_path}")

    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(device)
    model.eval()

    feature_keys = resolve_feature_keys(feature_field, config=config)
    scorer = TorchMLPRiskScorer(model=model, feature_keys=feature_keys, device=device)
    meta = {
        "config": config,
        "feature_field": feature_field if feature_field is not None else config.get("feature_field"),
        "feature_fields": config.get("feature_fields", []),
        "feature_key": feature_keys[0],
        "feature_keys": feature_keys,
    }
    return scorer, meta
