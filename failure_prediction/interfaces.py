"""Interfaces for later self-correction integration.

Lightweight abstractions so runtime intervention can plug in without rewrites.
Do NOT implement actual chunk interruption or recovery yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RiskScore:
    """Output of a risk scorer for one timestep."""
    logit: float
    prob: float
    raw_score: float | None = None


class RiskScorer(ABC):
    """Interface for supervised failure risk prediction."""

    @abstractmethod
    def predict_step(self, features: Any) -> RiskScore:
        """Compute risk for a single timestep. features: embedding or dict."""
        pass


@dataclass
class FiperScores:
    """RND + ACE scores for one timestep."""
    rnd_score: float
    ace_score: float
    alarm: bool = False


class FiperScorer(ABC):
    """Interface for FIPER-style OOD + uncertainty scoring."""

    @abstractmethod
    def compute_scores(self, embedding: Any, action_chunk: Any | None = None) -> FiperScores:
        """Compute RND and ACE scores, optionally trigger alarm."""
        pass


@dataclass
class InterventionDecision:
    """Whether to interrupt and what to do (placeholder)."""
    should_interrupt: bool
    reason: str = ""
    confidence: float = 0.0


class InterventionPolicy(ABC):
    """Interface for deciding when to intervene (placeholder)."""

    @abstractmethod
    def should_interrupt(
        self,
        risk_score: RiskScore | None = None,
        fiper_scores: FiperScores | None = None,
        **kwargs: Any,
    ) -> InterventionDecision:
        """Decide whether to interrupt based on risk/FIPER scores."""
        pass
