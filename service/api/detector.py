"""
Watermark detection logic.

Provides likelihood computation and Bayesian detection,
using the core detection logic from /src.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from watermark detection."""
    
    detected: bool
    confidence: float
    score: float
    threshold: float
    posterior: Optional[float] = None
    log_odds: Optional[float] = None
    n_elements: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "score": self.score,
            "threshold": self.threshold,
            "posterior": self.posterior,
            "log_odds": self.log_odds,
            "n_elements": self.n_elements,
        }


class Detector:
    """
    Watermark detector using S-statistic and Bayesian inference.
    
    This class wraps the detection logic from /src for use in the API.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        prior_watermarked: float = 0.5,
    ):
        """
        Initialize detector.
        
        Args:
            threshold: Detection threshold (Bayesian posterior)
            prior_watermarked: Prior probability of watermark
        """
        self.threshold = threshold
        self.prior_watermarked = prior_watermarked
    
    def detect_from_score(
        self,
        score: float,
        n_elements: int = 16384,
    ) -> DetectionResult:
        """
        Make detection decision from a pre-computed score.
        
        Uses Bayesian inference to compute posterior probability.
        
        Args:
            score: S-statistic or similar score
            n_elements: Number of elements used in computation
            
        Returns:
            DetectionResult with decision and statistics
        """
        try:
            from scipy import stats
            
            # Compute p-value (one-sided test: watermarked implies positive S)
            p_value = 1 - stats.norm.cdf(score)
            
            # Bayesian posterior
            # P(W|S) = P(S|W) * P(W) / P(S)
            # Using normal approximation:
            # P(S|W) ~ N(mu_w, 1), P(S|~W) ~ N(0, 1)
            # For simplicity, use likelihood ratio
            mu_watermarked = np.sqrt(n_elements) * 0.1  # Assumed effect size
            
            likelihood_ratio = np.exp(score * mu_watermarked - 0.5 * mu_watermarked**2)
            prior_odds = self.prior_watermarked / (1 - self.prior_watermarked)
            posterior_odds = prior_odds * likelihood_ratio
            posterior = posterior_odds / (1 + posterior_odds)
            
            log_odds = np.log(posterior_odds) if posterior_odds > 0 else float('-inf')
            
        except ImportError:
            # Fallback without scipy
            p_value = 0.5  # Unknown
            posterior = 0.5 if score > 0 else 0.0
            log_odds = 0.0
        
        # Detection decision based on threshold
        detected = posterior > self.threshold
        confidence = posterior if detected else (1 - posterior)
        
        return DetectionResult(
            detected=detected,
            confidence=float(confidence),
            score=float(score),
            threshold=self.threshold,
            posterior=float(posterior),
            log_odds=float(log_odds),
            n_elements=n_elements,
        )
    
    def detect_from_gpu_response(
        self,
        gpu_response: Dict[str, Any],
    ) -> DetectionResult:
        """
        Create DetectionResult from GPU worker response.
        
        Args:
            gpu_response: Response from GPU worker
            
        Returns:
            DetectionResult
        """
        return DetectionResult(
            detected=gpu_response.get("detected", False),
            confidence=gpu_response.get("confidence", 0.0),
            score=gpu_response.get("score", 0.0),
            threshold=self.threshold,
            posterior=gpu_response.get("posterior"),
            log_odds=gpu_response.get("log_odds"),
        )


# Stub detector that returns mock results (for testing without GPU)
class StubDetector(Detector):
    """
    Stub detector for testing without GPU worker.
    
    Always returns a plausible result based on random chance.
    """
    
    def detect_stub(
        self,
        key_id: str,
        simulate_watermarked: bool = True,
    ) -> DetectionResult:
        """
        Generate a stub detection result.
        
        Args:
            key_id: Key ID (used for deterministic randomness)
            simulate_watermarked: Whether to simulate a watermarked image
            
        Returns:
            Stub DetectionResult
        """
        import hashlib
        
        # Use key_id to generate deterministic "randomness"
        hash_val = int(hashlib.sha256(key_id.encode()).hexdigest()[:8], 16)
        base_score = (hash_val % 1000) / 1000.0
        
        if simulate_watermarked:
            score = 2.0 + base_score * 2.0  # Score between 2 and 4
            posterior = 0.85 + base_score * 0.14  # High posterior
        else:
            score = -0.5 + base_score * 1.0  # Score around 0
            posterior = 0.1 + base_score * 0.3  # Low posterior
        
        detected = posterior > self.threshold
        
        return DetectionResult(
            detected=detected,
            confidence=posterior if detected else (1 - posterior),
            score=score,
            threshold=self.threshold,
            posterior=posterior,
            log_odds=np.log(posterior / (1 - posterior)) if 0 < posterior < 1 else 0.0,
            n_elements=16384,
        )


# Global detector instance
_detector: Optional[Detector] = None


def get_detector() -> Detector:
    """Get the global detector instance."""
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def get_stub_detector() -> StubDetector:
    """Get a stub detector for testing."""
    return StubDetector()
