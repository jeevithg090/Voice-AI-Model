"""Optional acoustic emergency detector (Mimi encoder + trained classifier).

The detector is loaded best-effort: if the classifier module
(``src.models.emergency_classifier``), its checkpoint, or the Mimi weights are
unavailable, the server keeps running with distress detection disabled and
reports the reason via ``/ready`` and ``/healthz``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmergencyDetector:
    mimi: Any
    feature_extractor: Any
    classifier: Any
    device: str

    @property
    def sample_rate(self) -> int:
        return int(self.feature_extractor.sampling_rate)

    def predict(self, audio: np.ndarray) -> float:
        """Return the distress probability for one mono window at ``sample_rate``."""
        import torch

        inputs = self.feature_extractor(raw_audio=audio, sampling_rate=self.sample_rate, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            codes = self.mimi.encode(inputs["input_values"]).audio_codes
            return float(self.classifier(codes).item())


def load_emergency_detector(
    device: str,
    mimi_model: str,
    checkpoint_path: str,
    enabled: bool = True,
) -> tuple[EmergencyDetector | None, str]:
    """Try to load the detector. Returns ``(detector_or_None, status_message)``."""
    if not enabled:
        return None, "disabled by ENABLE_EMERGENCY_DETECTION"

    try:
        from src.models.emergency_classifier import MimiEmergencyClassifier
    except ImportError as exc:
        return None, f"classifier module unavailable ({exc})"

    if not os.path.isfile(checkpoint_path):
        return None, f"checkpoint not found at {checkpoint_path}"

    try:
        import torch
        from transformers import AutoFeatureExtractor, MimiModel

        mimi = MimiModel.from_pretrained(mimi_model).to(device).eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(mimi_model)
        classifier = MimiEmergencyClassifier().to(device)
        classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
        classifier.eval()
    except Exception as exc:  # network, corrupt weights, shape mismatch, ...
        logger.exception("Emergency detector failed to load")
        return None, f"load failed ({type(exc).__name__}: {exc})"

    return EmergencyDetector(mimi, feature_extractor, classifier, device), "ready"
