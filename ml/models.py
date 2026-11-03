"""Custom model classes and wrappers for PatchPilot ML models."""

from typing import Any
import numpy as np


class CalibratedModelWrapper:
    """Wrapper that applies post-hoc Platt scaling calibration to a base estimator."""

    def __init__(self, base_model: Any, calibrator: Any) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: Any) -> np.ndarray:
        raw_p = self.base_model.predict_proba(X)[:, 1]
        cal_p = self.calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - cal_p, cal_p])

    def predict(self, X: Any) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.50).astype(int)
