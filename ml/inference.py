"""Inference engine for predicting function-level risk probabilities."""

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

_LOADED_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_MODEL_TYPE: Optional[str] = None


def _get_artifacts_dir() -> Path:
    """Resolve artifacts directory relative to project root or current working dir."""
    candidate1 = Path(__file__).resolve().parent / "artifacts"
    if candidate1.exists():
        return candidate1
    candidate2 = Path("ml/artifacts")
    return candidate2


class RiskPredictor:
    """Predictor class for scoring function risk."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._bundle: Optional[Dict[str, Any]] = None
        self._model_type: Optional[str] = None
        self._loaded: bool = False

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._bundle is not None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if self.model_path is not None:
            if self.model_path.exists():
                try:
                    self._bundle = joblib.load(self.model_path)
                    self._model_type = "baseline"
                except Exception:
                    self._bundle = None
            return

        # Default resolution
        artifacts_dir = _get_artifacts_dir()
        codebert_dir = artifacts_dir / "codebert_model"
        if codebert_dir.exists() and (codebert_dir / "config.json").exists():
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(codebert_dir))
                model = AutoModelForSequenceClassification.from_pretrained(str(codebert_dir))
                model.eval()
                self._bundle = {"tokenizer": tokenizer, "model": model}
                self._model_type = "codebert"
                return
            except Exception:
                pass

        baseline_path = artifacts_dir / "baseline_model.joblib"
        if baseline_path.exists():
            try:
                self._bundle = joblib.load(baseline_path)
                self._model_type = "baseline"
                return
            except Exception:
                pass

    def predict_functions(self, code: str) -> Dict[int, float]:
        """Predict risk probability for functions in given code."""
        if not code or not code.strip():
            return {}

        self._ensure_loaded()
        if self._bundle is None or self._model_type is None:
            return {}

        functions = _extract_functions(code)
        if not functions:
            return {}

        start_lines = [item[0] for item in functions]
        func_codes = [item[1] for item in functions]

        try:
            if self._model_type == "codebert":
                import torch

                tokenizer = self._bundle["tokenizer"]
                model = self._bundle["model"]

                inputs = tokenizer(
                    func_codes,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                return {line: round(float(prob), 3) for line, prob in zip(start_lines, probs)}

            elif self._model_type == "baseline":
                feature_pipeline = self._bundle["feature_pipeline"]
                classifier = self._bundle["classifier"]
                is_dense = self._bundle.get("is_dense", False)

                X = feature_pipeline.transform(func_codes)
                if is_dense and hasattr(X, "toarray"):
                    X = X.toarray()

                if hasattr(classifier, "predict_proba"):
                    probs = classifier.predict_proba(X)[:, 1]
                else:
                    preds = classifier.predict(X)
                    probs = preds.astype(float)

                return {line: round(float(prob), 3) for line, prob in zip(start_lines, probs)}

        except Exception:
            return {}

        return {}


_DEFAULT_PREDICTOR: Optional[RiskPredictor] = None


def get_default_predictor() -> RiskPredictor:
    """Get or initialize singleton default RiskPredictor."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = RiskPredictor()
    return _DEFAULT_PREDICTOR


def _extract_functions(code: str, max_chars_per_func: int = 3000) -> List[Tuple[int, str]]:
    """Parse code and extract (start_line, function_source) for all functions."""
    if not code or not code.strip():
        return []

    funcs: List[Tuple[int, str]] = []
    lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback if syntax error: treat line 1 as a single pseudo-function
        return [(1, code[:max_chars_per_func])]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)

            # Extract source lines
            func_lines = lines[start_line - 1 : end_line]
            func_code = "\n".join(func_lines)[:max_chars_per_func]
            funcs.append((start_line, func_code))

    # If no functions found, treat entire top-level script as line 1
    if not funcs and code.strip():
        funcs.append((1, code[:max_chars_per_func]))

    return funcs


def predict_risk(code: str) -> Dict[int, float]:
    """Predict risk probability (0.0 to 1.0) for each function in the code.

    Returns:
        dict[int, float]: Mapping from function start line to predicted risk score.
        Returns {} if no model is found or prediction fails.
    """
    return get_default_predictor().predict_functions(code)


def map_function_risks_to_lines(code: str, func_risks: Dict[int, float]) -> Dict[int, float]:
    """Map function-level risk probabilities to all lines encompassed by each function."""
    if not code or not func_risks:
        return {}

    line_risks: Dict[int, float] = dict(func_risks)  # ensure start line is present
    lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return line_risks

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)
            risk = func_risks.get(start_line)
            if risk is not None:
                for line_num in range(start_line, end_line + 1):
                    line_risks[line_num] = risk

    return line_risks
