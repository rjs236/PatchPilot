"""AST and token-level feature extraction for Python function code risk classification."""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

RISKY_APIS = {
    "eval",
    "exec",
    "pickle",
    "loads",
    "dumps",
    "subprocess",
    "popen",
    "system",
    "yaml_load",
    "unsafe_load",
    "mktemp",
    "marshal",
    "shelve",
    "chmod",
    "socket",
}


def _get_tree_depth(node: ast.AST) -> int:
    """Recursively calculate the maximum depth of an AST tree."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_get_tree_depth(child) for child in children)


def extract_ast_features_single(code: str) -> Dict[str, float]:
    """Extract structural and syntactic metrics from a single Python function.

    Returns deterministic default features if AST parsing fails.
    """
    lines = code.splitlines() if code else []
    line_count = len(lines)
    char_count = len(code)

    feats: Dict[str, float] = {
        "line_count": float(line_count),
        "char_count": float(char_count),
        "complexity": 1.0,
        "max_depth": 1.0,
        "arg_count": 0.0,
        "num_ifs": 0.0,
        "num_loops": 0.0,
        "num_try": 0.0,
        "num_except": 0.0,
        "num_with": 0.0,
        "num_calls": 0.0,
        "num_assigns": 0.0,
        "num_returns": 0.0,
        "num_compares": 0.0,
        "num_bool_ops": 0.0,
        "num_raises": 0.0,
        "has_eval_exec": 0.0,
        "has_subprocess": 0.0,
        "has_pickle": 0.0,
        "has_yaml_load": 0.0,
        "risky_api_count": 0.0,
        "parse_success": 1.0,
    }

    if not code or not code.strip():
        feats["parse_success"] = 0.0
        return feats

    try:
        tree = ast.parse(code)
    except SyntaxError:
        feats["parse_success"] = 0.0
        # Heuristic regex fallbacks for syntax-broken functions
        feats["has_eval_exec"] = 1.0 if re.search(r"\b(eval|exec)\s*\(", code) else 0.0
        feats["has_subprocess"] = 1.0 if "subprocess" in code or "os.system" in code else 0.0
        feats["has_pickle"] = 1.0 if "pickle" in code else 0.0
        feats["has_yaml_load"] = 1.0 if "yaml.load" in code else 0.0
        return feats

    feats["max_depth"] = float(_get_tree_depth(tree))

    complexity = 1
    risky_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            feats["arg_count"] += (
                len(node.args.posonlyargs)
                + len(node.args.args)
                + len(node.args.kwonlyargs)
                + (1 if node.args.vararg else 0)
                + (1 if node.args.kwarg else 0)
            )

        if isinstance(node, (ast.If, ast.IfExp)):
            feats["num_ifs"] += 1
            complexity += 1
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            feats["num_loops"] += 1
            complexity += 1
        elif isinstance(node, ast.Try):
            feats["num_try"] += 1
        elif isinstance(node, ast.ExceptHandler):
            feats["num_except"] += 1
            complexity += 1
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            feats["num_with"] += 1
        elif isinstance(node, ast.Call):
            feats["num_calls"] += 1
            # Check function call name
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in {"eval", "exec"}:
                feats["has_eval_exec"] = 1.0
                risky_calls += 1
            elif func_name in {"system", "popen", "call", "check_output", "Popen"}:
                feats["has_subprocess"] = 1.0
                risky_calls += 1
            elif func_name in {"load", "loads"} and ("pickle" in code or "yaml" in code):
                if "pickle" in code:
                    feats["has_pickle"] = 1.0
                if "yaml" in code:
                    feats["has_yaml_load"] = 1.0
                risky_calls += 1
        elif isinstance(node, ast.Assign):
            feats["num_assigns"] += 1
        elif isinstance(node, ast.Return):
            feats["num_returns"] += 1
        elif isinstance(node, ast.Compare):
            feats["num_compares"] += 1
        elif isinstance(node, ast.BoolOp):
            feats["num_bool_ops"] += 1
            complexity += len(node.values) - 1
        elif isinstance(node, ast.Raise):
            feats["num_raises"] += 1

    feats["complexity"] = float(complexity)
    feats["risky_api_count"] = float(risky_calls)

    return feats


class ASTFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that maps code strings to AST feature matrices."""

    def __init__(self) -> None:
        sample = extract_ast_features_single("def f(): pass")
        self.feature_names_: List[str] = sorted(list(sample.keys()))

    def fit(self, X: Any, y: Any = None) -> "ASTFeatureExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, (pd.Series, np.ndarray)):
            X_list = list(X)
        elif isinstance(X, list):
            X_list = X
        else:
            X_list = [str(X)]

        matrix = np.zeros((len(X_list), len(self.feature_names_)), dtype=np.float32)
        for i, code in enumerate(X_list):
            feats = extract_ast_features_single(code if isinstance(code, str) else "")
            for j, fname in enumerate(self.feature_names_):
                matrix[i, j] = feats.get(fname, 0.0)

        return matrix

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.feature_names_)


TEST_AND_REPO_STOP_WORDS = {
    # Test identifiers
    "test",
    "tests",
    "assert",
    "asserts",
    "mock",
    "mocks",
    "fixture",
    "fixtures",
    "pytest",
    "unittest",
    "patch",
    "patcher",
    "fake",
    # Repository identifiers
    "salt",
    "saltstack",
    "ansible",
    "mlflow",
    "mlflowexception",
    "django",
    "tornado",
    "tornadoweb",
    "paramiko",
    "airflow",
    "w3af",
    "sqlmap",
    "sqlmapproject",
    "jupyter",
    "volatility",
    "volatilityfoundation",
    "pycqa",
    "bandit",
    "pyyaml",
    "requests",
    "flask",
    "pallets",
}


def build_feature_pipeline(
    max_tfidf_features: int = 500,
    filter_test_and_repo_tokens: bool = True,
) -> Pipeline:
    """Build a combined AST + TF-IDF feature pipeline."""
    ast_pipe = Pipeline([("ast", ASTFeatureExtractor())])
    stop_words = sorted(list(TEST_AND_REPO_STOP_WORDS)) if filter_test_and_repo_tokens else None
    tfidf_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"[a-zA-Z_]\w*",
                    max_features=max_tfidf_features,
                    min_df=3,
                    sublinear_tf=True,
                    stop_words=stop_words,
                ),
            )
        ]
    )

    union = FeatureUnion(
        transformer_list=[
            ("ast_features", ast_pipe),
            ("tfidf_features", tfidf_pipe),
        ]
    )
    return Pipeline([("union", union)])
