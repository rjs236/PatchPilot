# PatchPilot ML Risk Classifier Layer (Phase 2 Roadmap)

This directory houses the machine learning models and training scripts for the code risk scoring component of PatchPilot.

## Objective
Train a lightweight classifier (e.g., CodeBERT / XGBoost feature extractor) that predicts per-line risk probabilities based on:
1. AST complexity metrics (nesting depth, cyclomatic complexity, token entropy)
2. Historical bug-fix commit diffs and vulnerability datasets
3. Code churn and structural patterns

## Integration in Phase 1
In Phase 1, the ranker architecture already supports per-line ML risk injection:
```python
rank_findings(findings, ml_risks={line_number: probability_float})
```
Where `score = severity_weight * confidence * category_weight * (1 + ml_risk)`.
The classifier will be trained and integrated in Phase 2.
