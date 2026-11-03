# AI Tools and Disclosure Log

In accordance with hackathon rules and transparency guidelines, this document logs all AI tools, models, and assistants utilized during the design, development, and operation of **PatchPilot**.

| Tool / Model | Provider | Purpose / Phase | Notes |
|--------------|----------|-----------------|-------|
| **Antigravity (Agentic AI Assistant)** | Google DeepMind | Architecture design, scaffold generation, test development, documentation | Autonomous pair programming assistant in Antigravity IDE |
| **Gemini 2.5 Flash / Gemini 3.8 Flash** | Google (`google-genai` SDK) | Runtime LLM review layer: finding explanation, risk assessment, and automated minimal patch generation | Called at runtime via `patchpilot.llm.enrich` with JSON mode |
| **PyCode-Vul Dataset** | Hugging Face (`S-AIR-L/PyCode-Vul`) | Training dataset for Phase 3 function-level risk classification | 17,620 Python functions mined from security patches |
| **HistGradientBoostingClassifier / Scikit-Learn** | Scikit-Learn (Open Source) | CPU baseline function risk scoring model | Combined AST features + TF-IDF; packaged in `ml/artifacts/baseline_model.joblib` |
| **CodeBERT (`microsoft/codebert-base`)** | Hugging Face / Microsoft | Deep learning fine-tuning script (`ml/train_codebert.py`) | Colab T4 GPU-ready PyTorch/Transformers script |
| **PatchPilot Benchmark Suite** | Custom Benchmark (`eval/benchmark.py`) | Phase 4 honest pipeline evaluation | Evaluated Bandit, Static PatchPilot, Full PatchPilot, and ML Prior on 3,288 held-out non-test functions from PyCode-Vul |

## Human Oversight & Code Integrity
- All AST rules, McCabe complexity calculations, and scoring ranking logic are deterministically implemented and verified with exhaustive automated unit tests.
- Gemini LLM generation is isolated as an enrichment layer, designed with graceful fallback so failure or lack of an API key never degrades core code review functionality. Gemini is capped at 5 files per PR to conserve latency and quota.
- Machine learning risk predictions are estimated risk probabilities ($0.0 \dots 1.0$) that modulate ranking rather than emitting standalone alerts, backed by leak-proof group-aware splits and sanity controls.
- GitHub Pull Request Review Layer: Employs deterministic unified-diff parsing to isolate added and modified lines, ensuring only relevant diff findings are reported while pre-existing issues are hidden. A 100% offline demo fixture is provided for zero-network validation.
- Evaluation Benchmark: Strictly conducted on held-out test splits without LLM assistance, using frozen validation thresholds and 1,000-resample bootstrap confidence intervals. All trade-offs (including high FPR for static analysis and where Bandit achieves higher precision) are reported transparently.

