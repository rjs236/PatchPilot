# Google Colab Fine-Tuning Guide (CodeBERT on T4 GPU)

This guide provides reproducible step-by-step instructions for fine-tuning `microsoft/codebert-base` on Google Colab using a free T4 GPU.

---

## 1. Google Colab Environment Setup
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. Select **Runtime** > **Change runtime type** > **T4 GPU** > **Save**.
4. Confirm GPU acceleration:
   ```python
   !nvidia-smi
   ```

---

## 2. Clone Repository and Install Dependencies
In a new code cell, clone your PatchPilot repo and install the required deep learning packages:

```bash
# Clone repository
!git clone https://github.com/<your-username>/patchpilot.git
%cd patchpilot

# Install Colab deep learning dependencies
!pip install -q torch transformers datasets accelerate scikit-learn
```

---

## 3. Launch Fine-Tuning

Run the self-contained training script. It loads the exact same zero-crossing group-aware splits (`ml/data/train_group.csv`, `val_group.csv`, `test_group.csv`):

```bash
# Run fine-tuning on T4 GPU (approx. 25-35 minutes for 3 epochs with mixed precision fp16)
!python -m ml.train_codebert --epochs 3 --batch_size 16 --max_length 256
```

> **Tip for quick testing (10-minute dry run):**
> You can pass `--max_samples 3000` to train on a fast subset:
> ```bash
> !python -m ml.train_codebert --epochs 2 --batch_size 16 --max_samples 3000
> ```

---

## 4. How to Handle Artifacts and the 100 MB GitHub Limit

Once training finishes, the model and tokenizer are saved to:
`ml/artifacts/codebert_model/` (~490 MB uncompressed `model.safetensors`).

Because GitHub enforces a **100 MB hard file limit**, you have three deployment paths:

### Option A: Use Scikit-Learn as Default CPU Baseline (Recommended & Default)
- **PatchPilot is designed CPU-first**: The scikit-learn HistGradientBoosting model in `ml/artifacts/baseline_model.joblib` is only **~1.2 MB**, committed directly to Git, and performs inference in **<5ms** on any laptop CPU with **90.88% Accuracy** and **0.9704 ROC-AUC**.
- If `ml/artifacts/codebert_model` is not downloaded locally, PatchPilot automatically uses this baseline model with zero setup.

### Option B: Download Artifacts for Local Testing
To test the fine-tuned CodeBERT model locally on your machine:
```python
# In Colab: zip and download the model folder
!zip -r codebert_model.zip ml/artifacts/codebert_model/
from google.colab import files
files.download("codebert_model.zip")
```
Unzip into your local repo:
```bash
unzip codebert_model.zip -d ml/artifacts/
```
`ml/inference.py` will automatically detect `ml/artifacts/codebert_model/` and prioritize it!

### Option C: Upload to Hugging Face Hub (Production)
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="ml/artifacts/codebert_model",
    repo_id="your-username/patchpilot-codebert",
    repo_type="model"
)
```

---

## 5. Baseline vs. CodeBERT Comparison & Honesty Note
- **Strengths of CodeBERT**: Captures cross-token contextual semantics, subtle control flow order, and variable misuse that bag-of-words ignores.
- **When CodeBERT is NOT better**:
  - **Inference latency**: CodeBERT takes ~80-150ms per function on CPU vs ~2-5ms for HistGradientBoosting.
  - **Memory footprint**: CodeBERT requires ~500MB RAM + PyTorch runtime vs ~15MB for scikit-learn.
  - **Label artifacts**: If vulnerability datasets contain spurious test-method tokens (`test_`, `assert`), pre-trained language models can overfit to those tokens just as easily as TF-IDF unless carefully regularized.
