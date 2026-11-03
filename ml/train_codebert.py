"""Fine-tuning CodeBERT for Python function vulnerability classification on Google Colab (T4 GPU)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_NAME = "microsoft/codebert-base"
RANDOM_SEED = 42


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute standard classification evaluation metrics for Hugging Face Trainer."""
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    vuln_probs = probs[:, 1]

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(labels, vuln_probs))
    except Exception:
        auc = 0.50

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT on PyCode-Vul dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional subset of samples for quick run")
    parser.add_argument("--data_dir", type=str, default="ml/data", help="Directory with data splits")
    parser.add_argument("--output_dir", type=str, default="ml/artifacts/codebert_model", help="Final model output path")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(f"Error: Required deep learning dependencies not found: {e}")
        print("Install via: pip install torch transformers datasets accelerate")
        sys.exit(1)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    data_path = Path(args.data_dir)
    train_file = data_path / "train_group.csv"
    val_file = data_path / "val_group.csv"
    test_file = data_path / "test_group.csv"

    if not train_file.exists():
        print(f"Dataset splits not found in {data_path}. Generating now...")
        from ml.data import prepare_and_save_data

        prepare_and_save_data(output_dir=str(data_path))

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    if args.max_samples:
        train_df = train_df.head(args.max_samples)
        val_df = val_df.head(args.max_samples // 4)
        test_df = test_df.head(args.max_samples // 4)
        print(f"Using subset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["code"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    ds_train = Dataset.from_pandas(train_df[["code", "label"]]).map(tokenize_fn, batched=True)
    ds_val = Dataset.from_pandas(val_df[["code", "label"]]).map(tokenize_fn, batched=True)
    ds_test = Dataset.from_pandas(test_df[["code", "label"]]).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "CLEAN", 1: "VULNERABLE"},
        label2id={"CLEAN": 0, "VULNERABLE": 1},
    )

    checkpoint_dir = Path("ml/artifacts/checkpoints")
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting CodeBERT fine-tuning...")
    trainer.train()

    print("\nEvaluating fine-tuned CodeBERT on held-out test split...")
    test_results = trainer.evaluate(ds_test)
    print(f"Test Results: {json.dumps(test_results, indent=2)}")

    preds_raw = trainer.predict(ds_test)
    y_preds = np.argmax(preds_raw.predictions, axis=1)
    cm = confusion_matrix(test_df["label"].values, y_preds).tolist()

    final_metrics = {
        "model_name": "Fine-Tuned CodeBERT (microsoft/codebert-base)",
        "test_accuracy": test_results.get("eval_accuracy"),
        "test_precision": test_results.get("eval_precision"),
        "test_recall": test_results.get("eval_recall"),
        "test_f1": test_results.get("eval_f1"),
        "test_roc_auc": test_results.get("eval_roc_auc"),
        "confusion_matrix": cm,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Saved fine-tuned model and tokenizer to {out_path}")

    metrics_file = Path("ml/artifacts/codebert_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_file}")


if __name__ == "__main__":
    main()
