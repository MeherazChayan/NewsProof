from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["supports", "contradicts", "partially_related", "background_only", "unrelated", "insufficient"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row.get("relation_label") or "").strip()
            if label not in LABEL2ID:
                raise ValueError(f"{path}:{line_number} has unsupported relation_label {label!r}")
            rows.append(row)
    return rows


def format_pair(row: dict) -> str:
    evidence = " ".join(
        str(row.get(key) or "")
        for key in ("evidence_title", "evidence_snippet", "evidence_body")
    )
    return f"CLAIM: {row.get('claim_text', '')}\nEVIDENCE: {evidence}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a multilingual NewsProof claim-evidence verifier.")
    parser.add_argument("--train", type=Path, default=ROOT / "ml/datasets/newsproof_claim_evidence_seed.jsonl")
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ml/models/claim-evidence-verifier")
    parser.add_argument("--base-model", default="joeddav/xlm-roberta-large-xnli")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        import evaluate
        import numpy as np
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them in the backend environment, for example:\n"
            "pip install datasets evaluate transformers accelerate scikit-learn\n"
            f"Original import error: {exc.__class__.__name__}: {exc}"
        )

    train_rows = load_rows(args.train)
    eval_rows = load_rows(args.eval) if args.eval else train_rows[-max(1, len(train_rows) // 5):]
    if args.eval is None and len(train_rows) > 1:
        train_rows = train_rows[: -len(eval_rows)]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label={index: label for label, index in LABEL2ID.items()},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def encode(row: dict) -> dict:
        encoded = tokenizer(format_pair(row), truncation=True, max_length=args.max_length)
        encoded["labels"] = LABEL2ID[row["relation_label"]]
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(encode)
    eval_dataset = Dataset.from_list(eval_rows).map(encode)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {"labels": LABELS, "base_model": args.base_model, "train_rows": len(train_rows), "eval_rows": len(eval_rows)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "newsproof_training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved claim-evidence verifier to {args.output_dir}")


if __name__ == "__main__":
    main()
