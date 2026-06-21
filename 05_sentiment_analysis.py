import json
import os
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


INPUT_FILE = Path("dataset_processed/comments.jsonl")

OUTPUT_DIR = Path("sentiment")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "comments_sentiment.jsonl"

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 32
MAX_LENGTH = 256

torch.set_num_threads(os.cpu_count())

console = Console()
_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("{task.fields[detail]}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(texts: list[str], tokenizer, model, device) -> list[tuple[str, float]]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    # No gradients since we dont need training
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)
    scores, label_ids = probabilities.max(dim=-1)
    
    # Attach labels
    return [
        (model.config.id2label[label_id.item()], score.item())
        for label_id, score in zip(label_ids, scores)
    ]


def main():
    records = read_jsonl(INPUT_FILE)

    if not records:
        raise RuntimeError(f"Geen data gevonden in {INPUT_FILE}")

    tokenizer, model, device = load_model()

    texts = [record.get("text") or "" for record in records]

    # Sort by text length so each batch pads to a similar length, cutting wasted compute on CPU.
    order = sorted(range(len(records)), key=lambda i: len(texts[i]))
    total_batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE

    with Progress(*_PROGRESS_COLUMNS, console=console) as progress:
        task_id = progress.add_task(
            "Sentiment voorspellen",
            total=total_batches,
            detail=f"0/{len(records)} comments",
        )

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]

            predictions = predict_sentiment(batch_texts, tokenizer, model, device)

            for i, (label, score) in zip(batch_indices, predictions):
                records[i]["sentiment_label"] = label
                records[i]["sentiment_score"] = round(score, 4)

            progress.update(
                task_id,
                advance=1,
                detail=f"{min(start + len(batch_indices), len(records))}/{len(records)} comments",
            )

    write_jsonl(OUTPUT_FILE, records)

    label_counts: dict[str, int] = {}
    for record in records:
        label_counts[record["sentiment_label"]] = label_counts.get(record["sentiment_label"], 0) + 1

    console.print("Klaar.")
    console.print(f"Output: {OUTPUT_FILE}")
    console.print(f"Aantal comments geanalyseerd: {len(records)}")
    for label, count in sorted(label_counts.items(), key=lambda item: item[1], reverse=True):
        console.print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
