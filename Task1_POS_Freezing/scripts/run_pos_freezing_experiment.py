import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import requests
import torch
from conllu import parse_incr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup


UD_BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
UD_FILES = {
    "train": "en_ewt-ud-train.conllu",
    "validation": "en_ewt-ud-dev.conllu",
    "test": "en_ewt-ud-test.conllu",
}


@dataclass
class ExperimentConfig:
    model_name: str = "distilbert-base-multilingual-cased"
    language_treebank: str = "UD_English-EWT"
    max_train_samples: int = 2000
    max_eval_samples: int = 500
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 3
    seed: int = 42


class PosDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Tuple[List[str], List[str]]],
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.features = []
        for tokens, tags in examples:
            tokenized = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            word_ids = tokenized.word_ids()
            labels = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous_word_id:
                    labels.append(label_to_id[tags[word_id]])
                else:
                    labels.append(-100)
                previous_word_id = word_id

            tokenized["labels"] = labels
            self.features.append({key: torch.tensor(value) for key, value in tokenized.items()})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.features[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_ud_files(data_dir: Path) -> Dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, filename in UD_FILES.items():
        path = data_dir / filename
        paths[split] = path
        if path.exists() and path.stat().st_size > 0:
            continue
        url = f"{UD_BASE_URL}/{filename}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    return paths


def read_conllu(path: Path, limit: int | None = None) -> List[Tuple[List[str], List[str]]]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for sentence in parse_incr(handle):
            tokens = []
            tags = []
            for token in sentence:
                if not isinstance(token["id"], int):
                    continue
                form = token.get("form")
                upos = token.get("upos")
                if form and upos:
                    tokens.append(form)
                    tags.append(upos)
            if tokens:
                examples.append((tokens, tags))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def build_label_maps(examples: Iterable[Tuple[List[str], List[str]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({tag for _, tags in examples for tag in tags})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def freeze_distilbert_layers(model, freeze_embeddings: bool, freeze_encoder_layers: int) -> List[str]:
    frozen = []
    base = getattr(model, "distilbert", None)
    if base is None:
        raise ValueError("This freezing policy expects a DistilBERT-style model with a .distilbert module.")

    if freeze_embeddings:
        for parameter in base.embeddings.parameters():
            parameter.requires_grad = False
        frozen.append("distilbert.embeddings")

    layers = base.transformer.layer
    for layer_index, layer in enumerate(layers[:freeze_encoder_layers]):
        for parameter in layer.parameters():
            parameter.requires_grad = False
        frozen.append(f"distilbert.transformer.layer.{layer_index}")

    return frozen


def count_parameters(model) -> Dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def evaluate(model, dataloader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / total if total else 0.0,
        "tokens": total,
    }


def train_one_run(
    run_name: str,
    config: ExperimentConfig,
    tokenizer,
    datasets: Dict[str, PosDataset],
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    device: torch.device,
    output_dir: Path,
    freeze: bool,
) -> Dict[str, object]:
    set_seed(config.seed)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    frozen_modules: List[str] = []
    if freeze:
        frozen_modules = freeze_distilbert_layers(
            model,
            freeze_embeddings=config.freeze_embeddings,
            freeze_encoder_layers=config.freeze_encoder_layers,
        )
    model.to(device)

    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += outputs.loss.item()
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        validation_metrics = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(len(train_loader), 1),
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )

    runtime_seconds = time.perf_counter() - start_time
    test_metrics = evaluate(model, test_loader, device)
    parameters = count_parameters(model)

    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)

    return {
        "run_name": run_name,
        "frozen_modules": frozen_modules,
        "parameters": parameters,
        "history": history,
        "test_metrics": test_metrics,
        "runtime_seconds": runtime_seconds,
    }


def write_summary_csv(results: List[Dict[str, object]], output_path: Path) -> None:
    rows = []
    for result in results:
        last_epoch = result["history"][-1]
        params = result["parameters"]
        rows.append(
            {
                "run_name": result["run_name"],
                "validation_accuracy": last_epoch["validation_accuracy"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_loss": result["test_metrics"]["loss"],
                "runtime_minutes": result["runtime_seconds"] / 60.0,
                "total_parameters": params["total"],
                "trainable_parameters": params["trainable"],
                "frozen_parameters": params["frozen"],
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Distil-mBERT for UD PoS tagging with partial freezing.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "ud_english_ewt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "pos_freezing")
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--freeze-encoder-layers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        freeze_encoder_layers=args.freeze_encoder_layers,
    )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ud_paths = download_ud_files(args.data_dir)
    raw_examples = {
        "train": read_conllu(ud_paths["train"], config.max_train_samples),
        "validation": read_conllu(ud_paths["validation"], config.max_eval_samples),
        "test": read_conllu(ud_paths["test"], config.max_eval_samples),
    }
    label_to_id, id_to_label = build_label_maps(
        example for examples in raw_examples.values() for example in examples
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized_datasets = {
        split: PosDataset(examples, tokenizer, label_to_id, config.max_length)
        for split, examples in raw_examples.items()
    }

    runs = [
        ("baseline_full_finetune", False),
        (f"partial_freeze_embeddings_first_{config.freeze_encoder_layers}_layers", True),
    ]
    results = []
    for run_name, should_freeze in runs:
        results.append(
            train_one_run(
                run_name,
                config,
                tokenizer,
                tokenized_datasets,
                label_to_id,
                id_to_label,
                device,
                args.output_dir,
                should_freeze,
            )
        )

    metadata = {
        "config": asdict(config),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_sizes": {split: len(examples) for split, examples in raw_examples.items()},
        "labels": label_to_id,
        "ud_source": UD_BASE_URL,
        "results": results,
    }
    result_json = args.output_dir / "results.json"
    result_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary_csv(results, args.output_dir / "summary.csv")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
