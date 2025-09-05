# stages/train_classifier.py
import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from torch.utils.data import Subset
from datasets import load_dataset, DatasetDict
from PIL import Image

import evaluate
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)


# ---------------- UTILITIES ----------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def has_train_val_test_dirs(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, split)) for split in ["train", "val", "test"])


def has_train_test_dirs(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, split)) for split in ["train", "test"])


def build_dataset(data_dir: str, val_ratio: float = 0.2, test_ratio: float = 0.0, seed: int = 42) -> DatasetDict:
    """
    Loads dataset from different structures:
    - data_dir/train, val, test
    - data_dir/train, test
    - data_dir/images/<class>
    """
    if has_train_val_test_dirs(data_dir):
        ds = load_dataset("imagefolder", data_dir=data_dir)
    elif has_train_test_dirs(data_dir):
        ds = load_dataset("imagefolder", data_dir=data_dir)
        if "val" not in ds:
            train_val = ds["train"].train_test_split(test_size=val_ratio, stratify_by_column="label", seed=seed)
            ds = DatasetDict({
                "train": train_val["train"],
                "val": train_val["test"],
                "test": ds["test"]
            })
    else:
        images_dir = os.path.join(data_dir, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Couldn't find valid dataset structure in {data_dir}.\n"
                f"Expected:\n"
                f"- train/val/test\n"
                f"- train/test\n"
                f"- images/<class>/*"
            )
        full = load_dataset("imagefolder", data_dir=images_dir)["train"]
        if test_ratio > 0.0:
            train_temp = full.train_test_split(test_size=test_ratio, stratify_by_column="label", seed=seed)
            temp = train_temp["train"].train_test_split(test_size=val_ratio, stratify_by_column="label", seed=seed)
            ds = DatasetDict({
                "train": temp["train"],
                "val": temp["test"],
                "test": train_temp["test"]
            })
        else:
            temp = full.train_test_split(test_size=val_ratio, stratify_by_column="label", seed=seed)
            ds = DatasetDict({"train": temp["train"], "val": temp["test"]})
    return ds


# ---------------- AUGMENTATIONS ----------------
def build_train_pil_aug():
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.RandomApply([T.RandomRotation(degrees=7, expand=False, fill=255)], p=0.5),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.1, p=1.0)], p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomApply([T.RandomResizedCrop(size=(896, 896), scale=(0.9, 1.0))], p=0.3),
    ])


def build_val_pil_aug():
    return T.Compose([T.Lambda(lambda im: im.convert("RGB"))])


# ---------------- COLLATOR ----------------
@dataclass
class DocDataCollator:
    processor: AutoImageProcessor
    pil_transform = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [self.pil_transform(f["image"]) if self.pil_transform else f["image"] for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        enc = self.processor(images=images, return_tensors="pt")
        batch = {"pixel_values": enc["pixel_values"], "labels": labels}
        return batch


# ---------------- METRICS ----------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        result = {
            **accuracy.compute(predictions=preds, references=labels),
            **f1.compute(predictions=preds, references=labels, average="macro"),
        }
        return result
    return compute_metrics


# ---------------- TRAINING ----------------
def train_model(
    data_dir: str,
    output_dir: str = "./dit-finetuned-docs",
    checkpoint: str = "microsoft/dit-base",
    epochs: int = 15,
    lr: float = 2e-5,
    train_bs: int = 8,
    eval_bs: int = 8,
    weight_decay: float = 0.01,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    fp16: bool = False,
    no_augment: bool = False,
    freeze_backbone: bool = False,
):
    """Fine-tune Microsoft DiT on document classification."""
    seed_everything(seed)

    # 1) Dataset
    ds = build_dataset(data_dir, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    # 2) Processor & labels
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    label_names = ds["train"].features["label"].names
    id2label = {str(i): n for i, n in enumerate(label_names)}
    label2id = {n: str(i) for i, n in enumerate(label_names)}

    # 3) Model
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # 4) Collators
    train_collator = DocDataCollator(processor=processor)
    val_collator = DocDataCollator(processor=processor)
    train_collator.pil_transform = build_val_pil_aug() if no_augment else build_train_pil_aug()
    val_collator.pil_transform = build_val_pil_aug()

    class EvalCollatorTrainer(Trainer):
        def get_eval_dataloader(self, eval_dataset=None):
            dl = super().get_eval_dataloader(eval_dataset)
            dl.collate_fn = val_collator
            return dl

        def get_test_dataloader(self, test_dataset):
            dl = super().get_test_dataloader(test_dataset)
            dl.collate_fn = val_collator
            return dl

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        fp16=fp16,
        remove_unused_columns=False,
        report_to="none",
    )

    # 6) Trainer
    trainer = EvalCollatorTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=train_collator,
        tokenizer=processor,
        compute_metrics=build_compute_metrics(),
    )

    # 7) Train
    trainer.train()

    # 8) Evaluate
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=ds["test"]) if "test" in ds else None

    # 9) Save everything
    trainer.save_model()
    processor.save_pretrained(output_dir)

    meta = {
        "id2label": id2label,
        "label2id": label2id,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "args": {
            "epochs": epochs, "lr": lr, "train_bs": train_bs, "eval_bs": eval_bs,
            "weight_decay": weight_decay, "seed": seed,
            "val_ratio": val_ratio, "test_ratio": test_ratio,
            "fp16": fp16, "no_augment": no_augment, "freeze_backbone": freeze_backbone,
        }
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Training complete. Model saved to:", output_dir)
    return output_dir


# ---------------- CLI ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dit-finetuned-docs")
    parser.add_argument("--checkpoint", type=str, default="microsoft/dit-base")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        fp16=args.fp16,
        no_augment=args.no_augment,
        freeze_backbone=args.freeze_backbone,
    )
