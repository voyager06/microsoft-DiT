# stages/inference.py
import os
import json
import argparse
from typing import Dict, Tuple, List

import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ---------------- MODEL UTILS ----------------
def load_model(model_dir: str) -> Tuple[torch.nn.Module, AutoImageProcessor, Dict[str, str]]:
    """
    Load processor, model, and id2label mapping.
    """
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    meta_path = os.path.join(model_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        id2label = meta.get("id2label", model.config.id2label)
    else:
        id2label = model.config.id2label

    return model, processor, id2label


def predict_image(image_path: str, model, processor, id2label: Dict[str, str]) -> str:
    """
    Predict the class of a single document image.
    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
        pred_label = id2label[str(pred_id)]

    return pred_label


def predict_folder(image_dir: str, model, processor, id2label: Dict[str, str], output_csv: str = "predictions.csv") -> str:
    """
    Predict classes for all images in a folder.
    Saves results to CSV and returns path.
    """
    results: List[Dict[str, str]] = []
    supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for fname in os.listdir(image_dir):
        if not any(fname.lower().endswith(ext) for ext in supported_exts):
            continue

        img_path = os.path.join(image_dir, fname)
        try:
            label = predict_image(img_path, model, processor, id2label)
            results.append({"filename": fname, "prediction": label})
            print(f"✅ {fname} → {label}")
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(image_dir, output_csv)
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Predictions saved to: {csv_path}")
        return csv_path
    else:
        print("⚠️ No valid images found.")
        return ""


# ---------------- CLI ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where model is saved")
    parser.add_argument("--image_path", type=str, help="Path to a single image to classify")
    parser.add_argument("--image_dir", type=str, help="Path to a folder of images to classify")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="CSV filename for folder mode")
    args = parser.parse_args()

    model, processor, id2label = load_model(args.model_dir)

    if args.image_path:
        pred = predict_image(args.image_path, model, processor, id2label)
        print(f"✅ Predicted class: {pred}")

    elif args.image_dir:
        predict_folder(args.image_dir, model, processor, id2label, args.output_csv)

    else:
        print("❌ Please provide either --image_path or --image_dir")
