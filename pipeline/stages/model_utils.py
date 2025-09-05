# stages/model_utils.py
import os
import argparse
from transformers import AutoModel, AutoProcessor


def download_and_save(model_name: str = "microsoft/dit-base-finetuned-rvlcdip", save_path: str = "./microsoft_dit"):
    """
    Download a pretrained model and save it locally.
    """
    os.makedirs(save_path, exist_ok=True)

    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print(f"✅ Model and processor saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/dit-base-finetuned-rvlcdip", help="Model name from HuggingFace Hub")
    parser.add_argument("--save_path", type=str, default="./microsoft_dit", help="Local path to save model")
    args = parser.parse_args()

    download_and_save(args.model_name, args.save_path)
