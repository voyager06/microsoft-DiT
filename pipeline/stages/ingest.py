# stages/ingest.py
import os
import shutil
import json
from pathlib import Path

def extract_images(source_dir: str, destination_dir: str, method: str = "structured") -> int:
    """
    Extract images from invoice directory structure.
    Returns number of images extracted.
    """
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    extracted_count = 0

    if method == "structured":
        for root, dirs, files in os.walk(source_dir):
            if 'config.json' in files:
                invoice_folder = Path(root)
                invoice_number = invoice_folder.name
                for sub in invoice_folder.iterdir():
                    if sub.is_dir() and 'image' in sub.name.lower():
                        for image_file in sub.iterdir():
                            if image_file.suffix.lower() in image_extensions:
                                new_name = f"{invoice_number}_{image_file.name}"
                                shutil.copy2(image_file, Path(destination_dir) / new_name)
                                extracted_count += 1
    elif method == "alternative":
        for item in Path(source_dir).iterdir():
            if item.is_dir():
                invoice_number = item.name
                for img in item.rglob('*'):
                    if img.suffix.lower() in image_extensions and 'pdf' not in img.parent.name.lower():
                        new_name = f"{invoice_number}_{img.name}"
                        shutil.copy2(img, Path(destination_dir) / new_name)
                        extracted_count += 1
    elif method == "config":
        for folder in Path(source_dir).iterdir():
            if folder.is_dir():
                config_path = folder / "config.json"
                invoice_number = folder.name
                if config_path.exists():
                    try:
                        config = json.load(open(config_path, 'r', encoding='utf-8'))
                        invoice_number = config.get("invoice_number", folder.name)
                    except:
                        pass
                for sub in folder.iterdir():
                    if sub.is_dir() and any(k in sub.name.lower() for k in ["img", "image"]):
                        for img in sub.iterdir():
                            if img.suffix.lower() in image_extensions:
                                new_name = f"{invoice_number}_{img.name}"
                                shutil.copy2(img, Path(destination_dir) / new_name)
                                extracted_count += 1
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"✅ Extraction complete: {extracted_count} images saved to {destination_dir}")
    return extracted_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source invoice directory")
    parser.add_argument("dest", help="Destination directory for images")
    parser.add_argument("--method", choices=["structured", "alternative", "config"], default="structured")
    args = parser.parse_args()

    extract_images(args.source, args.dest, args.method)
