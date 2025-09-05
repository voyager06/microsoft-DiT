# stages/ocr_classify.py
import os
import csv
import argparse
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False


# ---------------- CONFIG ----------------
KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Tax_Invoices", ["tax invoice", "gst invoice", "tax-invoice"]),
    ("Bill_of_Entry", ["bill of entry", "boe"]),
    ("Bill_of_Lading", ["bill of lading", "b/l", "bol"]),
    ("Invoices", ["invoice", "commercial invoice"]),
]
CLASSES = [c[0] for c in KEYWORDS] + ["Others"]

DPI = 300
HEADER_HEIGHT_FRAC = 0.35
FUZZY_THRESHOLD = 85


# ---------------- HELPERS ----------------
def normalize(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def any_keyword_in_text(text: str) -> Optional[Tuple[str, str, int]]:
    """Match OCR text against keyword variants with optional fuzzy scoring."""
    norm_text = normalize(text)
    best = (None, None, 0)

    for class_name, variants in KEYWORDS:
        for kw in variants:
            norm_kw = normalize(kw)
            if HAVE_RAPIDFUZZ:
                score = fuzz.partial_ratio(norm_kw, norm_text)
            else:
                score = 100 if norm_kw in norm_text else 0
            if score > best[2]:
                best = (class_name, kw, score)

        if best[0] == class_name and best[2] == 100:
            return best

    if HAVE_RAPIDFUZZ and best[2] < FUZZY_THRESHOLD:
        return None
    if not HAVE_RAPIDFUZZ and best[2] == 0:
        return None
    return best


def render_page_to_image(page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def crop_header(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    header_h = int(h * frac)
    return img.crop((0, 0, w, header_h))


# ---------------- CORE FUNCTION ----------------
def classify_pdf_pages(pdf_path: str, out_dir: str) -> str:
    """
    Classify pages of a single PDF into classes (by OCR keyword match).
    Saves images into class subfolders + CSV log.
    Returns path to CSV log file.
    """
    os.makedirs(out_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    log_path = os.path.join(out_dir, f"{base}_log.csv")

    with fitz.open(pdf_path) as doc, open(log_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["pdf", "page", "class", "matched_keyword", "score"])

        for i, page in enumerate(doc):
            full_img = render_page_to_image(page, DPI)
            header_img = crop_header(full_img, HEADER_HEIGHT_FRAC)

            header_text = pytesseract.image_to_string(header_img, config="--psm 6")
            hit = any_keyword_in_text(header_text)

            if not hit:
                full_text = pytesseract.image_to_string(full_img, config="--psm 6")
                hit = any_keyword_in_text(full_text)

            if hit:
                class_name, kw, score = hit
            else:
                class_name, kw, score = ("Others", "", 0)

            out_name = f"{base}_p{i+1:03d}.png"
            out_path = os.path.join(out_dir, class_name, out_name)
            full_img.save(out_path, "PNG")

            writer.writerow([os.path.basename(pdf_path), i, class_name, kw, score])
            print(f"[{base} p{i+1:03d}] -> {class_name} (kw='{kw}', score={score})")

    print(f"✅ OCR classification done for {pdf_path}. Log saved at {log_path}")
    return log_path


def classify_folder(pdf_dir: str, out_dir: str, recursive: bool = False) -> None:
    """Process all PDFs in a folder (optionally recursive)."""
    os.makedirs(out_dir, exist_ok=True)

    for root, _, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                classify_pdf_pages(os.path.join(root, f), out_dir)
        if not recursive:
            break


# ---------------- CLI ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="PDF file or folder")
    parser.add_argument("-o", "--out", default="classified_pages", help="Output dir")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search subfolders recursively")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        classify_pdf_pages(args.input, args.out)
    else:
        classify_folder(args.input, args.out, recursive=args.recursive)
