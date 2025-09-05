# stages/segregate.py
import os
import shutil
import argparse
import pandas as pd
from pathlib import Path


def segregate_pdfs(excel_file: str, main_folder: str, output_base: str) -> None:
    """
    Segregates PDFs from main_folder into SHK, KFL, KFG, Others
    based on Mat. Doc. numbers in Excel sheets.
    """
    # Create output folders
    Path(output_base).mkdir(parents=True, exist_ok=True)
    for sub in ["SHK", "KFL", "KFG", "Others"]:
        os.makedirs(os.path.join(output_base, sub), exist_ok=True)

    # Load Excel sheets
    shk_df = pd.read_excel(excel_file, sheet_name="SHK", engine="openpyxl")
    kfl_df = pd.read_excel(excel_file, sheet_name="KFL", engine="openpyxl")
    kfg_df = pd.read_excel(excel_file, sheet_name="KFG", engine="openpyxl")

    # Extract document IDs
    shk_docs = set(shk_df["Mat. Doc."].astype(str).str.strip())
    kfl_docs = set(kfl_df["Mat. Doc."].astype(str).str.strip())
    kfg_docs = set(kfg_df["Mat. Doc."].astype(str).str.strip())

    # Walk through PDFs in main_folder
    for root, _, files in os.walk(main_folder):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            pdf_number = os.path.splitext(file)[0]  # remove .pdf extension
            src_path = os.path.join(root, file)

            if pdf_number in shk_docs:
                dst = os.path.join(output_base, "SHK", file)
                shutil.move(src_path, dst)
                print(f"Moved {file} → SHK")
            elif pdf_number in kfl_docs:
                dst = os.path.join(output_base, "KFL", file)
                shutil.move(src_path, dst)
                print(f"Moved {file} → KFL")
            elif pdf_number in kfg_docs:
                dst = os.path.join(output_base, "KFG", file)
                shutil.move(src_path, dst)
                print(f"Moved {file} → KFG")
            else:
                dst = os.path.join(output_base, "Others", file)
                shutil.move(src_path, dst)
                print(f"No match for {file}, moved → Others")

    print(f"✅ Segregation complete! Results in: {output_base}")


# ---------------- CLI ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_file", help="Path to Excel file with SHK/KFL/KFG sheets")
    parser.add_argument("main_folder", help="Folder containing PDFs")
    parser.add_argument("output_base", help="Destination folder for segregated PDFs")
    args = parser.parse_args()

    segregate_pdfs(args.excel_file, args.main_folder, args.output_base)
