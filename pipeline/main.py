# main.py
import argparse
import yaml

from stages import ingest, ocr_classify, inference, train_classifier, segregate, model_utils


def run_pipeline(config, stage=None):
    """
    Run full pipeline or a specific stage.
    """
    if stage in [None, "ingest"]:
        ingest.extract_images(config["source_dir"], config["image_dir"], method=config.get("ingest_method", "structured"))

    if stage in [None, "ocr_classify"]:
        ocr_classify.classify_folder(config["pdf_dir"], config["ocr_output"], recursive=config.get("ocr_recursive", False))

    if stage in [None, "train_classifier"]:
        train_classifier.train_model(
            data_dir=config["train_data_dir"],
            output_dir=config.get("model_output_dir", "./dit-finetuned-docs"),
            epochs=config.get("epochs", 10),
            lr=config.get("lr", 2e-5),
            train_bs=config.get("train_bs", 8),
            eval_bs=config.get("eval_bs", 8),
            fp16=config.get("fp16", False),
            no_augment=config.get("no_augment", False),
            freeze_backbone=config.get("freeze_backbone", False),
        )

    if stage in [None, "inference"]:
        model, processor, id2label = inference.load_model(config["model_dir"])
        inference.predict_folder(config["image_dir"], model, processor, id2label, config.get("predictions_csv", "predictions.csv"))

    if stage in [None, "segregate"]:
        segregate.segregate_pdfs(config["excel_file"], config["pdf_dir"], config["segregated_output"])

    if stage in [None, "model_utils"]:
        model_utils.download_and_save(config.get("hf_model", "microsoft/dit-base-finetuned-rvlcdip"),
                                      config.get("model_output_dir", "./microsoft_dit"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["ingest", "ocr_classify", "train_classifier", "inference", "segregate", "model_utils"],
                        help="Run only a specific stage")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config, stage=args.stage)
