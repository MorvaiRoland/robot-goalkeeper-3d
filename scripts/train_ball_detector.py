#!/usr/bin/env python3
"""
Fine-tune YOLOv8n for soccer/sports ball detection.

Dataset strategy
────────────────
Primary  : artemstakheev/ball-tracking  (Roboflow Universe, public)
Secondary: COCO 2017 val images with 'sports ball' class (class 32)
           Downloaded automatically via fiftyone (optional, can be skipped).

Output
──────
models/ball_detector.pt   – fine-tuned YOLOv8n weights
models/ball_detector_training/  – training run artefacts (metrics, curves)

Usage
─────
python3 scripts/train_ball_detector.py
python3 scripts/train_ball_detector.py --epochs 100 --no-coco
python3 scripts/train_ball_detector.py --roboflow-key YOUR_KEY  # faster download
"""

import argparse
import os
import shutil
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "ball_training"
MODELS_DIR = ROOT / "models"
OUTPUT_PT  = MODELS_DIR / "ball_detector.pt"
YAML_PATH  = DATA_DIR / "dataset.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dependency check
# ──────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> None:
    missing = []
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        missing.append("ultralytics")
    try:
        import roboflow  # noqa: F401
    except ImportError:
        missing.append("roboflow")
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")

    if missing:
        log.error("Missing packages: %s", ", ".join(missing))
        log.error("Install with:  pip install %s", " ".join(missing))
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Download Roboflow dataset (public, no API key required for public datasets)
# ──────────────────────────────────────────────────────────────────────────────

def download_roboflow(api_key: str, dest: Path) -> Path:
    """
    Downloads artemstakheev/ball-tracking from Roboflow Universe.

    Public datasets can be downloaded with an empty string as api_key,
    but a personal (free) key speeds things up and removes rate limits.
    Sign up free at: https://app.roboflow.com
    """
    from roboflow import Roboflow  # type: ignore[import]

    log.info("Connecting to Roboflow…")
    rf = Roboflow(api_key=api_key if api_key else "YOUR_ROBOFLOW_KEY")

    log.info("Downloading artemstakheev/ball-tracking (version 1)…")
    project = rf.workspace("artemstakheev").project("ball-tracking")
    version = project.version(1)

    rf_dest = dest / "roboflow_raw"
    rf_dest.mkdir(parents=True, exist_ok=True)

    dataset = version.download("yolov8", location=str(rf_dest))
    log.info("Roboflow dataset downloaded to: %s", rf_dest)
    return rf_dest


def download_roboflow_no_key(dest: Path) -> Path:
    """
    Fallback: download the public dataset without authentication
    using the Roboflow Universe export URL (YOLOv8 format ZIP).
    """
    import urllib.request
    import zipfile

    # Public export link for artemstakheev/ball-tracking v1 (YOLOv8 format)
    url = (
        "https://universe.roboflow.com/ds/wXJpBpnFk1"
        "?key=artemstakheev_ball_tracking"
    )

    rf_dest = dest / "roboflow_raw"
    zip_path = dest / "roboflow.zip"

    log.info("Attempting public download of artemstakheev/ball-tracking…")
    log.info("URL: %s", url)

    try:
        urllib.request.urlretrieve(url, zip_path)
        rf_dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(rf_dest)
        zip_path.unlink()
        log.info("Dataset extracted to: %s", rf_dest)
        return rf_dest
    except Exception as exc:
        log.warning("Public download failed (%s). Using API key method.", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3. Optionally augment with COCO sports-ball images
# ──────────────────────────────────────────────────────────────────────────────

def download_coco_balls(dest: Path, max_images: int = 500) -> Path:
    """
    Pull up to `max_images` COCO 2017 val images that contain a sports ball
    (class 32) using the fiftyone library, and convert annotations to YOLO format.
    """
    try:
        import fiftyone as fo                        # type: ignore[import]
        import fiftyone.zoo as foz                   # type: ignore[import]
    except ImportError:
        log.warning(
            "fiftyone not installed – skipping COCO augmentation.\n"
            "Install with:  pip install fiftyone"
        )
        return None

    log.info("Downloading COCO 2017 val sports-ball images via fiftyone…")
    coco_dest = dest / "coco_balls"
    coco_dest.mkdir(parents=True, exist_ok=True)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["sports ball"],
        max_samples=max_images,
        dataset_dir=str(coco_dest / "raw"),
    )

    # Export as YOLO format
    export_path = coco_dest / "yolo"
    dataset.export(
        export_dir=str(export_path),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=["sports ball"],
    )
    log.info("COCO sports-ball images exported to: %s", export_path)
    return export_path


# ──────────────────────────────────────────────────────────────────────────────
# 4. Merge datasets and write dataset.yaml
# ──────────────────────────────────────────────────────────────────────────────

def merge_datasets(rf_path: Path, coco_path: Path, dest: Path) -> None:
    """
    Merge Roboflow and (optional) COCO ball images into a unified YOLO dataset.
    Structure:
        dest/
          images/train/   images/val/   images/test/
          labels/train/   labels/val/   labels/test/
    """
    import yaml

    for split in ("train", "valid", "test"):
        yolo_split = "val" if split == "valid" else split
        (dest / "images" / yolo_split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / yolo_split).mkdir(parents=True, exist_ok=True)

    # ── Copy Roboflow data ────────────────────────────────────────────────────
    if rf_path is not None and rf_path.exists():
        log.info("Merging Roboflow data…")
        for split in ("train", "valid", "test"):
            yolo_split = "val" if split == "valid" else split
            for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
                for img in (rf_path / split / "images").glob(f"*.{ext}"):
                    shutil.copy2(img, dest / "images" / yolo_split / img.name)
            for lbl in (rf_path / split / "labels").glob("*.txt"):
                # Remap class index to 0 (ball), re-write label files
                _remap_label(lbl, dest / "labels" / yolo_split / lbl.name)
    else:
        log.warning("Roboflow dataset not available – proceeding without it.")

    # ── Copy COCO ball data ───────────────────────────────────────────────────
    if coco_path is not None and coco_path.exists():
        log.info("Merging COCO sports-ball data…")
        for split in ("train", "val"):
            src_img = coco_path / "images" / split
            src_lbl = coco_path / "labels" / split
            if src_img.exists():
                for img in src_img.iterdir():
                    shutil.copy2(img, dest / "images" / split / img.name)
            if src_lbl.exists():
                for lbl in src_lbl.glob("*.txt"):
                    _remap_label(lbl, dest / "labels" / split / lbl.name)

    # ── Count and report ──────────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        n = len(list((dest / "images" / split).glob("*")))
        log.info("  %s: %d images", split, n)

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    yaml_content = {
        "path": str(dest),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc": 1,
        "names": ["ball"],
    }
    with open(YAML_PATH, "w") as fh:
        yaml.safe_dump(yaml_content, fh)
    log.info("Dataset YAML written: %s", YAML_PATH)


def _remap_label(src: Path, dst: Path) -> None:
    """
    Copy a YOLO label file, forcing every class index to 0 (ball).
    This normalises datasets that use different class indices for the ball.
    """
    lines_out = []
    try:
        with open(src) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace class index with 0, keep bbox unchanged
                    lines_out.append("0 " + " ".join(parts[1:]))
    except Exception as exc:
        log.warning("Could not process label %s: %s", src, exc)
        return

    with open(dst, "w") as fh:
        fh.write("\n".join(lines_out))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Train YOLOv8n
# ──────────────────────────────────────────────────────────────────────────────

def train(epochs: int, imgsz: int, batch: int, device: str) -> Path:
    """Fine-tune YOLOv8n on the merged dataset."""
    from ultralytics import YOLO  # type: ignore[import]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Starting YOLOv8n fine-tuning")
    log.info("  Dataset  : %s", YAML_PATH)
    log.info("  Epochs   : %d", epochs)
    log.info("  Image sz : %d", imgsz)
    log.info("  Batch    : %d", batch)
    log.info("  Device   : %s", device)
    log.info("=" * 60)

    model = YOLO("yolov8n.pt")   # pre-trained COCO weights → transfer learning

    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(MODELS_DIR / "ball_detector_training"),
        name="run",
        exist_ok=True,
        # Augmentation – helps a lot with only 74 base images
        hsv_h=0.015,      # hue jitter (small – ball is white/round)
        hsv_s=0.5,        # saturation jitter
        hsv_v=0.4,        # value/brightness jitter (outdoor light changes)
        degrees=10.0,     # rotation
        translate=0.1,
        scale=0.5,        # scale jitter – simulates different distances
        fliplr=0.5,
        mosaic=1.0,       # mosaic augmentation (4-image composite)
        mixup=0.1,
        copy_paste=0.1,
        # Regularisation
        dropout=0.1,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,      # early stopping
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        verbose=True,
    )

    # Copy best weights to the canonical output path
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, OUTPUT_PT)
        log.info("✅ Best model saved → %s", OUTPUT_PT)
    else:
        log.warning("best.pt not found – check training output in %s", results.save_dir)

    return OUTPUT_PT


# ──────────────────────────────────────────────────────────────────────────────
# 6. Validate the trained model
# ──────────────────────────────────────────────────────────────────────────────

def validate(model_path: Path) -> None:
    """Run validation metrics on the test split."""
    from ultralytics import YOLO  # type: ignore[import]

    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        return

    log.info("Running validation on test split…")
    model  = YOLO(str(model_path))
    metrics = model.val(data=str(YAML_PATH), split="test", verbose=True)

    log.info("=" * 60)
    log.info("Validation results")
    log.info("  mAP50        : %.4f", metrics.box.map50)
    log.info("  mAP50-95     : %.4f", metrics.box.map)
    log.info("  Precision    : %.4f", metrics.box.mp)
    log.info("  Recall       : %.4f", metrics.box.mr)
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n ball detector (artemstakheev/ball-tracking)"
    )
    parser.add_argument(
        "--roboflow-key", default="",
        help="Roboflow API key (optional – leave empty for public dataset access). "
             "Get a free key at https://app.roboflow.com"
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size for training (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16, reduce if OOM)"
    )
    parser.add_argument(
        "--device", default="0",
        help="Device: '0' = first GPU, 'cpu' = CPU (default: '0')"
    )
    parser.add_argument(
        "--no-coco", action="store_true",
        help="Skip COCO sports-ball augmentation (faster setup, less data)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip training, only run validation on existing model"
    )
    args = parser.parse_args()

    check_dependencies()

    if args.validate_only:
        validate(OUTPUT_PT)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download Roboflow dataset ─────────────────────────────────────
    rf_path = None
    if args.roboflow_key:
        rf_path = download_roboflow(args.roboflow_key, DATA_DIR)
    else:
        log.info(
            "No Roboflow API key provided.\n"
            "  → Attempting public download (may fail for private datasets).\n"
            "  → Get a FREE key at https://app.roboflow.com and re-run with:\n"
            "       python3 scripts/train_ball_detector.py --roboflow-key YOUR_KEY\n"
        )
        rf_path = download_roboflow_no_key(DATA_DIR)

        if rf_path is None:
            log.warning(
                "Could not auto-download the dataset.\n"
                "Please download it manually:\n"
                "  1. Go to https://universe.roboflow.com/artemstakheev/ball-tracking\n"
                "  2. Click Export → YOLOv8 format → Download ZIP\n"
                "  3. Extract into:  %s/roboflow_raw/\n"
                "  4. Re-run this script.",
                DATA_DIR,
            )

    # ── Step 2: Optionally augment with COCO sports-ball images ───────────────
    coco_path = None
    if not args.no_coco:
        coco_path = download_coco_balls(DATA_DIR, max_images=500)
    else:
        log.info("Skipping COCO augmentation (--no-coco).")

    # ── Step 3: Merge into unified dataset ────────────────────────────────────
    merge_datasets(rf_path, coco_path, DATA_DIR)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    model_path = train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # ── Step 5: Validate ──────────────────────────────────────────────────────
    validate(model_path)

    log.info("")
    log.info("🎯 Done! To use the trained model, update config/system_config.yaml:")
    log.info("     detection:")
    log.info("       yolo_model_path: 'models/ball_detector.pt'")


if __name__ == "__main__":
    main()
