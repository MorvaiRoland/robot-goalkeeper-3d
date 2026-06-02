#!/usr/bin/env python3
"""
train_custom_ball.py
────────────────────
YOLOv8n betanítása a SAJÁT labda képeiden (auto_label.py kimenetén).

Ez az ajánlott workflow:
  1. python scripts/collect_training_frames.py   ← képgyűjtés kameráról
  2. python scripts/auto_label.py                ← automatikus labelezés YOLOv8l-el
  3. python scripts/train_custom_ball.py         ← YOLOv8n betanítása  ← TE ITT VAGY

Miért működik jól?
  - YOLOv8n nagyon gyors (GTX 1660Ti-n ~80-100 det-FPS)
  - A saját képeidre tanítva sokkal pontosabb mint az általános COCO modell
  - Transfer learning: a YOLOv8n COCO súlyokból indul, így kevés adat is elég
  - Csak 1 osztályt tanul: "ball" → nincs zaj más osztályoktól

Várható eredmény:
  - GTX 1660Ti: ~80-100 det-FPS (batch stereo: ~50-60 FPS pár)
  - Pontosság: >90% mAP50 saját labdán

Használat:
    python scripts/train_custom_ball.py
    python scripts/train_custom_ball.py --epochs 100
    python scripts/train_custom_ball.py --model yolov8s.pt   # kicsit pontosabb, kicsit lassabb
    python scripts/train_custom_ball.py --imgsz 416          # kisebb képméret → gyorsabb

Kimenet:
    models/custom_ball.pt   ← ezt kell a config-ban beállítani!
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
LABELED    = ROOT / "data" / "labeled"
YAML_PATH  = LABELED / "dataset.yaml"
MODELS_DIR = ROOT / "models"
OUTPUT_PT  = MODELS_DIR / "custom_ball.pt"


def check_dataset() -> bool:
    """Ellenőrzi hogy megvan-e a labeled dataset."""
    if not YAML_PATH.exists():
        log.error("Hiányzik: %s", YAML_PATH)
        log.error("Futtasd először: python scripts/auto_label.py")
        return False

    train_imgs = list((LABELED / "images" / "train").glob("*"))
    val_imgs   = list((LABELED / "images" / "val").glob("*"))

    if len(train_imgs) < 10:
        log.error("Túl kevés training kép: %d (minimum 10 kell)", len(train_imgs))
        log.error("Gyűjts több képet: python scripts/collect_training_frames.py")
        return False

    log.info("Dataset OK: %d train, %d val kép", len(train_imgs), len(val_imgs))
    return True


def train(base_model: str, epochs: int, imgsz: int, batch: int, device: str) -> Path:
    """YOLOv8n fine-tuning a saját labda dataset-en."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics nincs telepítve: pip install ultralytics")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("YOLOv8n CUSTOM TRAINING")
    log.info("  Alap modell: %s", base_model)
    log.info("  Dataset    : %s", YAML_PATH)
    log.info("  Epochs     : %d", epochs)
    log.info("  Képméret   : %d", imgsz)
    log.info("  Batch      : %d", batch)
    log.info("  Device     : %s", device)
    log.info("=" * 60)

    model = YOLO(base_model)

    # Egyéni tanítási könyvtár a saját futáshoz
    run_dir = MODELS_DIR / "custom_ball_training"

    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(run_dir),
        name="run",
        exist_ok=True,

        # ── Augmentáció (labda specifikus) ───────────────────────────
        # Szín jitter: fehér labdánál csak V (fényesség) számít igazán
        hsv_h=0.01,       # minimális hue jitter (fehér labdánál nem számít)
        hsv_s=0.4,        # szaturáció (fény változáshoz)
        hsv_v=0.5,        # fényesség (legfontosabb!)
        # Geometriai augmentáció
        degrees=15.0,     # forgatás (labda gömb, minden irányból ugyanolyan)
        translate=0.15,   # eltolás (labda a kép különböző részein lehet)
        scale=0.6,        # méret változás (közeli/távolabb labda)
        fliplr=0.5,       # vízszintes tükrözés
        flipud=0.1,       # függőleges tükrözés (ritkább)
        # Összetett augmentáció
        mosaic=1.0,       # mozaik (4 kép összerakva) – kevés adatnál nagyon hasznos
        mixup=0.15,       # mixup augmentáció
        copy_paste=0.1,   # copy-paste augmentáció

        # ── Regularizáció ──────────────────────────────────────────
        dropout=0.0,      # YOLOv8n-nél a dropout általában nem segít
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=40,      # early stopping (ha 40 epoch alatt nem javul)

        # ── Optimalizáló ───────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,        # kezdeti tanulási ráta
        lrf=0.01,         # végső lr arány (lr0 * lrf lesz a végső lr)
        momentum=0.937,

        # ── Egyéb ──────────────────────────────────────────────────
        verbose=True,
        plots=True,       # training görbék mentése
        save=True,
        save_period=10,   # checkpoint mentés minden 10. epoch-ban
    )

    # Legjobb súlyok mentése a canonical helyre
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, OUTPUT_PT)
        log.info("✅ Legjobb modell mentve: %s", OUTPUT_PT)
    else:
        log.warning("best.pt nem található: %s", results.save_dir)

    return OUTPUT_PT


def validate(model_path: Path) -> None:
    """Validáció a val split-en."""
    from ultralytics import YOLO

    if not model_path.exists():
        log.error("Modell nem található: %s", model_path)
        return

    log.info("Validáció futtatása (%s)...", model_path)
    model = YOLO(str(model_path))
    metrics = model.val(data=str(YAML_PATH), verbose=True)

    log.info("=" * 60)
    log.info("VALIDÁCIÓS EREDMÉNYEK")
    log.info("  mAP50    : %.4f  (célérték: >0.85)", metrics.box.map50)
    log.info("  mAP50-95 : %.4f", metrics.box.map)
    log.info("  Precision: %.4f", metrics.box.mp)
    log.info("  Recall   : %.4f", metrics.box.mr)
    log.info("=" * 60)

    if metrics.box.map50 >= 0.85:
        log.info("✅ Kiváló eredmény! A modell kész a használatra.")
    elif metrics.box.map50 >= 0.70:
        log.info("⚠️  Elfogadható, de több adat segítene (collect_training_frames.py).")
    else:
        log.warning("❌ Gyenge eredmény. Gyűjts több képet és labelezz pontosabban!")


def benchmark_fps(model_path: Path, device: str) -> None:
    """Gyors FPS becslés – megmutatja milyen fast lesz az éles detektálásnál."""
    import time
    from ultralytics import YOLO
    import numpy as np

    if not model_path.exists():
        return

    log.info("FPS becslés futtatása...")
    model = YOLO(str(model_path))

    # Warm-up
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, verbose=False, device=device)

    # Benchmark – sztereó batch (2 kép egyszerre, mint az éles kódban)
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict([dummy, dummy], verbose=False, device=device)
    dt = time.perf_counter() - t0

    fps_stereo = n / dt
    fps_single = fps_stereo * 2  # 2 kép / batch

    log.info("=" * 60)
    log.info("FPS BECSLÉS (stereo batch)")
    log.info("  Stereo det-FPS : %.1f  (2 kép / inference)", fps_stereo)
    log.info("  Egyenértékű FPS: %.1f  kép/sec", fps_single)

    if fps_stereo >= 50:
        log.info("✅ 50+ stereo FPS – kiváló, eléri a célt!")
    elif fps_stereo >= 30:
        log.info("⚠️  30-50 stereo FPS – elfogadható, de növeld a kamera FPS-t is.")
    else:
        log.warning("❌ <30 stereo FPS – próbáld --imgsz 416 paraméterrel újra tanítani.")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8n tanítása saját labda képeken"
    )
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Alap YOLO modell (default: yolov8n.pt). "
                             "Alternatíva: yolov8s.pt (kicsit pontosabb, lassabb)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Tanítási epoch-ok száma (default: 150)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kép méret a tanításhoz (default: 640). "
                             "416 gyorsabb, 640 pontosabb")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch méret (default: 16, csökkentsd ha nincs elég VRAM)")
    parser.add_argument("--device", default="0",
                        help="Eszköz: '0'=első GPU, 'cpu'=processzor (default: '0')")
    parser.add_argument("--validate-only", action="store_true",
                        help="Csak validáció, tanítás kihagyása")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="FPS becslés kihagyása")
    args = parser.parse_args()

    if args.validate_only:
        validate(OUTPUT_PT)
        return

    if not check_dataset():
        sys.exit(1)

    model_path = train(
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    validate(model_path)

    if not args.no_benchmark:
        benchmark_fps(model_path, args.device)

    log.info("")
    log.info("🎯 Kész! Frissítsd a config/system_config.yaml fájlt:")
    log.info("   detection:")
    log.info("     yolo_model_path: 'models/custom_ball.pt'")
    log.info("     method: 'yolo'")
    log.info("     confidence_threshold: 0.5")


if __name__ == "__main__":
    main()
