#!/usr/bin/env python3
"""
auto_label.py
─────────────
Automatikus labda-labelező script.

A YOLOv8l modellt (ami MŰKÖDIK) használja arra, hogy automatikusan
labelezze a collect_training_frames.py által gyűjtött képeket.
Ezután a kisebb, gyors YOLOv8n modellt be lehet tanítani ezeken.

Folyamat:
  1. Beolvassa a data/raw_frames/*.jpg fájlokat
  2. YOLOv8l-el detektálja a labdát minden képen
  3. Ellenőrzési ablakban mutatja a detekciót – jóváhagyod / elutasítod
  4. Menti a YOLO formátumú label fájlokat (data/labeled/)

Kimenet könyvtár struktúra (YOLO formátum):
  data/labeled/
    images/
      train/  (80%)
      val/    (20%)
    labels/
      train/
      val/
    dataset.yaml

Használat:
    python scripts/auto_label.py
    python scripts/auto_label.py --conf 0.3          # alacsonyabb conf küszöb
    python scripts/auto_label.py --no-review         # automatikus, kézi ellenőrzés nélkül
    python scripts/auto_label.py --model yolov8l.pt  # más modell

Billentyűk a review ablakban:
    SPACE / y  – elfogad (menti a labelt)
    n          – elutasít (nem menti)
    q          – kilépés
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_DIR  = ROOT / "data" / "raw_frames"
OUTPUT_DIR = ROOT / "data" / "labeled"
YAML_PATH  = OUTPUT_DIR / "dataset.yaml"

# YOLO class index a COCO-ban: 32 = sports ball
_COCO_BALL_CLASS = 32


def run_auto_label(model_path: str, conf_threshold: float, no_review: bool) -> None:
    """Főfolyamat: betölti a képeket, detektál, review, ment."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics nincs telepítve! Futtasd: pip install ultralytics")
        sys.exit(1)

    # Képek keresése
    images = sorted(list(INPUT_DIR.glob("*.jpg")) +
                    list(INPUT_DIR.glob("*.png")) +
                    list(INPUT_DIR.glob("*.jpeg")))

    if not images:
        log.error("Nincs kép a %s könyvtárban!", INPUT_DIR)
        log.error("Először futtasd: python scripts/collect_training_frames.py")
        sys.exit(1)

    log.info("Talált képek: %d (forrás: %s)", len(images), INPUT_DIR)
    log.info("YOLO modell betöltése: %s", model_path)

    model = YOLO(model_path)

    # Kimenet könyvtárak létrehozása
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    skipped  = 0

    if not no_review:
        win_name = "Auto-Label Review | SPACE=elfogad | n=elutasit | q=kilepes"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 540)

    # Keverés a train/val szétválasztáshoz
    images_shuffled = list(images)
    random.shuffle(images_shuffled)

    for i, img_path in enumerate(images_shuffled):
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning("Nem sikerült beolvasni: %s", img_path)
            skipped += 1
            continue

        h, w = frame.shape[:2]

        # Detektálás YOLOv8l-el
        results = model.predict(frame, verbose=False, conf=conf_threshold, classes=[_COCO_BALL_CLASS])
        result = results[0]

        # Legjobb detekció keresése (sports ball, class 32)
        best_box = _find_best_ball(result, conf_threshold)

        if best_box is None:
            log.debug("Nem talált labdát: %s", img_path.name)
            if not no_review:
                # Megmutatjuk a képet, de jelezzük hogy nem detektált
                display = frame.copy()
                _draw_no_detection(display, img_path.name, i, len(images_shuffled))
                cv2.imshow(win_name, display)
                key = cv2.waitKey(300) & 0xFF
                if key == ord('q'):
                    log.info("Kilépés felhasználói kérésre.")
                    break
            skipped += 1
            continue

        # Vizualizáció
        display = frame.copy()
        _draw_detection_preview(display, best_box, img_path.name, i, len(images_shuffled),
                                 accepted, rejected)

        if no_review:
            # Automatikus elfogadás
            accept = True
        else:
            cv2.imshow(win_name, display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                log.info("Kilépés felhasználói kérésre.")
                break
            accept = key in (ord(' '), ord('y'), 13)  # SPACE, y, Enter

        if accept:
            # 80/20 train/val szétválasztás
            split = "train" if (accepted + rejected) % 5 != 4 else "val"

            # Kép másolása
            dst_img = OUTPUT_DIR / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # Label mentése YOLO formátumban
            x1, y1, x2, y2, _conf = best_box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw_n = (x2 - x1) / w
            bh_n = (y2 - y1) / h

            label_name = img_path.stem + ".txt"
            dst_lbl = OUTPUT_DIR / "labels" / split / label_name
            with open(dst_lbl, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}\n")

            accepted += 1
            log.info("[%d/%d] ELFOGADVA → %s | conf: %.2f",
                     i+1, len(images_shuffled), split, best_box[4] if len(best_box) > 4 else 0)
        else:
            rejected += 1
            log.info("[%d/%d] ELUTASÍTVA: %s", i+1, len(images_shuffled), img_path.name)

    if not no_review:
        cv2.destroyAllWindows()

    # Dataset YAML írása
    _write_dataset_yaml()

    log.info("=" * 60)
    log.info("Labelezés kész!")
    log.info("  Elfogadva : %d kép", accepted)
    log.info("  Elutasítva: %d kép", rejected)
    log.info("  Kihagyva  : %d kép (nem detektált)", skipped)
    log.info("  YAML      : %s", YAML_PATH)
    log.info("=" * 60)
    log.info("")
    log.info("Következő lépés – tanítás:")
    log.info("  python scripts/train_custom_ball.py")


def _find_best_ball(result, conf_threshold: float):
    """Visszaadja a legjobb sports ball bounding box-ot [x1, y1, x2, y2, conf]."""
    best_conf = 0.0
    best_box = None

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # Elfogadunk COCO sports ball-t (32) vagy bármilyen osztályt ha csak 1 osztályos modell
        if cls not in (_COCO_BALL_CLASS, 0):
            continue
        if conf < conf_threshold:
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        bw = x2 - x1
        bh = y2 - y1
        # Kizárjuk az extrém méretű / arányú bounding boxokat
        if bw < 5 or bh < 5:
            continue
        aspect = bw / bh if bh > 0 else 0
        if not (0.3 < aspect < 3.0):
            continue
        if conf > best_conf:
            best_conf = conf
            best_box = [x1, y1, x2, y2, conf]

    return best_box


def _draw_detection_preview(frame: np.ndarray, box, name: str,
                             idx: int, total: int, accepted: int, rejected: int) -> None:
    """Vizualizáció a review ablakhoz."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    conf = box[4] if len(box) > 4 else 0.0

    # Zöld bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 5, (0, 255, 80), -1)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 80), 2)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 80), 2)

    label = f"ball {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 200, 60), -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Info sáv
    info = f"[{idx+1}/{total}]  Elfogadva: {accepted}  Elutasítva: {rejected}  |  {name}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Útmutató
    hint = "SPACE / y = elfogad    n = elutasit    q = kilepes"
    cv2.rectangle(frame, (0, frame.shape[0] - 38), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(frame, hint, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 230, 255), 1, cv2.LINE_AA)


def _draw_no_detection(frame: np.ndarray, name: str, idx: int, total: int) -> None:
    """Vizualizáció ha nincs detekció."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 60), -1)
    cv2.putText(frame, f"[{idx+1}/{total}]  NEM DETEKTÁLT – {name}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 1, cv2.LINE_AA)


def _write_dataset_yaml() -> None:
    """Elmenti a dataset.yaml fájlt a tanításhoz."""
    import yaml
    content = {
        "path": str(OUTPUT_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc": 1,
        "names": ["ball"],
    }
    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(content, f, default_flow_style=False)
    log.info("Dataset YAML mentve: %s", YAML_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-labelező: YOLOv8l-el labelezi a saját képeidet"
    )
    parser.add_argument("--model", default="yolov8l.pt",
                        help="YOLO modell elérési út (default: yolov8l.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Konfidencia küszöb (default: 0.25)")
    parser.add_argument("--no-review", action="store_true",
                        help="Kézi ellenőrzés kihagyása (minden detektált kép automatikusan elfogadva)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("AUTO-LABELER – YOLOv8l alapú automatikus labelezés")
    log.info("Forrás: %s", INPUT_DIR)
    log.info("Kimenet: %s", OUTPUT_DIR)
    log.info("Modell: %s | Conf küszöb: %.2f", args.model, args.conf)
    log.info("=" * 60)

    if not INPUT_DIR.exists() or not any(INPUT_DIR.glob("*.jpg")):
        log.error("Nincsenek képek a %s könyvtárban!", INPUT_DIR)
        log.error("Először gyűjts képeket: python scripts/collect_training_frames.py")
        sys.exit(1)

    run_auto_label(
        model_path=args.model,
        conf_threshold=args.conf,
        no_review=args.no_review,
    )


if __name__ == "__main__":
    main()
