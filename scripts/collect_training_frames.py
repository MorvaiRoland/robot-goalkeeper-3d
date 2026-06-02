#!/usr/bin/env python3
"""
collect_training_frames.py
──────────────────────────
Képeket gyűjt a XIMEA kamerákból a labda detektálás betanításához.

Használat:
    python scripts/collect_training_frames.py
    python scripts/collect_training_frames.py --camera 0      # csak bal kamera
    python scripts/collect_training_frames.py --interval 0.5  # 0.5 mp-enként ment
    python scripts/collect_training_frames.py --max-frames 500

Billentyűk futás közben:
    SPACE  – kép mentése (manuális)
    a      – automatikus gyűjtés be/ki
    q      – kilépés

Kimenet:  data/raw_frames/  (jpg képek)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Projekt gyökér meghatározása
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "data" / "raw_frames"


def collect_from_ximea(camera_index: int, interval: float, max_frames: int) -> None:
    """Képgyűjtés XIMEA kamerából."""
    try:
        from detection.camera import XimeaCamera
    except ImportError:
        log.error("Nem sikerült betölteni a camera modult. Futtatd a projekt gyökeréből!")
        sys.exit(1)

    # Config betöltés
    config_path = ROOT / "config" / "system_config.yaml"
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        cam_cfg = config["camera"]
    except Exception:
        log.warning("Config betöltés sikertelen, alapértelmezett értékek használva.")
        cam_cfg = {"resolution": {"width": 1280, "height": 720},
                   "exposure_time_us": 800, "gain": 0.0, "bandwidth_limit_mbs": 160}

    w = cam_cfg["resolution"]["width"]
    h = cam_cfg["resolution"]["height"]
    exp = cam_cfg.get("exposure_time_us", 800)
    gain = cam_cfg.get("gain", 0.0)
    bw = cam_cfg.get("bandwidth_limit_mbs", 160)

    cam = XimeaCamera(camera_index, w, h, exp, gain, bandwidth_limit_mbs=bw)
    log.info("Kamera megnyitása (index=%d)...", camera_index)
    if not cam.open():
        log.error("Nem sikerült megnyitni a kamerát!")
        sys.exit(1)

    _run_collection_loop(cam, interval, max_frames, f"cam{camera_index}")
    cam.close()


def collect_from_mock(interval: float, max_frames: int) -> None:
    """Képgyűjtés MockCamera-ból (teszteléshez)."""
    sys.path.insert(0, str(ROOT / "src"))
    from detection.camera import MockCamera

    cam = MockCamera(width=1280, height=720, fps=60, is_left=True)
    cam.open()
    log.info("Mock kamera megnyitva (szimulált labda).")
    _run_collection_loop(cam, interval, max_frames, "mock")
    cam.close()


def _run_collection_loop(cam, interval: float, max_frames: int, prefix: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    auto_mode = False
    last_save_t = 0.0

    win_name = f"Képgyűjtő – {prefix} | SPACE=ment | a=auto | q=quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    log.info("Képgyűjtés indítva. Kimenet: %s", OUTPUT_DIR)
    log.info("SPACE = kézi mentés | a = automata be/ki | q = kilépés")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        display = frame.copy()

        # Overlay
        mode_text = "AUTO" if auto_mode else "MANUAL"
        mode_color = (0, 255, 100) if auto_mode else (100, 200, 255)
        cv2.putText(display, f"Mód: {mode_text}  |  Mentett: {saved}/{max_frames}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, f"Mód: {mode_text}  |  Mentett: {saved}/{max_frames}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2, cv2.LINE_AA)
        cv2.putText(display, "SPACE=ment  a=auto  q=kilepes",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        if auto_mode:
            now = time.time()
            if (now - last_save_t) >= interval:
                _save_frame(frame, prefix, saved)
                saved += 1
                last_save_t = now
                if saved >= max_frames:
                    log.info("Elérte a maximális frame számot (%d). Befejezve.", max_frames)
                    break

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            log.info("Kilépés.")
            break
        elif key == ord(' '):
            _save_frame(frame, prefix, saved)
            saved += 1
            log.info("Kézi mentés: %d/%d", saved, max_frames)
            if saved >= max_frames:
                break
        elif key == ord('a'):
            auto_mode = not auto_mode
            last_save_t = time.time()
            log.info("Automata mód: %s", "BE" if auto_mode else "KI")

    cv2.destroyAllWindows()
    log.info("Összesen %d kép mentve ide: %s", saved, OUTPUT_DIR)


def _save_frame(frame: np.ndarray, prefix: str, idx: int) -> None:
    filename = OUTPUT_DIR / f"{prefix}_{idx:05d}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Képgyűjtő script XIMEA kamerából (labda detektálás tanításhoz)"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Kamera index (0=bal, 1=jobb). Default: 0")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="Auto mód: mentési időköz másodpercben (default: 0.3)")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Maximálisan gyűjtendő képek száma (default: 500)")
    parser.add_argument("--mock", action="store_true",
                        help="MockCamera használata valódi kamera helyett (teszteléshez)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LABDA DETEKTÁLÁS – KÉPGYŰJTŐ")
    log.info("Kimenet könyvtár: %s", OUTPUT_DIR)
    log.info("Maximális képek: %d", args.max_frames)
    log.info("Auto mentési időköz: %.2f mp", args.interval)
    log.info("=" * 60)

    if args.mock:
        collect_from_mock(args.interval, args.max_frames)
    else:
        collect_from_ximea(args.camera, args.interval, args.max_frames)


if __name__ == "__main__":
    main()
