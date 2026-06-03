#!/usr/bin/env python3
"""
rpi_headless.py  –  Raspberry Pi 5 + AI HAT oldal
══════════════════════════════════════════════════════
GUI nélküli "headless" futás a Raspberry Pi-n.

Feladatai:
  1. XIMEA kamerák olvasása (mindkét kamera)
  2. Labda detektálás (Hailo NPU vagy TensorRT/PyTorch fallback)
  3. Sztereó triangulálás → 3D koordináta
  4. Kalman-szűrő + pályabecslés
  5. UDP küldés → motor vezérlő (kis latencia, közvetlen)
  6. UDP küldés → PC monitor GUI (telemetria + tömörített preview kép)
  7. UDP fogadás → PC-től érkező konfig parancsok

Indítás:
    python3 src/rpi_headless.py
    python3 src/rpi_headless.py --config config/system_config.yaml
    python3 src/rpi_headless.py --profile indoor_field
"""

import argparse
import base64
import json
import logging
import math
import os
import queue
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# ── Projekt gyökér ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.network import UDPSender
from detection.ball_detector import BallDetector, DetectionResult
from detection.trajectory_predictor import BallTrajectoryPredictor
from stereo.triangulation import StereoTriangulator
from pc_tracker import load_config, _build_cameras

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "rpi_headless.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetria küldő (PC monitor felé)
# ─────────────────────────────────────────────────────────────────────────────

class TelemetrySender:
    """
    UDP alapú telemetria küldő a PC monitor GUI felé.
    Küld: koordináták, FPS, konfidencia, tömörített preview képek.
    """

    def __init__(self, pc_ip: str, pc_port: int, preview_quality: int = 40):
        self.pc_ip = pc_ip
        self.pc_port = pc_port
        self.preview_quality = preview_quality  # JPEG minőség (0-100)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Nagy küldési buffer (preview képekhez)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
        logger.info("TelemetrySender → %s:%d", pc_ip, pc_port)

    def send(
        self,
        tracked: bool,
        x: float, y: float, z: float,
        pred_x: float, pred_y: float, t_impact: float,
        pred_conf: float, speed_mms: float,
        cam_fps: float, det_fps: float,
        conf_l: float, conf_r: float,
        frame_l: Optional[np.ndarray] = None,
        frame_r: Optional[np.ndarray] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "tracked": tracked,
            "x": round(x, 1), "y": round(y, 1), "z": round(z, 1),
            "px": round(pred_x, 1), "py": round(pred_y, 1),
            "ti": round(t_impact, 3),
            "pc": round(pred_conf, 3),
            "spd": round(speed_mms, 1),
            "fps": round(cam_fps, 1),
            "dfps": round(det_fps, 1),
            "cl": round(conf_l, 3),
            "cr": round(conf_r, 3),
        }

        # Preview képek: kis méretű tömörített JPEG (nem teljes felbontás!)
        for key, frame in (("fl", frame_l), ("fr", frame_r)):
            if frame is not None:
                small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_NEAREST)
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.preview_quality])
                payload[key] = base64.b64encode(buf.tobytes()).decode("ascii")

        try:
            msg = json.dumps(payload).encode("utf-8")
            self._sock.sendto(msg, (self.pc_ip, self.pc_port))
        except Exception as exc:
            logger.debug("Telemetria küldési hiba: %s", exc)

    def close(self):
        self._sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Konfig parancs fogadó (PC-től)
# ─────────────────────────────────────────────────────────────────────────────

class CommandReceiver(threading.Thread):
    """
    UDP-n fogadja a PC monitor parancsait (exposure, conf, stb.).
    """

    def __init__(self, listen_ip: str, listen_port: int):
        super().__init__(daemon=True, name="CommandReceiver")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((listen_ip, listen_port))
        self._running = True
        self._pending: queue.Queue = queue.Queue()
        logger.info("CommandReceiver listening on %s:%d", listen_ip, listen_port)

    def run(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                cmd = json.loads(data.decode("utf-8"))
                self._pending.put(cmd)
            except socket.timeout:
                pass
            except Exception as exc:
                if self._running:
                    logger.warning("Parancs fogadási hiba: %s", exc)

    def get_pending(self) -> list:
        cmds = []
        while not self._pending.empty():
            try:
                cmds.append(self._pending.get_nowait())
            except queue.Empty:
                break
        return cmds

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Fő RPi tracker
# ─────────────────────────────────────────────────────────────────────────────

class RPiHeadlessTracker:
    """
    A RPi fő vezérlő osztálya.
    Elvégzi a kamera olvasást, detektálást, számítást, UDP küldést.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._running = True

        net_cfg    = config["network"]
        cam_cfg    = config["camera"]
        stereo_cfg = config["stereo"]
        det_cfg    = config["detection"]
        traj_cfg   = config.get("trajectory", {})
        rpi_cfg    = config.get("rpi", {})

        # ── Motor vezérlő UDP küldő ──────────────────────────────────────────
        self.motor_sender = UDPSender(
            ip=net_cfg["rpi_ip"],
            port=net_cfg["port"],
        )

        # ── PC monitor telemetria küldő ──────────────────────────────────────
        pc_ip   = rpi_cfg.get("pc_monitor_ip", "192.168.10.2")
        pc_port = rpi_cfg.get("pc_monitor_port", 5010)
        preview_quality = rpi_cfg.get("preview_jpeg_quality", 40)
        self.telemetry = TelemetrySender(pc_ip, pc_port, preview_quality)

        # ── PC parancs fogadó ────────────────────────────────────────────────
        cmd_port = rpi_cfg.get("cmd_listen_port", 5011)
        self.cmd_receiver = CommandReceiver("0.0.0.0", cmd_port)

        # ── Kamerák ──────────────────────────────────────────────────────────
        self.cam_left, self.cam_right = _build_cameras(cam_cfg, stereo_cfg)

        # ── YOLO detektor ────────────────────────────────────────────────────
        kalman_cfg = det_cfg.get("kalman", {})
        if not kalman_cfg.get("enabled", True):
            kalman_cfg = {"max_coast_frames": 0}

        self.detector = BallDetector(
            method=det_cfg.get("method", "yolo"),
            yolo_model_path=det_cfg.get("yolo_model_path", "models/custom_ball.engine"),
            yolo_class_filter=det_cfg.get("yolo_class_filter", 0),
            hsv_bounds=det_cfg.get("hsv_bounds"),
            hough_cfg=det_cfg.get("hough"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.35),
            kalman_cfg=kalman_cfg,
            roi_cfg=det_cfg.get("roi"),
        )

        # ── Triangulálás ─────────────────────────────────────────────────────
        self.triangulator = StereoTriangulator(
            baseline_mm=stereo_cfg["baseline_mm"],
            focal_length_px=stereo_cfg["focal_length_px"],
            cx=stereo_cfg.get("principal_point_x", cam_cfg["resolution"]["width"] / 2.0),
            cy=stereo_cfg.get("principal_point_y", cam_cfg["resolution"]["height"] / 2.0),
        )

        # ── Pályabecslő ──────────────────────────────────────────────────────
        goal_z  = float(traj_cfg.get("goal_z_mm", 600.0))
        gravity = float(traj_cfg.get("gravity_mm_s2", 9810.0))
        self._predictor = BallTrajectoryPredictor(
            gravity_mm_s2=gravity,
            goal_z_mm=goal_z,
        )
        self._pred_last_t: Optional[float] = None

        # ── Async detektálás queue ───────────────────────────────────────────
        self._det_queue: queue.Queue = queue.Queue(maxsize=2)
        self._det_lock  = threading.Lock()
        self._det_state: Dict = {
            "result_l": DetectionResult(), "result_r": DetectionResult(),
            "tracked": False, "x": 0.0, "y": 0.0, "z": 0.0,
            "det_fps": 0.0, "conf_l": 0.0, "conf_r": 0.0,
            "pred_x": 0.0, "pred_y": 0.0, "t_impact": 0.0,
            "pred_conf": 0.0, "speed_mms": 0.0,
        }

        # ── CLAHE (opcionális képfeldolgozás) ────────────────────────────────
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.use_clahe = cam_cfg.get("use_clahe", False)

        logger.info("RPiHeadlessTracker inicializálva.")

    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Elindítja a kamerákat és az összes szálat."""
        # Kamerák párhuzamos megnyitása
        with ThreadPoolExecutor(max_workers=2) as ex:
            ok_l = ex.submit(self.cam_left.open).result()
            ok_r = ex.submit(self.cam_right.open).result()

        if not ok_l or not ok_r:
            logger.error("Nem sikerült megnyitni a kamerákat!")
            return

        logger.info("Kamerák megnyitva. Detektálás indul...")

        # Async detektálási szál
        det_thread = threading.Thread(
            target=self._detection_worker, daemon=True, name="DetectionWorker"
        )
        det_thread.start()

        # PC parancs fogadó szál
        self.cmd_receiver.start()

        # Fő capture loop
        self._capture_loop()

        # Leállítás
        self.cmd_receiver.stop()
        self.cam_left.close()
        self.cam_right.close()
        self.motor_sender.close()
        self.telemetry.close()
        logger.info("RPiHeadlessTracker leállítva.")

    # ─────────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Fő kamera olvasó loop – minél gyorsabban olvas, nem vár GUI-ra."""
        cam_fps_ema = 0.0
        # Telemetria max 30x/mp a PC felé (nem kell több, a GUI-nak elég)
        _TELEM_INTERVAL = 1.0 / 30.0
        _last_telem = 0.0

        while self._running:
            t_start = time.perf_counter()

            # PC parancsok feldolgozása
            self._process_commands()

            # ── Párhuzamos kamera olvasás ──────────────────────────────────
            buf = [None, None, None, None]

            def _read_l():
                buf[0], buf[1] = self.cam_left.read()

            def _read_r():
                buf[2], buf[3] = self.cam_right.read()

            tl = threading.Thread(target=_read_l, daemon=True)
            tr = threading.Thread(target=_read_r, daemon=True)
            tl.start(); tr.start()
            tl.join();  tr.join()

            ret_l, frame_l, ret_r, frame_r = buf[0], buf[1], buf[2], buf[3]

            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                time.sleep(0.005)
                continue

            # CLAHE (opcionális)
            if self.use_clahe:
                frame_l = self._apply_clahe(frame_l)
                frame_r = self._apply_clahe(frame_r)

            # Detektáláshoz beküldés (nem blokkoló)
            if not self._det_queue.full():
                try:
                    ts = self.cam_left.get_timestamp()
                    self._det_queue.put_nowait((frame_l.copy(), frame_r.copy(), ts))
                except queue.Full:
                    pass

            # Legfrissebb detektálási eredmény
            with self._det_lock:
                det = dict(self._det_state)

            dt = max(time.perf_counter() - t_start, 1e-9)
            cam_fps_ema = 0.9 * cam_fps_ema + 0.1 * (1.0 / dt)

            # Telemetria küldés a PC-nek (rate-limited)
            _now = time.perf_counter()
            if _now - _last_telem >= _TELEM_INTERVAL:
                _last_telem = _now
                self.telemetry.send(
                    tracked=det["tracked"],
                    x=det["x"], y=det["y"], z=det["z"],
                    pred_x=det["pred_x"], pred_y=det["pred_y"],
                    t_impact=det["t_impact"],
                    pred_conf=det["pred_conf"],
                    speed_mms=det["speed_mms"],
                    cam_fps=cam_fps_ema,
                    det_fps=det["det_fps"],
                    conf_l=det["conf_l"],
                    conf_r=det["conf_r"],
                    frame_l=frame_l,
                    frame_r=frame_r,
                )

    # ─────────────────────────────────────────────────────────────────────────

    def _detection_worker(self):
        """Async GPU/NPU detektálási szál."""
        det_fps_ema = 0.0

        while self._running:
            try:
                frame_l, frame_r, timestamp = self._det_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # ── Detektálás ────────────────────────────────────────────────
            result_l, result_r = self.detector.detect_stereo(frame_l, frame_r)

            x_3d = y_3d = z_3d = 0.0
            tracking_success = False

            if result_l.success and result_r.success:
                tracking_success, x_3d, y_3d, z_3d = self.triangulator.triangulate(
                    (result_l.x, result_l.y),
                    (result_r.x, result_r.y),
                )

            # ── Pályabecslés ──────────────────────────────────────────────
            pred_x = pred_y = t_impact = 0.0
            pred_confidence = 0.0
            vx = vy = vz = 0.0
            curr_t = time.perf_counter()

            if tracking_success:
                dt_pred = (curr_t - self._pred_last_t) if self._pred_last_t else 0.033
                self._predictor.update(x_3d, y_3d, z_3d, dt_pred)
                self._pred_last_t = curr_t

                impact = self._predictor.get_impact_point()
                if impact:
                    pred_x, pred_y, t_impact = impact

                pred_confidence = self._predictor.confidence
                vx, vy, vz = self._predictor.get_velocity_mms()
            else:
                self._predictor.reset()
                self._pred_last_t = None

            # ── Motor vezérlő UDP (közvetlen, kis latencia) ───────────────
            self.motor_sender.send_target_position(
                x_3d, y_3d, z_3d, tracking_success, timestamp,
                pred_x, pred_y, t_impact,
            )

            dt = max(time.perf_counter() - t0, 1e-9)
            det_fps_ema = 0.9 * det_fps_ema + 0.1 * (1.0 / dt)

            with self._det_lock:
                self._det_state = {
                    "result_l": result_l, "result_r": result_r,
                    "tracked": tracking_success,
                    "x": x_3d, "y": y_3d, "z": z_3d,
                    "det_fps": det_fps_ema,
                    "conf_l": float(result_l.confidence) if result_l.success else 0.0,
                    "conf_r": float(result_r.confidence) if result_r.success else 0.0,
                    "pred_x": pred_x, "pred_y": pred_y,
                    "t_impact": t_impact,
                    "pred_conf": pred_confidence,
                    "speed_mms": math.sqrt(vx**2 + vy**2 + vz**2),
                }

    # ─────────────────────────────────────────────────────────────────────────

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _process_commands(self):
        """PC-től érkező parancsok feldolgozása."""
        for cmd in self.cmd_receiver.get_pending():
            action = cmd.get("cmd")
            value  = cmd.get("value")
            try:
                if action == "set_exposure":
                    self.cam_left.set_exposure(int(value))
                    self.cam_right.set_exposure(int(value))
                    logger.info("Exposure → %d µs", value)
                elif action == "set_gain":
                    self.cam_left.set_gain(float(value))
                    self.cam_right.set_gain(float(value))
                    logger.info("Gain → %.1f dB", value)
                elif action == "set_conf_threshold":
                    self.detector.confidence_threshold = float(value)
                    logger.info("Conf threshold → %.2f", value)
                elif action == "reset_kalman":
                    self.detector._kalman_left.reset()
                    self.detector._kalman_right.reset()
                    logger.info("Kalman filterek visszaállítva.")
                elif action == "set_clahe":
                    self.use_clahe = bool(value)
                    logger.info("CLAHE: %s", self.use_clahe)
                elif action == "stop":
                    logger.info("Stop parancs érkezett PC-ről.")
                    self._running = False
            except Exception as exc:
                logger.warning("Parancs feldolgozási hiba (%s): %s", action, exc)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Belépési pont
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RPi headless tracker")
    parser.add_argument("--config", default="config/system_config.yaml")
    parser.add_argument("--profile", default=None,
                        help="Profil neve (pl. 'indoor_field')")
    args = parser.parse_args()

    # Log mappa
    (ROOT / "logs").mkdir(exist_ok=True)

    config = load_config(args.config, profile=args.profile)

    logger.info("=" * 60)
    logger.info("RPi Headless Tracker indul")
    logger.info("Config: %s | Profil: %s", args.config, args.profile or "alapértelmezett")
    logger.info("=" * 60)

    tracker = RPiHeadlessTracker(config)

    try:
        tracker.start()
    except KeyboardInterrupt:
        logger.info("Ctrl+C – leállítás...")
        tracker.stop()


if __name__ == "__main__":
    main()
