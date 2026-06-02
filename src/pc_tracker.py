"""
Real-Time 3D Ball Tracker – PC-side entry point.

Reads stereo camera frames, detects the ball in both views, triangulates its
3D position, streams coordinates to the Raspberry Pi controller via UDP, and
shows a live composite display with optional HSV-calibration sliders.

Usage:
    python3 src/pc_tracker.py
    python3 src/pc_tracker.py --calibrate          # interactive HSV tuning
    python3 src/pc_tracker.py --config custom.yaml # alternative config path
"""

import logging
import argparse
import time
from typing import Any, Dict, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from common.network import UDPSender
from detection.ball_detector import BallDetector, DetectionResult, draw_detection
from detection.trajectory_predictor import BallTrajectoryPredictor
from detection.camera import MockCamera, MindVisionCamera, XimeaCamera
from stereo.triangulation import StereoTriangulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "network": {"rpi_ip": "127.0.0.1", "port": 5005},
    "camera": {
        "type": "mock",
        "resolution": {"width": 1280, "height": 720},
        "fps": 60,
        "exposure_time_us": 800,
        "gain": 0.0,
    },
    "stereo": {
        "baseline_mm": 300.0,
        "focal_length_px": 1200.0,
        "principal_point_x": 640.0,
        "principal_point_y": 360.0,
        "left_camera_index": 0,
        "right_camera_index": 1,
    },
    "detection": {
        "method": "hybrid",
        "yolo_model_path": "yolov8l.pt",    # ← Large modell
        "confidence_threshold": 0.35,        # ← Kicsit alacsonyabb küszöb
        "hsv_bounds": {
            "lower_h": 0,   "lower_s": 0,   "lower_v": 160,   # ← V csökkentve
            "upper_h": 180, "upper_s": 60,  "upper_v": 255,
        },
        "hough": {
            "min_dist": 25, "param1": 80, "param2": 25,       # ← Érzékenyebb
            "min_radius": 8, "max_radius": 150,
        },
        "kalman": {
            "enabled": True,
            "process_noise": 0.01,
            "measurement_noise": 0.1,
            "max_coast_frames": 12,            # ← Kicsit több coasting
        },
        "roi": {"enabled": True, "padding_factor": 3.0},       # ← Nagyobb ROI
    },
}

# HUD colour palette (BGR)
_COL_GREEN  = (0,   255,   0)
_COL_YELLOW = (0,   220, 255)
_COL_ORANGE = (0,   165, 255)
_COL_RED    = (0,     0, 255)
_COL_CYAN   = (255, 255,   0)
_COL_WHITE  = (255, 255, 255)

# Method → circle colour mapping for display
_METHOD_COLOR: Dict[str, Tuple[int, int, int]] = {
    "yolo":   _COL_GREEN,
    "hsv":    _COL_CYAN,
    "kalman": _COL_YELLOW,
    "none":   _COL_RED,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration, returning built-in defaults on any failure."""
    try:
        import yaml  # type: ignore[import]
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)
        logger.info("Loaded configuration from '%s'.", config_path)
        return config
    except ImportError:
        logger.warning("PyYAML not installed – using built-in defaults.")
    except Exception as exc:
        logger.warning("Cannot load config '%s' (%s) – using built-in defaults.", config_path, exc)
    return _DEFAULT_CONFIG


def save_hsv_config(
    config_path: str,
    config: Dict[str, Any],
    bounds: Dict[str, int],
) -> None:
    """Persist calibrated HSV bounds back into the YAML config file."""
    try:
        import yaml  # type: ignore[import]
        config["detection"]["hsv_bounds"] = bounds
        with open(config_path, "w") as fh:
            yaml.safe_dump(config, fh, default_flow_style=False)
        logger.info("HSV calibration saved to '%s'.", config_path)
    except Exception as exc:
        logger.error("Failed to save HSV calibration: %s", exc)


# ---------------------------------------------------------------------------
# Camera factory
# ---------------------------------------------------------------------------

def _build_cameras(cam_cfg: Dict[str, Any], stereo_cfg: Dict[str, Any]):
    """Instantiate left and right cameras according to the configured type."""
    cam_type: str = cam_cfg.get("type", "mock").lower()
    width: int = cam_cfg["resolution"]["width"]
    height: int = cam_cfg["resolution"]["height"]
    fps: int = cam_cfg["fps"]
    left_idx: int = stereo_cfg["left_camera_index"]
    right_idx: int = stereo_cfg["right_camera_index"]

    if cam_type == "ximea":
        logger.info("Initializing XIMEA cameras (indices %d, %d).", left_idx, right_idx)
        exposure: int = cam_cfg.get("exposure_time_us", 800)
        gain: float = cam_cfg.get("gain", 0.0)
        offset_x = cam_cfg["resolution"].get("offset_x")
        offset_y = cam_cfg["resolution"].get("offset_y")
        bw_limit = cam_cfg.get("bandwidth_limit_mbs", 160)
        cam_left = XimeaCamera(left_idx, width, height, exposure, gain, offset_x, offset_y, bw_limit)
        cam_right = XimeaCamera(right_idx, width, height, exposure, gain, offset_x, offset_y, bw_limit)

    elif cam_type == "mindvision":
        logger.info("Initializing MindVision cameras (indices %d, %d).", left_idx, right_idx)
        cam_left = MindVisionCamera(left_idx, width, height)
        cam_right = MindVisionCamera(right_idx, width, height)

    elif cam_type == "opencv":
        logger.info("Initializing OpenCV/V4L2 cameras (indices %d, %d).", left_idx, right_idx)
        cam_left = MindVisionCamera(left_idx, width, height)
        cam_right = MindVisionCamera(right_idx, width, height)
        # Force OpenCV fallback by hiding the SDK reference
        cam_left._mvsdk = None
        cam_right._mvsdk = None

    else:
        logger.info("Initializing mock cameras (synthetic 3D simulation).")
        cam_left  = MockCamera(width, height, fps, is_left=True)
        cam_right = MockCamera(width, height, fps, is_left=False)

    return cam_left, cam_right


# ---------------------------------------------------------------------------
# HSV calibration UI helpers
# ---------------------------------------------------------------------------

def _trackbar_noop(value: int) -> None:
    """Required no-op callback for OpenCV createTrackbar."""


def _setup_calibration_window(hsv_bounds: Dict[str, int]) -> None:
    """Create the HSV calibration window with six sliders."""
    win = "HSV Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 400, 350)
    cv2.createTrackbar("Lower H", win, hsv_bounds.get("lower_h", 0),   179, _trackbar_noop)
    cv2.createTrackbar("Upper H", win, hsv_bounds.get("upper_h", 180), 179, _trackbar_noop)
    cv2.createTrackbar("Lower S", win, hsv_bounds.get("lower_s", 0),   255, _trackbar_noop)
    cv2.createTrackbar("Upper S", win, hsv_bounds.get("upper_s", 50),  255, _trackbar_noop)
    cv2.createTrackbar("Lower V", win, hsv_bounds.get("lower_v", 200), 255, _trackbar_noop)
    cv2.createTrackbar("Upper V", win, hsv_bounds.get("upper_v", 255), 255, _trackbar_noop)


def _read_hsv_trackbars() -> Tuple[int, int, int, int, int, int]:
    """Return current trackbar values as (lh, uh, ls, us, lv, uv)."""
    win = "HSV Calibration"
    return (
        cv2.getTrackbarPos("Lower H", win),
        cv2.getTrackbarPos("Upper H", win),
        cv2.getTrackbarPos("Lower S", win),
        cv2.getTrackbarPos("Upper S", win),
        cv2.getTrackbarPos("Lower V", win),
        cv2.getTrackbarPos("Upper V", win),
    )


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time 3D Ball Tracker (PC side)")
    parser.add_argument("--config", default="config/system_config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enable interactive HSV calibration sliders")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    config     = load_config(args.config)
    net_cfg    = config["network"]
    cam_cfg    = config["camera"]
    stereo_cfg = config["stereo"]
    det_cfg    = config["detection"]

    is_hsv_calibrate: bool = args.calibrate

    # ── Build subsystems ─────────────────────────────────────────────────────
    sender = UDPSender(ip=net_cfg["rpi_ip"], port=net_cfg["port"])

    cam_left, cam_right = _build_cameras(cam_cfg, stereo_cfg)

    # Build Kalman kwargs only if enabled
    kalman_cfg = det_cfg.get("kalman", {})
    if not kalman_cfg.get("enabled", True):
        # Pass empty dict so BallDetector uses defaults (Kalman always on internally,
        # but with very large coast=0 effectively disables coasting).
        kalman_cfg = {"max_coast_frames": 0}

    detector = BallDetector(
        method=det_cfg.get("method", "hybrid"),
        yolo_model_path=det_cfg.get("yolo_model_path", "yolov8n.pt"),
        yolo_class_filter=det_cfg.get("yolo_class_filter", 32),
        hsv_bounds=det_cfg.get("hsv_bounds"),
        hough_cfg=det_cfg.get("hough"),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.4),
        kalman_cfg=kalman_cfg,
        roi_cfg=det_cfg.get("roi"),
    )

    triangulator = StereoTriangulator(
        baseline_mm=stereo_cfg["baseline_mm"],
        focal_length_px=stereo_cfg["focal_length_px"],
        cx=stereo_cfg.get("principal_point_x", cam_cfg["resolution"]["width"] / 2.0),
        cy=stereo_cfg.get("principal_point_y", cam_cfg["resolution"]["height"] / 2.0),
    )

    # ── Init Trajectory Predictor ────────────────────────────────────────────
    traj_cfg = config.get("trajectory", {})
    traj_enabled = traj_cfg.get("enabled", True)
    gravity = float(traj_cfg.get("gravity_mm_s2", 9810.0))
    goal_z = float(traj_cfg.get("goal_z_mm", 1500.0))
    predictor = BallTrajectoryPredictor(gravity_mm_s2=gravity, goal_z_mm=goal_z)
    last_t = None

    # ── Open cameras ─────────────────────────────────────────────────────────
    if not cam_left.open() or not cam_right.open():
        logger.error("Failed to open one or both cameras. Exiting.")
        sender.close()
        return

    logger.info("Both cameras open. Starting tracking loop. Press 'q' to quit.")

    height     = cam_cfg["resolution"]["height"]
    window_name = "Real-Time 3D Ball Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if is_hsv_calibrate:
        logger.info("HSV Calibration mode – adjust sliders, press 's' to save, 'q' to quit.")
        _setup_calibration_window(det_cfg.get("hsv_bounds", {}))

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while True:
            t_loop_start = time.perf_counter()

            # 1. Sync calibration sliders with detector
            if is_hsv_calibrate:
                lh, uh, ls, us, lv, uv = _read_hsv_trackbars()
                detector.lower_hsv = np.array([lh, ls, lv], dtype=np.uint8)
                detector.upper_hsv = np.array([uh, us, uv], dtype=np.uint8)

            # 2. Capture stereo frames
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                logger.warning("Incomplete stereo pair – skipping frame.")
                continue

            timestamp = cam_left.get_timestamp()

            # 3. Detect ball in both views
            result_l, result_r = detector.detect_stereo(frame_l, frame_r)

            x_3d = y_3d = z_3d = 0.0
            tracking_success = False

            # 4. Triangulate when a valid position is available in both cameras.
            #    We triangulate even on Kalman-predicted frames so the robot keeps
            #    responding smoothly – the RPi receives a lower-confidence coordinate.
            if result_l.success and result_r.success:
                tracking_success, x_3d, y_3d, z_3d = triangulator.triangulate(
                    (result_l.x, result_l.y),
                    (result_r.x, result_r.y),
                )
                if tracking_success:
                    _draw_tracking_markers(frame_l, result_l)
                    _draw_tracking_markers(frame_r, result_r)

            # 5. Predict Trajectory & Stream to Raspberry Pi
            pred_x, pred_y, t_impact = 0.0, 0.0, 0.0
            if tracking_success and traj_enabled:
                curr_t = time.perf_counter()
                if last_t is not None:
                    dt = curr_t - last_t
                    predictor.update(x_3d, y_3d, z_3d, dt)
                    pred = predictor.predict_intersection()
                    if pred is not None:
                        pred_x, pred_y, t_impact = pred
                        
                        # Draw the predicted trajectory curve on both frames
                        state = predictor.get_current_state()
                        path_pts_l, path_pts_r = [], []
                        # 10 segments along the curve
                        for step_t in np.linspace(0, t_impact, num=10):
                            sim_x = state[0] + state[3] * step_t
                            sim_y = state[1] + state[4] * step_t - 0.5 * predictor.gravity * (step_t**2)
                            sim_z = state[2] + state[5] * step_t
                            pt_l, pt_r = triangulator.project_to_2d(sim_x, sim_y, sim_z)
                            path_pts_l.append(pt_l)
                            path_pts_r.append(pt_r)
                            
                        # Draw lines
                        for i in range(1, len(path_pts_l)):
                            cv2.line(frame_l, path_pts_l[i-1], path_pts_l[i], (255, 0, 255), 2)
                            cv2.line(frame_r, path_pts_r[i-1], path_pts_r[i], (255, 0, 255), 2)
                        
                        # Draw target marker
                        if len(path_pts_l) > 0:
                            cv2.drawMarker(frame_l, path_pts_l[-1], (255, 0, 255), cv2.MARKER_CROSS, 20, 3)
                            cv2.drawMarker(frame_r, path_pts_r[-1], (255, 0, 255), cv2.MARKER_CROSS, 20, 3)
                last_t = curr_t
            else:
                last_t = None
                predictor.reset()

            sender.send_target_position(x_3d, y_3d, z_3d, tracking_success, timestamp, pred_x, pred_y, t_impact)

            # 6. Build display frame and overlay HUD
            display_frame = _build_display_frame(
                frame_l, frame_r, detector, height, is_hsv_calibrate
            )
            hud_y = (height * 2 - 30) if is_hsv_calibrate else (height - 30)
            pos_y = (height * 2 - 60) if is_hsv_calibrate else (height - 60)
            det_y = (height * 2 - 90) if is_hsv_calibrate else (height - 90)

            fps_actual = 1.0 / max(time.perf_counter() - t_loop_start, 1e-9)
            if not hasattr(main, 'loop_cnt'):
                main.loop_cnt = 0
                main.t_fps_start = time.perf_counter()
            main.loop_cnt += 1
            if main.loop_cnt % 30 == 0:
                avg_fps = main.loop_cnt / (time.perf_counter() - main.t_fps_start)
                logger.info("Average FPS (last 30 frames): %.1f", avg_fps)
            _draw_hud(
                display_frame, fps_actual, tracking_success,
                x_3d, y_3d, z_3d,
                result_l, result_r,
                hud_y, pos_y, det_y,
                is_hsv_calibrate,
            )

            cv2.imshow(window_name, display_frame)

            # 7. Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and is_hsv_calibrate:
                save_hsv_config(
                    args.config, config,
                    {"lower_h": lh, "upper_h": uh,
                     "lower_s": ls, "upper_s": us,
                     "lower_v": lv, "upper_v": uv},
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam_left.close()
        cam_right.close()
        sender.close()
        cv2.destroyAllWindows()
        logger.info("Tracker shutdown complete.")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _draw_tracking_markers(frame: np.ndarray, result: DetectionResult) -> None:
    """
    A labda teljes területét kiszínezi + körvonalat + crosshairt rajzol.
    Az eredeti egyszerű cv2.circle hívások helyett az új draw_detection()-t
    használja, ami kontúr alapú kitöltést is végez.
    """
    draw_detection(frame, result, alpha=0.35)


def _build_display_frame(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    detector: BallDetector,
    height: int,
    calibrate: bool,
) -> np.ndarray:
    raw = np.hstack((frame_l, frame_r))
    if not calibrate:
        return raw

    hsv_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2HSV)
    hsv_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2HSV)
    mask_l = cv2.cvtColor(
        cv2.inRange(hsv_l, detector.lower_hsv, detector.upper_hsv),
        cv2.COLOR_GRAY2BGR,
    )
    mask_r = cv2.cvtColor(
        cv2.inRange(hsv_r, detector.lower_hsv, detector.upper_hsv),
        cv2.COLOR_GRAY2BGR,
    )
    return np.vstack((raw, np.hstack((mask_l, mask_r))))


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    tracked: bool,
    x: float, y: float, z: float,
    result_l: DetectionResult,
    result_r: DetectionResult,
    hud_y: int,
    pos_y: int,
    det_y: int,
    calibrate: bool,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS + tracking status
    track_color = _COL_GREEN if tracked else _COL_RED
    cv2.putText(frame, f"FPS: {fps:.1f} | Tracked: {tracked}",
                (20, hud_y), font, 0.7, _COL_CYAN, 2, cv2.LINE_AA)

    if calibrate:
        cv2.putText(frame, "CALIBRATION MODE  's' = save  'q' = quit",
                    (20, 30), font, 0.7, _COL_ORANGE, 2, cv2.LINE_AA)

    # 3D position
    if tracked:
        cv2.putText(frame, f"3D Pos: [{x:.1f}, {y:.1f}, {z:.1f}] mm",
                    (20, pos_y), font, 0.7, _COL_GREEN, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "BALL OUT OF SIGHT",
                    (20, pos_y), font, 0.7, _COL_RED, 2, cv2.LINE_AA)

    # Detection method / confidence breakdown
    def _method_label(r: DetectionResult, side: str) -> str:
        if not r.success:
            return f"{side}: --"
        tag = f"[~]" if r.is_predicted else ""
        return f"{side}: {r.method.upper()}{tag} {r.confidence:.2f}"

    label_l = _method_label(result_l, "L")
    label_r = _method_label(result_r, "R")
    col_l   = _METHOD_COLOR.get(result_l.method, _COL_WHITE)
    col_r   = _METHOD_COLOR.get(result_r.method, _COL_WHITE)

    cv2.putText(frame, label_l, (20,  det_y), font, 0.55, col_l, 1, cv2.LINE_AA)
    cv2.putText(frame, label_r, (320, det_y), font, 0.55, col_r, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
