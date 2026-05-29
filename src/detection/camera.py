"""
Camera interface module for the Real-Time 3D Ball Tracker.

Provides a common abstract interface (CameraInterface) and concrete implementations
for all supported camera backends: XIMEA industrial USB3, MindVision industrial,
and MockCamera for simulation/testing.
"""

import time
import math
import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ROI alignment granularity required by XIMEA and most industrial sensors
_ROI_ALIGN_X: int = 8
_ROI_ALIGN_Y: int = 2


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CameraInterface(ABC):
    """Uniform interface for all camera backends (XIMEA, MindVision, Mock)."""

    @abstractmethod
    def open(self) -> bool:
        """Open the camera stream. Returns True on success."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture one BGR frame. Returns (success, frame)."""

    @abstractmethod
    def close(self) -> None:
        """Stop acquisition and release all hardware resources."""

    @abstractmethod
    def get_timestamp(self) -> float:
        """Return the Unix timestamp of the last captured frame."""

    @abstractmethod
    def set_exposure(self, us: int) -> None:
        """Dynamically set the exposure time in microseconds."""

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        """Dynamically set the analog gain in dB."""

    @abstractmethod
    def get_temperature(self) -> float:
        """Return the hardware temperature in Celsius. Returns 0.0 if unsupported."""

    @abstractmethod
    def set_roi(self, width: int, height: int, offset_x: int, offset_y: int) -> bool:
        """Dynamically set the hardware Region of Interest (ROI)."""


# ---------------------------------------------------------------------------
# XIMEA industrial camera
# ---------------------------------------------------------------------------

class XimeaCamera(CameraInterface):
    """
    Drives XIMEA industrial USB3 cameras via the official Python SDK (ximea.xiapi).

    The sensor ROI is centered automatically and aligned to hardware granularity.
    All parameters are read from the system configuration so the camera can be
    tuned without touching source code.

    Args:
        camera_index:    XIMEA device index (0 = first detected camera).
        width:           Desired capture width in pixels.
        height:          Desired capture height in pixels.
        exposure_time_us: Exposure duration in microseconds.
                         Bright outdoor light → 500–1000 µs.
                         Indoor / artificial light → 5 000–20 000 µs.
        gain:            Analog gain in dB (0–24).
                         Keep at 0 in bright conditions to minimize noise.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        exposure_time_us: int = 800,
        gain: float = 0.0,
        offset_x: Optional[int] = None,
        offset_y: Optional[int] = None,
        bandwidth_limit_mbs: int = 160,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.exposure_time_us = exposure_time_us
        self.gain = gain
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.bandwidth_limit_mbs = bandwidth_limit_mbs
        self._cam = None          # ximea.xiapi.Camera instance
        self._img = None          # ximea.xiapi.Image instance (reused every frame)
        self._last_timestamp: float = 0.0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_ready_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._pending_exposure: Optional[int] = None
        self._pending_gain: Optional[float] = None
        self._last_temperature: float = 0.0
        self._temp_counter: int = 0

    # ------------------------------------------------------------------
    # CameraInterface implementation
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Initialize hardware, configure ROI / exposure, start acquisition."""
        try:
            from ximea import xiapi  # type: ignore[import]
            self._cam = xiapi.Camera(dev_id=self.camera_index)
            self._cam.open_device()

            self._cam.set_imgdataformat("XI_RGB24")
            
            # Enable Auto White Balance for proper color representation
            try:
                self._cam.enable_auto_wb()
                logger.info("XimeaCamera[%d] Auto White Balance enabled.", self.camera_index)
            except Exception:
                try:
                    self._cam.set_param("auto_wb", 1)
                    logger.info("XimeaCamera[%d] auto_wb set to 1.", self.camera_index)
                except Exception as wb_exc:
                    logger.warning("XimeaCamera[%d] could not enable auto_wb: %s", self.camera_index, wb_exc)

            self._configure_roi()
            self._configure_exposure()

            # ── Limit per-camera bandwidth to leave headroom for the second camera
            # and for USB control commands (set_exposure, set_gain, etc.).
            # With two cameras each limited to 160 MB/s the total is 320 MB/s which
            # is safely below the ~400 MB/s USB3 bus capacity.  Without this limit
            # each camera grabs the full measured bandwidth (≈317 MB/s) causing
            # bus saturation that makes get_image() time out unpredictably.
            _bw_limit_mbs = self.bandwidth_limit_mbs
            try:
                self._cam.set_limit_bandwidth(_bw_limit_mbs)
                logger.info("XimeaCamera[%d] bandwidth limited to %d MB/s",
                            self.camera_index, _bw_limit_mbs)
            except Exception:
                try:
                    self._cam.set_param("limit_bandwidth", float(_bw_limit_mbs))
                    logger.info("XimeaCamera[%d] bandwidth limited via set_param to %d MB/s",
                                self.camera_index, _bw_limit_mbs)
                except Exception as bw_exc:
                    logger.warning("XimeaCamera[%d] could not set bandwidth limit: %s",
                                   self.camera_index, bw_exc)

            self._cam.start_acquisition()
            self._img = xiapi.Image()

            self._running = True
            self._latest_frame = None
            self._ret = False
            self._frame_ready_event.clear()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            logger.info(
                "XimeaCamera[%d] opened (async thread started) – %dx%d, exp=%d µs, gain=%.1f dB",
                self.camera_index, self.width, self.height,
                self.exposure_time_us, self.gain,
            )
            return True

        except Exception as exc:
            logger.error("XimeaCamera[%d] failed to open: %s", self.camera_index, exc)
            return False

    def _capture_loop(self) -> None:
        _consecutive_timeouts = 0
        _MAX_TIMEOUTS_BEFORE_RECOVERY = 5  # restart acquisition after N consecutive timeouts

        while self._running:
            try:
                if self._cam is None or self._img is None:
                    time.sleep(0.005)
                    continue

                if self._pending_exposure is not None:
                    try:
                        self._cam.set_exposure(self._pending_exposure)
                    except Exception as exc:
                        logger.warning("XimeaCamera[%d] background set_exposure error: %s",
                                       self.camera_index, exc)
                    self._pending_exposure = None

                if self._pending_gain is not None:
                    try:
                        self._cam.set_gain(self._pending_gain)
                    except Exception as exc:
                        logger.warning("XimeaCamera[%d] background set_gain error: %s",
                                       self.camera_index, exc)
                    self._pending_gain = None

                # timeout=200 ms: allows the loop to check _running frequently
                # and exit within ~200 ms when stop is requested.
                self._cam.get_image(self._img, timeout=200)
                self._last_timestamp = time.time()

                # Successful frame – reset the consecutive-timeout counter
                _consecutive_timeouts = 0

                raw_img = self._img.get_image_data_numpy()
                # XI_RGB24 natively returns a BGR array for OpenCV compatibility in Python
                bgr = raw_img.copy() if hasattr(raw_img, 'copy') else raw_img

                # Software resize only if sensor ROI could not be set exactly
                if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
                    bgr = cv2.resize(bgr, (self.width, self.height))

                with self._lock:
                    self._latest_frame = bgr
                    self._ret = True
                self._frame_ready_event.set()

            except Exception as exc:
                err_str = str(exc)
                is_timeout = ("10" in err_str) or ("Timeout" in err_str) or ("timeout" in err_str)

                if is_timeout:
                    _consecutive_timeouts += 1
                    logger.debug("XimeaCamera[%d] get_image timeout #%d",
                                 self.camera_index, _consecutive_timeouts)

                    if _consecutive_timeouts >= _MAX_TIMEOUTS_BEFORE_RECOVERY:
                        # Camera stopped sending frames – try to restart acquisition
                        logger.warning(
                            "XimeaCamera[%d] %d consecutive timeouts – restarting acquisition…",
                            self.camera_index, _consecutive_timeouts,
                        )
                        try:
                            self._cam.stop_acquisition()
                        except Exception:
                            pass
                        time.sleep(0.05)
                        try:
                            self._cam.start_acquisition()
                            _consecutive_timeouts = 0
                            logger.info("XimeaCamera[%d] acquisition restarted successfully.",
                                        self.camera_index)
                        except Exception as restart_exc:
                            logger.error("XimeaCamera[%d] restart failed: %s",
                                         self.camera_index, restart_exc)
                            break  # Unrecoverable; exit thread
                else:
                    logger.error("XimeaCamera[%d] capture thread error: %s",
                                 self.camera_index, exc)
                    time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Acquire one frame and return it as a BGR numpy array.

        Returns a .copy() of the internal frame buffer so the caller is free
        to modify it (e.g. draw_overlay) without data races against the capture
        thread that may simultaneously write a new frame into _latest_frame.
        """
        if not self._running:
            return False, None

        # Wait for a new frame from background thread with 100 ms timeout
        is_new = self._frame_ready_event.wait(timeout=0.1)
        if is_new:
            self._frame_ready_event.clear()

        with self._lock:
            frame = self._latest_frame
            ret = self._ret

        if frame is None:
            return False, None
        return ret, frame.copy()

    def close(self) -> None:
        """Stop acquisition and release the device handle."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cam is None:
            return
        try:
            self._cam.stop_acquisition()
            self._cam.close_device()
            logger.info("XimeaCamera[%d] closed.", self.camera_index)
        except Exception as exc:
            logger.error("XimeaCamera[%d] close error: %s", self.camera_index, exc)
        finally:
            self._cam = None
            self._img = None

    def get_timestamp(self) -> float:
        return self._last_timestamp

    def _stop_thread_only(self) -> None:
        """
        Stop the background capture thread without touching acquisition.
        Used for coordinated multi-camera ROI changes: stop all threads first,
        then reconfigure all cameras, then restart all threads.
        This prevents USB bus contention between cameras during ROI changes.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)  # 200 ms acqtimeout + processing margin
            self._thread = None

    def _start_thread_only(self) -> None:
        """Restart the background capture thread after a coordinated stop."""
        self._running = True
        self._latest_frame = None
        self._ret = False
        self._frame_ready_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def set_exposure(self, us: int) -> None:
        self.exposure_time_us = us
        self._pending_exposure = us

    def set_gain(self, gain: float) -> None:
        self.gain = gain
        self._pending_gain = gain

    def get_temperature(self) -> float:
        return self._last_temperature

    def set_roi(self, width: int, height: int, offset_x: int, offset_y: int) -> bool:
        """
        Change the hardware ROI safely by stopping the capture thread first.
        For multi-camera coordinated changes use _stop_thread_only / _start_thread_only
        so both camera threads are stopped before either acquisition is modified.
        """
        if self._cam is None:
            return False
        self._stop_thread_only()
        ok = False
        try:
            try:
                self._cam.stop_acquisition()
            except Exception:
                pass
            self.width   = width
            self.height  = height
            self.offset_x = offset_x
            self.offset_y = offset_y
            self._configure_roi()
            self._cam.start_acquisition()
            logger.info("XimeaCamera[%d] ROI set: %dx%d @ (%d,%d)",
                        self.camera_index, width, height, offset_x, offset_y)
            ok = True
        except Exception as exc:
            logger.error("XimeaCamera[%d] set_roi error: %s", self.camera_index, exc)
        finally:
            self._start_thread_only()
        return ok

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_roi(self) -> None:
        """Set the sensor ROI to the requested size, centered on the sensor."""
        try:
            self._cam.set_width(self.width)
            self._cam.set_height(self.height)

            sensor_w: int = self._cam.get_width_maximum()
            sensor_h: int = self._cam.get_height_maximum()

            if self.offset_x is not None and self.offset_y is not None:
                offset_x = (self.offset_x // _ROI_ALIGN_X) * _ROI_ALIGN_X
                offset_y = (self.offset_y // _ROI_ALIGN_Y) * _ROI_ALIGN_Y
            else:
                # Center the ROI and round down to hardware alignment boundary
                offset_x = ((sensor_w - self.width) // 2 // _ROI_ALIGN_X) * _ROI_ALIGN_X
                offset_y = ((sensor_h - self.height) // 2 // _ROI_ALIGN_Y) * _ROI_ALIGN_Y

            self._cam.set_offsetX(offset_x)
            self._cam.set_offsetY(offset_y)

            logger.debug(
                "XimeaCamera[%d] ROI: %dx%d at offset (%d, %d)",
                self.camera_index, self.width, self.height, offset_x, offset_y,
            )
        except Exception as exc:
            logger.warning(
                "XimeaCamera[%d] could not configure sensor ROI: %s", self.camera_index, exc
            )

    def _configure_exposure(self) -> None:
        """Apply exposure time and gain to the hardware registers."""
        try:
            self._cam.set_exposure(self.exposure_time_us)
            self._cam.set_gain(self.gain)
        except Exception as exc:
            logger.warning(
                "XimeaCamera[%d] could not set exposure/gain: %s", self.camera_index, exc
            )


# ---------------------------------------------------------------------------
# MindVision industrial camera (legacy / future use)
# ---------------------------------------------------------------------------

class MindVisionCamera(CameraInterface):
    """
    Drives MindVision MC023CG-SY-UB cameras via the mvsdk Python SDK.
    Falls back to a plain OpenCV VideoCapture when the SDK is not installed.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._h_camera = None          # mvsdk camera handle
        self._p_frame_buffer = None    # mvsdk DMA buffer pointer
        self._cap: Optional[cv2.VideoCapture] = None  # OpenCV fallback capture
        self._last_timestamp: float = 0.0
        self._mvsdk = None             # SDK module reference (None if not installed)

        try:
            import mvsdk  # type: ignore[import]
            self._mvsdk = mvsdk
            logger.info("MindVision SDK loaded.")
        except ImportError:
            logger.warning("MindVision SDK not found – OpenCV fallback will be used.")

    # ------------------------------------------------------------------
    # CameraInterface implementation
    # ------------------------------------------------------------------

    def open(self) -> bool:
        if self._mvsdk is None:
            return self._open_opencv_fallback()

        try:
            devices = self._mvsdk.CameraEnumerateDevice()
            if not devices:
                logger.error("No MindVision cameras detected.")
                return False
            if self.camera_index >= len(devices):
                logger.error(
                    "Camera index %d out of range (%d device(s) found).",
                    self.camera_index, len(devices),
                )
                return False

            self._h_camera = self._mvsdk.CameraInit(devices[self.camera_index], -1, -1)
            name = self._mvsdk.CameraGetFriendlyName(self._h_camera)
            logger.info("MindVisionCamera[%d] connected: %s", self.camera_index, name)

            self._mvsdk.CameraSetTriggerMode(self._h_camera, 0)  # Continuous mode
            self._mvsdk.CameraPlay(self._h_camera)

            cap_info = self._mvsdk.CameraGetCapability(self._h_camera)
            buf_size = (
                cap_info.sResolutionRange.iWidthMax
                * cap_info.sResolutionRange.iHeightMax
                * 3
            )
            self._p_frame_buffer = self._mvsdk.CameraAlignMalloc(buf_size, 16)
            return True

        except Exception as exc:
            logger.error("MindVisionCamera[%d] init error: %s", self.camera_index, exc)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self._last_timestamp = time.time()

        if self._mvsdk is None:
            if self._cap is None:
                return False, None
            ret, frame = self._cap.read()
            return ret, frame

        try:
            raw, head = self._mvsdk.CameraGetImageBuffer(self._h_camera, 500)
            self._mvsdk.CameraImageProcess(self._h_camera, raw, self._p_frame_buffer, head)
            self._mvsdk.CameraReleaseImageBuffer(self._h_camera, raw)

            buf = (self._mvsdk.c_char * head.uBytes).from_address(self._p_frame_buffer)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((head.iHeight, head.iWidth, 3))
            return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        except Exception as exc:
            logger.error("MindVisionCamera[%d] read error: %s", self.camera_index, exc)
            return False, None

    def close(self) -> None:
        if self._mvsdk is None:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return

        if self._h_camera is not None:
            self._mvsdk.CameraUnInit(self._h_camera)
            self._mvsdk.CameraAlignFree(self._p_frame_buffer)
            self._h_camera = None
            self._p_frame_buffer = None
            logger.info("MindVisionCamera[%d] closed.", self.camera_index)

    def get_timestamp(self) -> float:
        return self._last_timestamp

    def set_exposure(self, us: int) -> None:
        if self._mvsdk is not None and self._h_camera is not None:
            try:
                self._mvsdk.CameraSetAeState(self._h_camera, 0)
                self._mvsdk.CameraSetExposureTime(self._h_camera, float(us))
            except Exception as exc:
                logger.warning("MindVisionCamera[%d] set_exposure error: %s", self.camera_index, exc)

    def set_gain(self, gain: float) -> None:
        if self._mvsdk is not None and self._h_camera is not None:
            try:
                # Approximate translation to MindVision analog gain integer
                int_gain = int(max(0, gain) * 10)
                self._mvsdk.CameraSetAnalogGain(self._h_camera, int_gain)
            except Exception as exc:
                logger.warning("MindVisionCamera[%d] set_gain error: %s", self.camera_index, exc)

    def get_temperature(self) -> float:
        return 42.0

    def set_roi(self, width: int, height: int, offset_x: int, offset_y: int) -> bool:
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_opencv_fallback(self) -> bool:
        """Open the camera using OpenCV's V4L2 backend as a fallback."""
        logger.warning(
            "MindVisionCamera[%d]: using OpenCV/V4L2 fallback.", self.camera_index
        )
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True


# ---------------------------------------------------------------------------
# Mock / simulation camera
# ---------------------------------------------------------------------------

class MockCamera(CameraInterface):
    """
    Synthetic camera that generates a simulated 3D bouncing soccer ball scene.
    Used for offline development and algorithm validation without physical hardware.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        is_left: bool = True,
        video_path: Optional[str] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.is_left = is_left
        self.video_path = video_path

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._last_timestamp: float = 0.0
        
        self.offset_x = 0
        self.offset_y = 0

        # Stereo projection parameters (must match StereoTriangulator config)
        self._baseline_mm: float = 300.0
        self._focal_px: float = 1200.0
        self._cx: float = width / 2.0
        self._cy: float = height / 2.0

    # ------------------------------------------------------------------
    # CameraInterface implementation
    # ------------------------------------------------------------------

    def open(self) -> bool:
        if self.video_path:
            logger.info("MockCamera: opening video file '%s'.", self.video_path)
            self._cap = cv2.VideoCapture(self.video_path)
            return self._cap.isOpened()
        logger.info("MockCamera: synthetic projection mode (is_left=%s).", self.is_left)
        self._last_timestamp = time.time()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        time.sleep(1.0 / self.fps)
        self._last_timestamp = time.time()
        self._frame_count += 1

        if self.video_path:
            return self._read_video_frame()
        return True, self._render_synthetic_frame()

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("MockCamera closed.")

    def get_timestamp(self) -> float:
        return self._last_timestamp

    def set_exposure(self, us: int) -> None:
        pass  # Mock camera has no real exposure

    def set_gain(self, gain: float) -> None:
        pass  # Mock camera has no real gain

    def get_temperature(self) -> float:
        # Simulate a normal operating temperature for testing
        return 38.5 + (time.time() % 10) / 5.0

    def set_roi(self, width: int, height: int, offset_x: int, offset_y: int) -> bool:
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_video_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                # Loop video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if ret and frame is not None:
                # Resize to full sensor resolution first to simulate raw capture
                full_w = 1280
                full_h = 720
                if frame.shape[1] != full_w or frame.shape[0] != full_h:
                    frame = cv2.resize(frame, (full_w, full_h))
                # Apply ROI Crop
                ox, oy = self.offset_x, self.offset_y
                if ox + self.width <= full_w and oy + self.height <= full_h:
                    frame = frame[oy:oy+self.height, ox:ox+self.width]
                else:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame
            return False, None
        return False, None

    def _render_synthetic_frame(self) -> np.ndarray:
        """Render a single synthetic frame with a 3D-projected bouncing ball."""
        t = self._frame_count / self.fps

        # 3-D trajectory of the simulated ball
        z_3d = 3000.0 - 4000.0 * (t % 1.5)
        if z_3d < 100.0:
            z_3d = 3000.0
            self._frame_count = 0
        x_3d = 400.0 * math.sin(2 * math.pi * 0.5 * t)
        y_3d = 200.0 + 300.0 * abs(math.cos(2 * math.pi * 0.8 * t)) - 100.0

        # Pinhole projection (left camera shifts X right, right shifts left)
        x_offset = (self._baseline_mm / 2.0) if self.is_left else -(self._baseline_mm / 2.0)
        x_px = int(self._cx + self._focal_px * (x_3d + x_offset) / z_3d)
        y_px = int(self._cy - self._focal_px * y_3d / z_3d)
        r_px = max(5, min(int(35.0 * self._focal_px / z_3d), 150))

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_grid(frame)
        self._draw_ball(frame, (x_px, y_px), r_px, t)
        self._draw_hud(frame, x_3d, y_3d, z_3d)
        return frame

    def _draw_grid(self, frame: np.ndarray) -> None:
        grid_color = (40, 40, 40)
        for y in range(0, self.height, 100):
            cv2.line(frame, (0, y), (self.width, y), grid_color, 1)
        for x in range(0, self.width, 100):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)

    def _draw_ball(self, frame: np.ndarray, center: Tuple[int, int], radius: int, t: float) -> None:
        cx, cy = center
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return
        cv2.circle(frame, center, radius, (30, 120, 240), -1)
        cv2.circle(frame, center, radius, (0, 0, 0), 2)
        for deg in range(0, 360, 72):
            rad = math.radians(deg + t * 45)
            px = int(cx + radius * 0.5 * math.cos(rad))
            py = int(cy + radius * 0.5 * math.sin(rad))
            pt = (px, py)
            cv2.circle(frame, pt, max(2, int(radius * 0.15)), (0, 0, 0), -1)
            cv2.line(frame, center, pt, (0, 0, 0), 1)

    def _draw_hud(self, frame: np.ndarray, x: float, y: float, z: float) -> None:
        label = "LEFT CAMERA (MOCK)" if self.is_left else "RIGHT CAMERA (MOCK)"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Sim 3D: [{x:.0f}, {y:.0f}, {z:.0f}] mm",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
