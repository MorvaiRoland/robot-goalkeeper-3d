#!/usr/bin/env python3
"""
pc_monitor.py  –  PC / Laptop GUI monitor oldal
══════════════════════════════════════════════════════════════════════
Ez a program a PC-n fut. Ethernet / WiFi kapcsolaton keresztül
fogadja az RPi telemetriáját és megjeleníti a GUI-ban.

Feladatai:
  - UDP fogadás az RPi-től (kamera preview + detektálási adatok)
  - A meglévő GUI megjelenítése (kép, 3D, adatok, konfig)
  - UDP küldés az RPi-nek (konfigurációs parancsok)
  - Felvétel (DVR), debug, profil kezelés

Semmi számítást nem végez – minden adat az RPi-től jön!

Indítás:
    python3 src/pc_monitor.py
    python3 src/pc_monitor.py --rpi-ip 192.168.10.1
"""

import argparse
import base64
import json
import logging
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QFont, QTextCursor, QAction
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QSlider, QGroupBox, QFormLayout, QPushButton, QComboBox,
        QPlainTextEdit, QMessageBox, QSizePolicy, QDockWidget, QTabWidget,
        QLineEdit, QToolBar, QCheckBox, QSpinBox
    )
    import qdarktheme
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError:
    print("Szükséges csomagok: pip3 install PyQt6 pyqtdarktheme pyqtgraph PyOpenGL psutil")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# RPi telemetria fogadó szál
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryReceiver(QThread):
    """
    Háttérszálban fogadja az RPi UDP telemetria csomagjait.
    Qt szignálon továbbítja a GUI felé.
    """

    telemetry_received = pyqtSignal(dict)

    def __init__(self, listen_port: int = 5010):
        super().__init__()
        self._listen_port = listen_port
        self._running = True
        self._sock: Optional[socket.socket] = None

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        self._sock.settimeout(1.0)
        self._sock.bind(("0.0.0.0", self._listen_port))
        logger.info("TelemetryReceiver listen ::%d", self._listen_port)

        while self._running:
            try:
                data, _ = self._sock.recvfrom(128 * 1024)
                payload = json.loads(data.decode("utf-8"))
                self.telemetry_received.emit(payload)
            except socket.timeout:
                pass
            except Exception as exc:
                if self._running:
                    logger.warning("Telemetria fogadási hiba: %s", exc)

        self._sock.close()

    def stop(self):
        self._running = False
        self.wait(2000)


# ─────────────────────────────────────────────────────────────────────────────
# RPi parancs küldő
# ─────────────────────────────────────────────────────────────────────────────

class RPiCommandSender:
    """Egyszerű UDP parancs küldő az RPi felé."""

    def __init__(self, rpi_ip: str, cmd_port: int = 5011):
        self._rpi_ip = rpi_ip
        self._cmd_port = cmd_port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info("RPiCommandSender → %s:%d", rpi_ip, cmd_port)

    def send(self, cmd: str, value: Any = None):
        try:
            payload = {"cmd": cmd}
            if value is not None:
                payload["value"] = value
            msg = json.dumps(payload).encode("utf-8")
            self._sock.sendto(msg, (self._rpi_ip, self._cmd_port))
        except Exception as exc:
            logger.warning("Parancs küldési hiba: %s", exc)

    def close(self):
        self._sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Kép dekódoló segédfüggvény
# ─────────────────────────────────────────────────────────────────────────────

def _decode_frame(b64_str: str) -> Optional[np.ndarray]:
    """Base64 JPEG → numpy BGR kép."""
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _frame_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """numpy BGR → QPixmap a Qt megjelenítéshez."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


# ─────────────────────────────────────────────────────────────────────────────
# PC Monitor főablak
# ─────────────────────────────────────────────────────────────────────────────

class PCMonitorWindow(QMainWindow):
    """
    A PC monitorozó GUI főablaka.
    Megjeleníti az RPi-től kapott kamera képet és detektálási adatokat.
    Lehetővé teszi az RPi konfigurálását.
    """

    def __init__(self, rpi_ip: str, telemetry_port: int = 5010, cmd_port: int = 5011):
        super().__init__()
        self._rpi_ip = rpi_ip

        # Hálózat
        self._cmd_sender = RPiCommandSender(rpi_ip, cmd_port)
        self._telemetry  = TelemetryReceiver(telemetry_port)
        self._telemetry.telemetry_received.connect(self._on_telemetry)

        # Adatok
        self._last_frame_l: Optional[np.ndarray] = None
        self._last_frame_r: Optional[np.ndarray] = None
        self._fps_history: list = []
        self._det_fps_history: list = []
        self._no_signal_timer = 0.0

        self._build_ui()
        self._telemetry.start()

        # Frissítő timer (GUI animáció, FPS grafikon)
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_ui)
        self._ui_timer.start(33)  # ~30 Hz UI frissítés

    # ─────────────────────────────────────────────────────────────────────────
    # UI felépítés
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle(f"Robot Goalkeeper 3D – PC Monitor  [RPi: {self._rpi_ip}]")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(6)

        # ── Bal oldal: kamera képek ───────────────────────────────────────
        cam_panel = QVBoxLayout()

        self._lbl_cam_l = QLabel("Bal kamera – nincs jel")
        self._lbl_cam_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cam_l.setMinimumSize(560, 315)
        self._lbl_cam_l.setStyleSheet("background:#111; color:#888; border:1px solid #333;")

        self._lbl_cam_r = QLabel("Jobb kamera – nincs jel")
        self._lbl_cam_r.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cam_r.setMinimumSize(560, 315)
        self._lbl_cam_r.setStyleSheet("background:#111; color:#888; border:1px solid #333;")

        cam_panel.addWidget(QLabel("📷 Bal kamera (RPi preview)"))
        cam_panel.addWidget(self._lbl_cam_l)
        cam_panel.addWidget(QLabel("📷 Jobb kamera (RPi preview)"))
        cam_panel.addWidget(self._lbl_cam_r)
        main_layout.addLayout(cam_panel, stretch=3)

        # ── Jobb oldal: adatok + vezérlők ────────────────────────────────
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # ── Kapcsolat státusz ─────────────────────────────────────────────
        conn_box = QGroupBox("📡 Kapcsolat")
        conn_layout = QFormLayout(conn_box)
        self._lbl_conn = QLabel("⏳ Várakozás...")
        self._lbl_rpi_ip = QLabel(self._rpi_ip)
        conn_layout.addRow("Státusz:", self._lbl_conn)
        conn_layout.addRow("RPi IP:", self._lbl_rpi_ip)
        right_panel.addWidget(conn_box)

        # ── Élő adatok ────────────────────────────────────────────────────
        live_box = QGroupBox("📊 Élő adatok")
        live_layout = QFormLayout(live_box)
        self._lbl_tracking  = QLabel("---")
        self._lbl_coords    = QLabel("---")
        self._lbl_cam_fps   = QLabel("---")
        self._lbl_det_fps   = QLabel("---")
        self._lbl_conf      = QLabel("---")
        self._lbl_pred      = QLabel("---")
        self._lbl_t_impact  = QLabel("---")
        self._lbl_speed     = QLabel("---")
        live_layout.addRow("Tracking:", self._lbl_tracking)
        live_layout.addRow("Koordináta (mm):", self._lbl_coords)
        live_layout.addRow("Camera FPS:", self._lbl_cam_fps)
        live_layout.addRow("Detection FPS:", self._lbl_det_fps)
        live_layout.addRow("Konfidencia:", self._lbl_conf)
        live_layout.addRow("Prediktált pont:", self._lbl_pred)
        live_layout.addRow("Becsapódás:", self._lbl_t_impact)
        live_layout.addRow("Sebesség:", self._lbl_speed)
        right_panel.addWidget(live_box)

        # ── FPS grafikon ──────────────────────────────────────────────────
        fps_box = QGroupBox("📈 FPS grafikon")
        fps_vbox = QVBoxLayout(fps_box)
        self._fps_plot = pg.PlotWidget(title="")
        self._fps_plot.setBackground("#1a1a1a")
        self._fps_plot.setYRange(0, 200)
        self._fps_plot.setMaximumHeight(120)
        self._fps_curve_cam = self._fps_plot.plot(pen=pg.mkPen("#4ecdc4", width=2), name="Camera")
        self._fps_curve_det = self._fps_plot.plot(pen=pg.mkPen("#ff6b6b", width=2), name="Detection")
        fps_vbox.addWidget(self._fps_plot)
        right_panel.addWidget(fps_box)

        # ── RPi konfig parancsok ──────────────────────────────────────────
        ctrl_box = QGroupBox("⚙️ RPi vezérlés")
        ctrl_layout = QVBoxLayout(ctrl_box)

        # Expozíció
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Expozíció (µs):"))
        self._spin_exposure = QSpinBox()
        self._spin_exposure.setRange(100, 50000)
        self._spin_exposure.setValue(800)
        self._btn_exposure = QPushButton("Alkalmaz")
        self._btn_exposure.clicked.connect(lambda: self._cmd_sender.send(
            "set_exposure", self._spin_exposure.value()))
        exp_row.addWidget(self._spin_exposure)
        exp_row.addWidget(self._btn_exposure)
        ctrl_layout.addLayout(exp_row)

        # Konfidencia küszöb
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Konfidencia:"))
        self._slider_conf = QSlider(Qt.Orientation.Horizontal)
        self._slider_conf.setRange(10, 90)
        self._slider_conf.setValue(35)
        self._lbl_conf_val = QLabel("0.35")
        self._slider_conf.valueChanged.connect(
            lambda v: self._lbl_conf_val.setText(f"{v/100:.2f}"))
        self._btn_conf = QPushButton("Alkalmaz")
        self._btn_conf.clicked.connect(lambda: self._cmd_sender.send(
            "set_conf_threshold", self._slider_conf.value() / 100.0))
        conf_row.addWidget(self._slider_conf)
        conf_row.addWidget(self._lbl_conf_val)
        conf_row.addWidget(self._btn_conf)
        ctrl_layout.addLayout(conf_row)

        # Gombok
        btn_row = QHBoxLayout()
        self._btn_reset_kalman = QPushButton("🔄 Kalman reset")
        self._btn_reset_kalman.clicked.connect(lambda: self._cmd_sender.send("reset_kalman"))
        self._btn_stop_rpi = QPushButton("⏹ RPi leállítás")
        self._btn_stop_rpi.setStyleSheet("background: #c0392b;")
        self._btn_stop_rpi.clicked.connect(self._confirm_stop_rpi)
        btn_row.addWidget(self._btn_reset_kalman)
        btn_row.addWidget(self._btn_stop_rpi)
        ctrl_layout.addLayout(btn_row)
        right_panel.addWidget(ctrl_box)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=1)

    # ─────────────────────────────────────────────────────────────────────────
    # Telemetria fogadás
    # ─────────────────────────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_telemetry(self, data: dict):
        """Az RPi-től érkező csomag feldolgozása – UI frissítés."""
        self._no_signal_timer = time.perf_counter()

        tracked   = bool(data.get("tracked", False))
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        z = float(data.get("z", 0))
        px = float(data.get("px", 0))
        py = float(data.get("py", 0))
        ti = float(data.get("ti", 0))
        cl = float(data.get("cl", 0))
        cr = float(data.get("cr", 0))
        cam_fps = float(data.get("fps", 0))
        det_fps = float(data.get("dfps", 0))
        speed   = float(data.get("spd", 0))

        # Tracking
        if tracked:
            self._lbl_tracking.setText("🟢 TRACKING")
            self._lbl_tracking.setStyleSheet("color: #2ecc71; font-weight:bold;")
        else:
            self._lbl_tracking.setText("🔴 ELVESZETT")
            self._lbl_tracking.setStyleSheet("color: #e74c3c; font-weight:bold;")

        self._lbl_coords.setText(f"X: {x:.0f}  Y: {y:.0f}  Z: {z:.0f}")
        self._lbl_cam_fps.setText(f"{cam_fps:.1f} FPS")
        self._lbl_det_fps.setText(f"{det_fps:.1f} FPS")
        self._lbl_conf.setText(f"L: {cl:.2f}  R: {cr:.2f}")
        self._lbl_pred.setText(f"X: {px:.0f}  Y: {py:.0f}")
        self._lbl_t_impact.setText(f"{ti*1000:.0f} ms")
        self._lbl_speed.setText(f"{speed/1000:.2f} m/s")

        # FPS history
        self._fps_history.append(cam_fps)
        self._det_fps_history.append(det_fps)
        if len(self._fps_history) > 120:
            self._fps_history.pop(0)
            self._det_fps_history.pop(0)

        # Képek dekódolása
        if "fl" in data:
            frame = _decode_frame(data["fl"])
            if frame is not None:
                self._last_frame_l = frame
        if "fr" in data:
            frame = _decode_frame(data["fr"])
            if frame is not None:
                self._last_frame_r = frame

    # ─────────────────────────────────────────────────────────────────────────
    # UI timer
    # ─────────────────────────────────────────────────────────────────────────

    def _update_ui(self):
        """~30 Hz-es UI frissítés: képek és FPS grafikon."""
        # Kapcsolat státusz
        since = time.perf_counter() - self._no_signal_timer
        if self._no_signal_timer == 0.0 or since > 3.0:
            self._lbl_conn.setText("❌ Nincs kapcsolat")
            self._lbl_conn.setStyleSheet("color: #e74c3c;")
        else:
            self._lbl_conn.setText(f"✅ Kapcsolva ({since*1000:.0f} ms)")
            self._lbl_conn.setStyleSheet("color: #2ecc71;")

        # Kamera képek megjelenítése
        if self._last_frame_l is not None:
            pix = _frame_to_qpixmap(self._last_frame_l)
            self._lbl_cam_l.setPixmap(
                pix.scaled(self._lbl_cam_l.size(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation))

        if self._last_frame_r is not None:
            pix = _frame_to_qpixmap(self._last_frame_r)
            self._lbl_cam_r.setPixmap(
                pix.scaled(self._lbl_cam_r.size(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation))

        # FPS grafikon
        if self._fps_history:
            self._fps_curve_cam.setData(self._fps_history)
            self._fps_curve_det.setData(self._det_fps_history)

    # ─────────────────────────────────────────────────────────────────────────

    def _confirm_stop_rpi(self):
        reply = QMessageBox.question(
            self, "RPi leállítása",
            "Biztosan leállítod az RPi trackert?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._cmd_sender.send("stop")
            logger.info("Stop parancs elküldve az RPi-nek.")

    def closeEvent(self, event):
        self._telemetry.stop()
        self._cmd_sender.close()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Belépési pont
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PC Monitor GUI – RPi telemetria megjelenítő")
    parser.add_argument("--rpi-ip", default="192.168.10.1",
                        help="RPi IP-cím (default: 192.168.10.1)")
    parser.add_argument("--telem-port", type=int, default=5010,
                        help="Telemetria fogadási port (default: 5010)")
    parser.add_argument("--cmd-port", type=int, default=5011,
                        help="RPi parancs port (default: 5011)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))

    win = PCMonitorWindow(
        rpi_ip=args.rpi_ip,
        telemetry_port=args.telem_port,
        cmd_port=args.cmd_port,
    )
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
