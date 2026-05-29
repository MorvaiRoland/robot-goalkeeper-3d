"""
ball_detector.py  –  Optimalizált fehér focilabda detektor (5-ös méret)

Detektálási pipeline (3 réteg + Kalman):
─────────────────────────────────────────
Layer 1 – YOLOv8l  (large model, pontosabb mint nano/small)
    • COCO class 32 „sports ball"
    • Mindkét sztereó képkocka egyetlen GPU batch-hívásban

Layer 2 – Adaptív HSV + Kontúr / Hough hibrid  (fallback)
    • Fehér labda detektálása: a labda BELSEJE vegyes (fehér + sötét minták)
      → ezért nem csak fehér pixeleket keresünk, hanem:
        a) fehér külső gyűrű maszkot keresünk (annular mask)
        b) morphological closing → a mintákat kitölti
        c) Hough + kontúr konvex hull → a legjobb kör kiválasztása
    • Adaptive thresholding a fény-változáshoz

Layer 3 – Kalman szűrő  (coasting, ha egyik sem talál)
    • Állapotvektor: [x, y, vx, vy]
    • max_coast_frames után reset

Vizualizáció:
─────────────
    • draw_detection() – a labda teljes területét kitölti félátlátszó overlay-jel
      + vastag körrajz + crosshair + metódus felirat
    • Szín kódolás:
        Zöld  → YOLO
        Sárga → HSV/Kontúr
        Cyan  → Kalman (prediktált)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detection.ball_tracker import BallKalmanTracker

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Eredmény adatstruktúra
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Egységes detektálási eredmény bármelyik rétegtől.

    Mezők
    -----
    success      : True ha érvényes pozíció elérhető (akár prediktált).
    x, y         : Labda középpont pixelben.
    radius       : Becsült labda sugár pixelben.
    confidence   : 0.0–1.0 pontszám.
    method       : "yolo" | "hsv" | "kalman" | "none"
    is_predicted : True ha csak Kalman extrapoláció (nincs nyers detektálás).
    contour      : Opcionális kontúr pontok (numpy array) teljesebb vizualizációhoz.
    """
    success: bool = False
    x: int = 0
    y: int = 0
    radius: int = 0
    confidence: float = 0.0
    method: str = "none"
    is_predicted: bool = False
    contour: Optional[np.ndarray] = None

    def as_tuple(self) -> Optional[Tuple[int, int, int]]:
        return (self.x, self.y, self.radius) if self.success else None


# ──────────────────────────────────────────────────────────────────────────────
# Vizualizáció – teljes labda területének kiszínezése
# ──────────────────────────────────────────────────────────────────────────────

# Metódus → szín (BGR)
_COLOR_YOLO   = (0,   220,  50)   # zöld
_COLOR_HSV    = (0,   200, 255)   # sárga/narancs
_COLOR_KALMAN = (255, 200,   0)   # cyan
_COLOR_MAP = {
    "yolo":   _COLOR_YOLO,
    "hsv":    _COLOR_HSV,
    "kalman": _COLOR_KALMAN,
    "none":   (0, 0, 255),
}


def draw_detection(frame: np.ndarray, result: DetectionResult, alpha: float = 0.35) -> np.ndarray:
    """
    A labda teljes területét kiszínezi félátlátszó overlay-jel,
    vastag körrajzot, crosshairt és metódus feliratot rajzol rá.

    Args:
        frame  : BGR képkocka (in-place módosítás)
        result : DetectionResult
        alpha  : Az overlay átlátszósága (0=teljesen átlátszó, 1=teli)

    Returns:
        A módosított frame (ugyanaz az objektum, in-place).
    """
    if not result.success:
        return frame

    color = _COLOR_MAP.get(result.method, (255, 255, 255))
    cx, cy, r = result.x, result.y, max(result.radius, 10)

    # ── 1. Teljes labda kitöltése overlay-jel ────────────────────────────────
    overlay = frame.copy()

    if result.contour is not None and len(result.contour) >= 5:
        # Ha van pontos kontúr: konvex hull kitöltve
        hull = cv2.convexHull(result.contour)
        cv2.fillConvexPoly(overlay, hull, color)
    else:
        # Kör alapú kitöltés
        cv2.circle(overlay, (cx, cy), r, color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── 2. Vastag körrajz ────────────────────────────────────────────────────
    thickness = 1 if result.is_predicted else 3
    cv2.circle(frame, (cx, cy), r, color, thickness)
    if result.is_predicted:
        # Szaggatott hatás: második, kicsit nagyobb kör
        cv2.circle(frame, (cx, cy), r + 4, color, 1)

    # ── 3. Crosshair ─────────────────────────────────────────────────────────
    arm = r + 10
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, 2)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)

    # ── 4. Felirat ───────────────────────────────────────────────────────────
    pred_tag = " [~]" if result.is_predicted else ""
    label = f"{result.method.upper()}{pred_tag} {result.confidence:.2f}"
    lx, ly = cx + r + 6, cy - r
    # Fekete háttér a jobb olvashatóságért
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Fő detektor
# ──────────────────────────────────────────────────────────────────────────────

class BallDetector:
    """
    Háromrétegű hibrid labda detektor sztereó kamera rendszerhez.

    Legfontosabb változások a korábbi verzióhoz képest:
    • YOLOv8l (large) alapértelmezetten → pontosabb detektálás
    • Adaptív HSV fallback: a fehér labda sötét mintái miatt
      nem csupán fehér pixeleket keresünk, hanem:
        – fehér gyűrű (annular) maszkot
        – morphological closing a minták kitöltéséhez
        – konvex hull kontúr alapú kör becslés
    • draw_detection() integráció: a vizualizáció mindig a teljes
      labda területét színezi ki
    """

    _COCO_BALL_CLASS = 32

    def __init__(
        self,
        method: str = "hybrid",
        yolo_model_path: str = "yolov8l.pt",   # Large model alapértelmezetten
        hsv_bounds: Optional[Dict[str, int]] = None,
        hough_cfg: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.35,    # Kicsit alacsonyabb küszöb
        kalman_cfg: Optional[Dict[str, Any]] = None,
        roi_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.method = method.lower()
        self.confidence_threshold = confidence_threshold

        # ── HSV határok (fehér focilabda napfényben/beltéren) ────────────────
        b = hsv_bounds or {}
        self.lower_hsv = np.array([
            b.get("lower_h",   0),
            b.get("lower_s",   0),
            b.get("lower_v", 160),   # Kicsit alacsonyabb V → árnyékos labda is látszik
        ], dtype=np.uint8)
        self.upper_hsv = np.array([
            b.get("upper_h", 180),
            b.get("upper_s",  60),   # Alacsony szaturáció → fehér
            b.get("upper_v", 255),
        ], dtype=np.uint8)

        # ── Hough konfig ─────────────────────────────────────────────────────
        hc = hough_cfg or {}
        self._hough_min_dist   = hc.get("min_dist",   25)
        self._hough_param1     = hc.get("param1",     80)  # Engedékenyebb él detektor
        self._hough_param2     = hc.get("param2",     25)  # Több jelölt kör
        self._hough_min_radius = hc.get("min_radius",  8)
        self._hough_max_radius = hc.get("max_radius", 150)

        # ── ROI konfig ───────────────────────────────────────────────────────
        rc = roi_cfg or {}
        self._roi_enabled        = rc.get("enabled", True)
        self._roi_padding_factor = rc.get("padding_factor", 3.0)  # Nagyobb ROI

        self._roi_left:  Optional[Tuple[int, int, int, int]] = None
        self._roi_right: Optional[Tuple[int, int, int, int]] = None

        # ── Kalman trackerek (kameránként egy) ───────────────────────────────
        kc = kalman_cfg or {}
        kalman_kwargs = {
            "process_noise":     kc.get("process_noise",    1e-2),
            "measurement_noise": kc.get("measurement_noise", 1e-1),
            "max_coast_frames":  kc.get("max_coast_frames",  12),
        }
        self._kalman_left  = BallKalmanTracker(**kalman_kwargs)
        self._kalman_right = BallKalmanTracker(**kalman_kwargs)

        # ── YOLO ─────────────────────────────────────────────────────────────
        self.yolo_model = None
        if self.method in ("yolo", "hybrid"):
            self._init_yolo(yolo_model_path)

        logger.info(
            "BallDetector kész | method=%s | yolo=%s | conf_thr=%.2f",
            self.method,
            "betöltve" if self.yolo_model else "nem elérhető",
            self.confidence_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO inicializálás
    # ──────────────────────────────────────────────────────────────────────────

    def _init_yolo(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            # Warm-up: üres predikció a GPU JIT kompilációhoz
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.yolo_model.predict(dummy, verbose=False, conf=self.confidence_threshold)
            logger.info("YOLOv8 modell betöltve és warm-up kész: %s", model_path)
        except ImportError:
            logger.warning("ultralytics nincs telepítve – YOLO réteg kikapcsolva.")
            if self.method == "yolo":
                self.method = "hsv"
        except Exception as exc:
            logger.warning("Nem sikerült betölteni a YOLO modellt '%s': %s", model_path, exc)
            if self.method == "yolo":
                self.method = "hsv"

    # ──────────────────────────────────────────────────────────────────────────
    # Publikus API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame is None:
            return DetectionResult()
        result, new_roi = self._detect_frame(frame, self._kalman_left, self._roi_left)
        self._roi_left = new_roi
        return result

    def detect_stereo(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
    ) -> Tuple[DetectionResult, DetectionResult]:
        if frame_left is None or frame_right is None:
            return DetectionResult(), DetectionResult()

        yolo_left: Optional[DetectionResult] = None
        yolo_right: Optional[DetectionResult] = None

        # YOLO batch (egyetlen GPU hívás mindkét képkockához)
        if self.method in ("yolo", "hybrid") and self.yolo_model is not None:
            yolo_left, yolo_right = self._yolo_batch(frame_left, frame_right)

        result_left,  self._roi_left  = self._finalise(
            frame_left,  yolo_left,  self._kalman_left,  self._roi_left
        )
        result_right, self._roi_right = self._finalise(
            frame_right, yolo_right, self._kalman_right, self._roi_right
        )
        return result_left, result_right

    # ──────────────────────────────────────────────────────────────────────────
    # Belső pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _finalise(
        self,
        frame: np.ndarray,
        yolo_result: Optional[DetectionResult],
        kalman: BallKalmanTracker,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[DetectionResult, Optional[Tuple[int, int, int, int]]]:
        # Réteg 1: YOLO
        if yolo_result is not None and yolo_result.success:
            sx, sy = kalman.update(yolo_result.x, yolo_result.y)
            yolo_result.x, yolo_result.y = sx, sy
            return yolo_result, self._make_roi(yolo_result, frame.shape)

        # Réteg 2: Adaptív HSV + Kontúr/Hough
        if self.method in ("hsv", "hybrid"):
            hsv_result = self._detect_adaptive(frame, roi)
            if hsv_result.success:
                sx, sy = kalman.update(hsv_result.x, hsv_result.y)
                hsv_result.x, hsv_result.y = sx, sy
                return hsv_result, self._make_roi(hsv_result, frame.shape)

        # Réteg 3: Kalman coasting
        predicted = kalman.predict()
        if predicted is not None:
            return DetectionResult(
                success=True,
                x=predicted[0], y=predicted[1],
                radius=self._last_radius(kalman),
                confidence=kalman.confidence,
                method="kalman",
                is_predicted=True,
            ), roi

        return DetectionResult(), None

    def _detect_frame(
        self,
        frame: np.ndarray,
        kalman: BallKalmanTracker,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[DetectionResult, Optional[Tuple[int, int, int, int]]]:
        yolo_result: Optional[DetectionResult] = None
        if self.method in ("yolo", "hybrid") and self.yolo_model is not None:
            results = self.yolo_model.predict(frame, verbose=False, conf=self.confidence_threshold)
            yolo_result = self._parse_yolo_result(results[0])
        return self._finalise(frame, yolo_result, kalman, roi)

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO
    # ──────────────────────────────────────────────────────────────────────────

    def _yolo_batch(
        self, frame_left: np.ndarray, frame_right: np.ndarray
    ) -> Tuple[Optional[DetectionResult], Optional[DetectionResult]]:
        try:
            results = self.yolo_model.predict(
                [frame_left, frame_right],
                verbose=False,
                conf=self.confidence_threshold,
            )
            return (
                self._parse_yolo_result(results[0]),
                self._parse_yolo_result(results[1]),
            )
        except Exception as exc:
            logger.warning("YOLO batch hiba: %s", exc)
            return None, None

    def _parse_yolo_result(self, result: Any) -> Optional[DetectionResult]:
        best_conf = 0.0
        best_box  = None

        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != self._COCO_BALL_CLASS:
                continue
            if conf < self.confidence_threshold:
                continue
            xyxy   = box.xyxy[0].cpu().numpy()
            width  = float(xyxy[2] - xyxy[0])
            height = float(xyxy[3] - xyxy[1])
            if height < 1:
                continue
            aspect = width / height
            if not (0.35 < aspect < 2.8):   # Engedékenyebb aspect ratio
                continue
            if conf > best_conf:
                best_conf = conf
                best_box  = xyxy

        if best_box is None:
            return None

        x_center = int((best_box[0] + best_box[2]) / 2)
        y_center = int((best_box[1] + best_box[3]) / 2)
        radius   = int((best_box[2] - best_box[0] + best_box[3] - best_box[1]) / 4)

        return DetectionResult(
            success=True,
            x=x_center, y=y_center,
            radius=max(radius, 1),
            confidence=best_conf,
            method="yolo",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Adaptív HSV + Kontúr / Hough hibrid  ← FŐ ÚJÍTÁS
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_adaptive(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> DetectionResult:
        """
        Fehér focilabda detektálása adaptív módszerekkel.

        A fehér labdán sötét minták (pentagonok) vannak → a labda belseje
        nem teljesen fehér. Ezért:

        1. Széles HSV maszk (fehér + világos szürke) + morphological closing
           → a minták kitöltése, összefüggő fehér blob
        2. Kontúr alapú keresés: legkerekebb, megfelelő méretű blob
        3. Hough fallback ha kontúr nem ad eredményt
        4. CLAHE pre-processzálás adaptív fényességhez
        """
        h_frame, w_frame = frame.shape[:2]
        x_off = y_off = 0
        work = frame

        # ROI kivágás
        if self._roi_enabled and roi is not None:
            x1, y1, x2, y2 = roi
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_frame, x2); y2 = min(h_frame, y2)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                work  = frame[y1:y2, x1:x2]
                x_off = x1
                y_off = y1

        # ── CLAHE pre-processzálás (adaptív kontraszt) ────────────────────────
        lab   = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ── Fő maszk: fehér + világos területek ──────────────────────────────
        hsv  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # ── Morfológiai closing: a sötét mintákat kitölti ────────────────────
        # Nagyobb kernel = több mintát tölt ki (de lassabb)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)  # minták kitöltése
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)   # zaj eltávolítás

        # ── Kontúr alapú detektálás ───────────────────────────────────────────
        result = self._contour_detection(mask, x_off, y_off, h_frame, w_frame)
        if result.success:
            return result

        # ── Hough fallback ────────────────────────────────────────────────────
        return self._hough_detection(mask, x_off, y_off)

    def _contour_detection(
        self,
        mask: np.ndarray,
        x_off: int, y_off: int,
        h_full: int, w_full: int,
    ) -> DetectionResult:
        """
        Kontúr alapú detektor.
        A maszkon megkeresi az összes kontúrt, és a legkerekebb,
        megfelelő méretű blob-ot választja ki labdaként.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return DetectionResult()

        best_score = 0.0
        best_cnt   = None
        best_cx = best_cy = best_r = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Méret szűrés: min ~50px² (nagyon közeli kis labda)  max ~50000px²
            if area < 50 or area > 55000:
                continue

            # Kerekség = 4π·terület / kerület²  (tökéletes kör = 1.0)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.45:   # Durván kör alakú blob szűrés
                continue

            # Befoglaló kör számítása
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx = int(cx) + x_off
            cy = int(cy) + y_off
            r  = int(radius)

            # Sugár szűrés képmérethez viszonyítva
            if r < self._hough_min_radius or r > self._hough_max_radius:
                continue

            # Konvex hull tömörség ellenőrzés
            hull   = cv2.convexHull(cnt)
            hull_a = cv2.contourArea(hull)
            solidity = area / hull_a if hull_a > 0 else 0
            if solidity < 0.6:   # Ne fogadjon el C alakú blobokat
                continue

            # Összpontszám: kerekség × tömörség × terület-normált
            score = circularity * solidity
            if score > best_score:
                best_score = score
                best_cnt   = cnt
                best_cx, best_cy, best_r = cx, cy, r

        if best_cnt is None or best_score < 0.5:
            return DetectionResult()

        # Kontúr koordinátáit eltoljuk a teljes képkoordináta-rendszerbe
        shifted_cnt = best_cnt.copy()
        shifted_cnt[:, 0, 0] += x_off
        shifted_cnt[:, 0, 1] += y_off

        # Konfidencia: kerekség + tömörség → 0.55–0.88
        conf = min(0.55 + 0.33 * best_score, 0.88)

        return DetectionResult(
            success=True,
            x=best_cx, y=best_cy,
            radius=best_r,
            confidence=conf,
            method="hsv",
            contour=shifted_cnt,
        )

    def _hough_detection(
        self,
        mask: np.ndarray,
        x_off: int, y_off: int,
    ) -> DetectionResult:
        """Hough Circle Transform fallback a kontúr detektálás után."""
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self._hough_min_dist,
            param1=self._hough_param1,
            param2=self._hough_param2,
            minRadius=self._hough_min_radius,
            maxRadius=self._hough_max_radius,
        )
        if circles is None:
            return DetectionResult()

        circles = np.round(circles[0]).astype(int)
        best    = circles[0]
        cx = int(best[0]) + x_off
        cy = int(best[1]) + y_off
        r  = int(best[2])

        if r < 2:
            return DetectionResult()

        # Lefedettség ellenőrzés a maszkon (alacsonyabb küszöb: 15%)
        coverage = self._circle_mask_coverage(mask, int(best[0]), int(best[1]), r)
        if coverage < 0.15:
            return DetectionResult()

        conf = 0.45 + 0.3 * min(coverage, 1.0)
        return DetectionResult(
            success=True,
            x=cx, y=cy, radius=r,
            confidence=conf,
            method="hsv",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Segédfüggvények
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _circle_mask_coverage(mask: np.ndarray, cx: int, cy: int, r: int) -> float:
        h, w = mask.shape[:2]
        cmask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cmask, (cx, cy), max(r, 1), 255, -1)
        ca = np.count_nonzero(cmask)
        if ca == 0:
            return 0.0
        return np.count_nonzero(cv2.bitwise_and(mask, cmask)) / ca

    def _make_roi(
        self,
        result: DetectionResult,
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not result.success or result.radius < 1:
            return None
        pad = int(result.radius * self._roi_padding_factor)
        h, w = frame_shape[:2]
        return (
            max(0,     result.x - pad),
            max(0,     result.y - pad),
            min(w - 1, result.x + pad),
            min(h - 1, result.y + pad),
        )

    @staticmethod
    def _last_radius(kalman: BallKalmanTracker) -> int:
        return 18   # Ésszerű alapértelmezés coasting közben
