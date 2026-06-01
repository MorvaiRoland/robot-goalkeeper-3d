"""
BallTrajectoryPredictor – Profi 3D pályaprediktor fizikai Kalman-szűrővel.

Állapotvektor: [X, Y, Z, Vx, Vy, Vz]  (mm és mm/s egységekben)
Mérésvektor:   [X, Y, Z]
Vezérlővektor: [Ax, Ay, Az] = [0, -g, 0]  (gravitáció Y irányban lefelé)

Használat:
    predictor = BallTrajectoryPredictor(gravity_mm_s2=9810.0, goal_z_mm=600.0)

    # Minden triangulált 3D ponthoz:
    predictor.update(x, y, z, dt)

    # Becsapódási pont lekérése:
    impact = predictor.get_impact_point()   # → (x_mm, y_mm, t_s) | None

    # Pályapontok a vizualizációhoz:
    pts = predictor.get_trajectory_points(n=20)  # → [(x,y,z), ...]

    # Konfidencia:
    conf = predictor.confidence             # → 0.0 … 1.0
"""

import cv2
import math
import logging
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BallTrajectoryPredictor
# ──────────────────────────────────────────────────────────────────────────────

class BallTrajectoryPredictor:
    """
    Fizikai alapú Kalman-szűrős 3D pályaprediktor labdakövetéshez.

    Koordinátarendszer (StereoTriangulator egyezmény):
        X  – vízszintes (balra negatív, jobbra pozitív)
        Y  – magassági (felfelé pozitív, lefelé negatív)
        Z  – mélység / távolság a kameráktól (pozitív = kameráktól távolabb)

    A kapu-sík Z = goal_z_mm pozícióban van.  A labda általában közeledik
    (Vz < 0), tehát a becsapódás t = (goal_z - Z) / Vz pillanatban következik.
    """

    # Minimum frame-szám a megbízható predikciókhoz
    _MIN_FRAMES_FOR_PREDICTION: int = 2

    # Maximum idő (s), ami után a predikciót elvetjük (nem hihető)
    _MAX_IMPACT_TIME_S: float = 5.0

    # Minimális Z-sebesség (mm/s) a labda közeledéséhez (csak impact-ponthoz kell)
    _MIN_VZ_FOR_PREDICTION: float = 20.0

    def __init__(
        self,
        gravity_mm_s2: float = 9810.0,
        goal_z_mm: float = 600.0,
    ) -> None:
        """
        :param gravity_mm_s2: Gravitációs gyorsulás mm/s²-ban (alapért. 9810).
        :param goal_z_mm:     A kapu-sík Z-koordinátája mm-ben.
        """
        self.gravity = gravity_mm_s2
        self.goal_z = goal_z_mm

        # ── OpenCV Kalman-szűrő ──────────────────────────────────────────────
        # 6 állapot: [X, Y, Z, Vx, Vy, Vz]
        # 3 mérés:   [X, Y, Z]
        # 3 vezérlő: [Ax, Ay, Az]
        self.kf = cv2.KalmanFilter(6, 3, 3)

        # Mérési mátrix H (csak pozíciót mérünk)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Folyamatzaj-kovariancia Q – alapértelmezett (dt-vel frissül)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-1

        # Mérési zaj-kovariancia R – triangulációs bizonytalanság
        self.kf.measurementNoiseCov = np.diag(
            [5.0, 5.0, 15.0]  # Z mérés zajosabb (mélység-becslés)
        ).astype(np.float32)

        # Kezdeti hiba-kovariancia P
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 500.0

        # Gravitációs vezérlő-bemenet u = [0, -g, 0]
        self._u = np.array([[0.0], [-self.gravity], [0.0]], dtype=np.float32)

        # ── Belső állapot ────────────────────────────────────────────────────
        self._initialized: bool = False
        self._frame_count: int = 0          # Eddigi frissítések száma
        self._coast_count: int = 0          # Egymást követő mérés nélküli lépések

        # Sebesség simítás (EMA)
        self._vel_ema = np.zeros(3, dtype=np.float64)
        self._vel_ema_alpha = 0.3

        # Utolsó kiszámított becsapódási pont (cache)
        self._cached_impact: Optional[Tuple[float, float, float]] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Publikus API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, x: float, y: float, z: float, dt: float) -> None:
        """
        Új 3D mérési pontot ad a szűrőnek.

        :param x:  Labda X koordinátája mm-ben.
        :param y:  Labda Y koordinátája mm-ben.
        :param z:  Labda Z koordinátája mm-ben.
        :param dt: Az előző mérés óta eltelt idő másodpercben.
        """
        if dt <= 0 or dt > 0.5:
            # Nem reális dt → csak coasting
            self._coast_count += 1
            if self._initialized:
                self._update_matrices(min(dt, 0.1))
                self.kf.predict(self._u)
                self._cached_impact = self._compute_impact()
            return

        measurement = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])

        if not self._initialized:
            # Első mérés: állapot inicializálása nulla-sebességgel
            self.kf.statePost = np.array(
                [[x], [y], [z], [0.0], [0.0], [0.0]], dtype=np.float32
            )
            self._initialized = True
            self._frame_count = 1
            self._coast_count = 0
            logger.debug("TrajectoryPredictor init @ (%.0f, %.0f, %.0f) mm", x, y, z)
            return

        # Mátrixok frissítése az aktuális dt-vel
        self._update_matrices(dt)

        # Folyamatzaj adaptálása a sebesség nagyságához
        self._adapt_process_noise(dt)

        # Kalman: előrejelzés + korrekció
        self.kf.predict(self._u)
        self.kf.correct(measurement)

        # Sebesség EMA frissítése
        vel = self.kf.statePost[3:6, 0].astype(np.float64)
        self._vel_ema = (
            self._vel_ema_alpha * vel
            + (1.0 - self._vel_ema_alpha) * self._vel_ema
        )

        self._frame_count += 1
        self._coast_count = 0

        # Becsapódási pont cache-elése
        self._cached_impact = self._compute_impact()

    def get_impact_point(self) -> Optional[Tuple[float, float, float]]:
        """
        Visszaadja a becsült becsapódási pontot.

        :return: (X_mm, Y_mm, t_impact_s) tuple, vagy None ha nem meghatározható.
                 X_mm, Y_mm: kapu-síkban a labda becsült pozíciója mm-ben.
                 t_impact_s: becsült hátralévő idő másodpercben.
        """
        if self._frame_count < self._MIN_FRAMES_FOR_PREDICTION:
            return None
        return self._cached_impact

    def get_trajectory_points(
        self,
        n: int = 20,
        t_max: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Szimulált jövőbeli pozíciók listáját adja vissza.

        :param n:     Generálandó pontok száma.
        :param t_max: Maximum szimulációs idő (s). Ha None, impact-time-ig vagy 2s-ig.
        :return: [(X_mm, Y_mm, Z_mm), ...] lista n elemmel.
        """
        if not self._initialized or self._frame_count < self._MIN_FRAMES_FOR_PREDICTION:
            return []

        state = self.kf.statePost
        x0 = float(state[0, 0])
        y0 = float(state[1, 0])
        z0 = float(state[2, 0])
        vx = float(state[3, 0])
        vy = float(state[4, 0])
        vz = float(state[5, 0])

        # Ha nincs impact, 2 másodpercig szimulálunk
        if t_max is None:
            if self._cached_impact is not None:
                t_max = min(self._cached_impact[2] * 1.05, self._MAX_IMPACT_TIME_S)
            else:
                t_max = 2.0

        t_max = max(t_max, 0.05)
        points: List[Tuple[float, float, float]] = []

        for i in range(n):
            t = t_max * (i / max(n - 1, 1))
            px = x0 + vx * t
            py = y0 + vy * t - 0.5 * self.gravity * (t ** 2)
            pz = z0 + vz * t
            points.append((px, py, pz))

        return points

    def get_velocity_mms(self) -> Tuple[float, float, float]:
        """
        Visszaadja az aktuális sebesség-vektort mm/s-ban (EMA simított).

        :return: (Vx_mms, Vy_mms, Vz_mms)
        """
        if not self._initialized:
            return 0.0, 0.0, 0.0
        return float(self._vel_ema[0]), float(self._vel_ema[1]), float(self._vel_ema[2])

    def get_speed_mms(self) -> float:
        """Visszaadja a labda skaláris sebességét mm/s-ban."""
        vx, vy, vz = self.get_velocity_mms()
        return math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    def get_current_state(self) -> Tuple[float, float, float, float, float, float]:
        """Visszaadja a szűrt állapotot: (X, Y, Z, Vx, Vy, Vz)."""
        if not self._initialized:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        s = self.kf.statePost
        return (
            float(s[0, 0]), float(s[1, 0]), float(s[2, 0]),
            float(s[3, 0]), float(s[4, 0]), float(s[5, 0]),
        )

    @property
    def confidence(self) -> float:
        """
        Konfidencia-szint [0.0 … 1.0] a jelenlegi predikció megbízhatóságáról.

        Figyelembe veszi:
          - Összegyűjtött mérési pontok száma
          - Z-irányú sebesség nagysága (mozog-e felénk?)
          - Coasting (mérés nélküli lépések) aránya
        """
        if not self._initialized or self._frame_count < 1:
            return 0.0

        # Mérési pontok hányada (5 pont felett teljes pontszám)
        frame_score = min(self._frame_count / 5.0, 1.0)

        # Z sebesség hányada (500 mm/s felett teljes pontszám)
        vz = abs(self._vel_ema[2])
        vel_score = min(vz / 500.0, 1.0)

        # Coasting büntetés
        coast_penalty = min(self._coast_count / 5.0, 1.0)

        raw = frame_score * 0.4 + vel_score * 0.6
        return max(0.0, raw - coast_penalty * 0.5)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        """Visszaállítja a szűrő állapotát (labda elvesztésekor hívandó)."""
        self._initialized = False
        self._frame_count = 0
        self._coast_count = 0
        self._vel_ema = np.zeros(3, dtype=np.float64)
        self._cached_impact = None
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 500.0
        logger.debug("TrajectoryPredictor reset.")

    # ──────────────────────────────────────────────────────────────────────────
    # Belső segédmetódusok
    # ──────────────────────────────────────────────────────────────────────────

    def _update_matrices(self, dt: float) -> None:
        """Frissíti az átmeneti F és vezérlő B mátrixokat az aktuális dt-vel."""
        dt2 = dt * dt

        # Állapot-átmeneti mátrix F (konstans-sebesség modell + gravitáció vezérlőn)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)

        # Vezérlő-mátrix B (gyorsulás → pozíció- és sebesség-változás)
        self.kf.controlMatrix = np.array([
            [0.5 * dt2, 0,         0        ],
            [0,         0.5 * dt2, 0        ],
            [0,         0,         0.5 * dt2],
            [dt,        0,         0        ],
            [0,         dt,        0        ],
            [0,         0,         dt       ],
        ], dtype=np.float32)

    def _adapt_process_noise(self, dt: float) -> None:
        """
        Adaptívan beállítja a Q folyamatzaj-kovarianciát a labda sebessége alapján.
        Gyorsan mozgó labdánál nagyobb zaj engedélyezett (pl. pattanás, rugó).
        """
        speed = self.get_speed_mms()
        # 0–500 mm/s között 0.05–2.0 tartomány
        q_scale = 0.05 + min(speed / 500.0, 1.0) * 1.95
        self.kf.processNoiseCov = (np.eye(6, dtype=np.float32) * q_scale * dt)

    def _compute_impact(self) -> Optional[Tuple[float, float, float]]:
        """
        Belső metódus: kiszámítja a kapu-síkkal való metszéspontot.

        :return: (X_impact_mm, Y_impact_mm, t_impact_s) vagy None.
        """
        if not self._initialized:
            return None

        s = self.kf.statePost
        x0  = float(s[0, 0])
        y0  = float(s[1, 0])
        z0  = float(s[2, 0])
        vx  = float(s[3, 0])
        vy  = float(s[4, 0])
        vz  = float(s[5, 0])

        # A labdának közelednie kell a kapu felé
        dz = self.goal_z - z0
        if abs(vz) < self._MIN_VZ_FOR_PREDICTION:
            return None
        # Ha a távolság és a sebesség ellentétes előjelű → labda távolodik
        if dz * vz < 0:
            return None

        t_impact = dz / vz

        if t_impact <= 0 or t_impact > self._MAX_IMPACT_TIME_S:
            return None

        # Pozíció a becsapódás pillanatában (parabolikus Y-trajektória)
        x_impact = x0 + vx * t_impact
        y_impact = y0 + vy * t_impact - 0.5 * self.gravity * (t_impact ** 2)

        return x_impact, y_impact, t_impact
