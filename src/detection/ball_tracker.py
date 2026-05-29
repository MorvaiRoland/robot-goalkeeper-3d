import logging
from typing import Tuple, Optional

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BallKalmanTracker:
    """
    A lightweight Kalman filter wrapper for tracking a 2D ball position across frames.

    State vector: [x, y, vx, vy]  (position + velocity)
    Measurement:  [x, y]

    Workflow per frame:
        - If a detection arrives  → call update(x, y) → returns smoothed position
        - If no detection arrives → call predict()    → returns extrapolated position
        - After max_coast_frames consecutive misses   → is_lost becomes True → reset()

    The 'confidence' property reflects how reliable the current estimate is:
        1.0  – fresh measurement-backed estimate
        0.0  – tracker has been coasting for max_coast_frames frames
    """

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        max_coast_frames: int = 10,
    ) -> None:
        """
        :param process_noise:     Q diagonal value – higher = trust the motion model less
        :param measurement_noise: R diagonal value – higher = trust detections less
        :param max_coast_frames:  How many frames we predict without a measurement before
                                  declaring the ball lost and resetting the filter.
        """
        self.max_coast_frames = max_coast_frames

        # OpenCV Kalman filter: 4 state dims, 2 measurement dims
        self._kf = cv2.KalmanFilter(4, 2)

        # Transition matrix (constant velocity model):
        # [ 1 0 1 0 ]   x  ← x + vx
        # [ 0 1 0 1 ]   y  ← y + vy
        # [ 0 0 1 0 ]   vx ← vx
        # [ 0 0 0 1 ]   vy ← vy
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Measurement matrix: we only observe x and y
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # Process noise covariance Q
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance R
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Initial error covariance (generous – we don't know state yet)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self._initialized: bool = False
        self._coast_count: int = 0
        self._last_estimate: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: int, y: int) -> Tuple[int, int]:
        """
        Feed a new measurement into the filter and return the corrected estimate.

        :param x: Measured ball center X (pixels)
        :param y: Measured ball center Y (pixels)
        :return:  Smoothed (x, y) in pixels
        """
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not self._initialized:
            # Seed the state with the first measurement
            self._kf.statePre = np.array(
                [[np.float32(x)], [np.float32(y)], [0.0], [0.0]],
                dtype=np.float32,
            )
            self._kf.statePost = self._kf.statePre.copy()
            self._initialized = True
            logger.debug("KalmanTracker initialized at (%d, %d).", x, y)

        self._kf.predict()
        corrected = self._kf.correct(measurement)
        self._coast_count = 0

        est = (int(corrected[0, 0]), int(corrected[1, 0]))
        self._last_estimate = est
        return est

    def predict(self) -> Optional[Tuple[int, int]]:
        """
        Advance the filter by one step without a measurement (coasting).

        :return: Predicted (x, y) or None if the tracker was never initialized
                 or has exceeded max_coast_frames.
        """
        if not self._initialized:
            return None

        self._coast_count += 1

        if self._coast_count > self.max_coast_frames:
            logger.debug(
                "KalmanTracker lost ball after %d coast frames. Resetting.",
                self.max_coast_frames,
            )
            self.reset()
            return None

        predicted = self._kf.predict()
        est = (int(predicted[0, 0]), int(predicted[1, 0]))
        self._last_estimate = est
        return est

    def reset(self) -> None:
        """
        Reset the filter state (call when the ball is definitively lost).
        The next update() call will re-seed the filter position.
        """
        self._initialized = False
        self._coast_count = 0
        self._last_estimate = None
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        logger.debug("KalmanTracker reset.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True after the first successful update()."""
        return self._initialized

    @property
    def is_coasting(self) -> bool:
        """True if we are predicting without a backing measurement."""
        return self._initialized and self._coast_count > 0

    @property
    def is_lost(self) -> bool:
        """True once coast count has exceeded max_coast_frames and filter was reset."""
        return not self._initialized and self._last_estimate is None

    @property
    def coast_count(self) -> int:
        """Number of consecutive frames without a measurement."""
        return self._coast_count

    @property
    def confidence(self) -> float:
        """
        Returns a confidence score [0.0, 1.0] for the current estimate.
        1.0 = fresh measurement, linearly decays to 0.0 as coasting increases.
        """
        if not self._initialized:
            return 0.0
        if self._coast_count == 0:
            return 1.0
        return max(0.0, 1.0 - self._coast_count / self.max_coast_frames)

    @property
    def last_estimate(self) -> Optional[Tuple[int, int]]:
        """Last smoothed or predicted position, or None if never initialized."""
        return self._last_estimate
