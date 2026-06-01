import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class StereoTriangulator:
    """
    Computes the 3D spatial coordinates (X, Y, Z) of an object (the ball)
    from its 2D coordinates tracked in two horizontally aligned (stereo) cameras.
    """
    def __init__(self, 
                 baseline_mm: float, 
                 focal_length_px: float, 
                 cx: float, 
                 cy: float) -> None:
        """
        :param baseline_mm: Distance between the left and right camera centers (in mm).
        :param focal_length_px: Focal length of the camera lenses (in pixels).
        :param cx: X coordinate of the camera principal point (usually image_width / 2).
        :param cy: Y coordinate of the camera principal point (usually image_height / 2).
        """
        self.baseline_mm: float = baseline_mm
        self.focal_length_px: float = focal_length_px
        self.cx: float = cx
        self.cy: float = cy
        
        logger.info(f"StereoTriangulator initialized with Baseline: {self.baseline_mm}mm, Focal Length: {self.focal_length_px}px")

    def triangulate(self, 
                    pt_left: Tuple[int, int], 
                    pt_right: Tuple[int, int]) -> Tuple[bool, float, float, float]:
        """
        Triangulates a 3D coordinate from left and right 2D pixel coordinates.
        The origin (0, 0, 0) is the center point between the optical centers of both cameras.
        
        - X axis: Horizontal position (left is negative, right is positive)
        - Y axis: Vertical position / Height (up is positive, down is negative)
        - Z axis: Depth / Distance (away from cameras is positive)
        
        :param pt_left: (x, y) pixel coordinates in the left camera frame
        :param pt_right: (x, y) pixel coordinates in the right camera frame
        :return: (success, X_mm, Y_mm, Z_mm)
        """
        x_left, y_left = pt_left
        x_right, y_right = pt_right
        
        # Disparity (difference in horizontal pixels)
        # Due to parallax, the object will appear further right in the left camera image.
        # Thus, x_left should be greater than x_right.
        disparity = x_left - x_right
        
        # Guard against zero or negative disparity (objects behind infinity or detection errors)
        if disparity <= 0.5:
            return False, 0.0, 0.0, 0.0
            
        # 1. Calculate Z (Depth)
        # Z = (Focal Length * Baseline) / Disparity
        z_mm = (self.focal_length_px * self.baseline_mm) / disparity
        
        # 2. Calculate X (Horizontal position relative to camera center)
        # X = Baseline * (x_left + x_right - 2*cx) / (2 * Disparity)
        x_mm = (self.baseline_mm * (x_left + x_right - 2.0 * self.cx)) / (2.0 * disparity)
        
        # 3. Calculate Y (Vertical position relative to camera center, inverted for image y-axis)
        # Y = Baseline * (2*cy - y_left - y_right) / (2 * Disparity)
        y_mm = (self.baseline_mm * (2.0 * self.cy - y_left - y_right)) / (2.0 * disparity)
        
        return True, x_mm, y_mm, z_mm

    def project_to_2d(self, x_mm: float, y_mm: float, z_mm: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Projects a 3D coordinate (X, Y, Z) back into left and right 2D camera pixels.
        Returns ((u_left, v_left), (u_right, v_right)).
        """
        if z_mm <= 0.1:
            return (0, 0), (0, 0)
            
        disparity = (self.focal_length_px * self.baseline_mm) / z_mm
        
        u_left = self.cx + (disparity / 2.0) + (x_mm * disparity / self.baseline_mm)
        u_right = self.cx - (disparity / 2.0) + (x_mm * disparity / self.baseline_mm)
        
        v = self.cy - (y_mm * disparity / self.baseline_mm)
        
        return (int(u_left), int(v)), (int(u_right), int(v))

