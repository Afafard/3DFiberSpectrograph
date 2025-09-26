import math
from typing import List

import numpy as np
# Utility functions for coordinate transformations
def compute_gaze_angles(world_point: tuple, target_point: tuple = (0, 0, 0)) -> tuple:
    """
    Calculate roll, pitch, yaw needed to point from world_point towards target_point.

    Args:
        world_point: (x, y, z) current position
        target_point: (x, y, z) point to look at

    Returns:
        Tuple of (roll, pitch, yaw) in degrees
    """
    dx = target_point[0] - world_point[0]
    dy = target_point[1] - world_point[1]
    dz = target_point[2] - world_point[2]

    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if r == 0:
        return 0, 0, 0

    ux = dx / r
    uy = dy / r
    uz = dz / r

    pitch = math.asin(-uz)
    yaw = math.atan2(ux, uy)
    roll = 0  # Assume fixed roll for simplicity

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def world_to_arm_coords(world_point: tuple, R: List[List[float]], T: List[float]) -> list:
    """
    Transform world coordinates to arm base frame using rotation matrix and translation vector.

    Args:
        world_point: (x, y, z) in world coordinates
        R: 3x3 rotation matrix as list of lists
        T: translation vector [tx, ty, tz]

    Returns:
        (x, y, z) in arm base coordinates
    """
    wp = np.array(world_point)
    R_mat = np.array(R)
    T_vec = np.array(T)

    # Transform: arm_coords = R * (world_point - T)
    ap = R_mat @ (wp - T_vec)

    return ap.tolist()

