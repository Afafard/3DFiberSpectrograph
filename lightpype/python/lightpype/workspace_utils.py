# workspace_utils.py
"""
Utility functions for checking points against calibrated workspaces.
"""

import pickle
import numpy as np
from scipy.spatial import ConvexHull


def load_workspace(filename):
    """
    Load workspace data from file.

    Args:
        filename: Path to workspace file

    Returns:
        Dictionary with workspace data
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def is_point_in_hull(point, hull):
    """
    Check if a 3D point is inside a convex hull.

    Args:
        point: 3D point as [x, y, z]
        hull: ConvexHull object

    Returns:
        bool: True if point is inside hull
    """
    new_points = np.append(hull.points, [point], axis=0)
    try:
        new_hull = ConvexHull(new_points)
        return np.array_equal(new_hull.vertices, hull.vertices)
    except:
        return False


def is_safe_point(filename, x, y, z):
    """
    Check if a point is safe for the calibrated workspace.

    Args:
        filename: Path to workspace file
        x, y, z: Point coordinates

    Returns:
        bool: True if point is safe
    """
    try:
        workspace = load_workspace(filename)
        point = np.array([x, y, z])
        return is_point_in_hull(point, workspace['hull'])
    except Exception as e:
        print(f"Error checking point: {e}")
        return False


def batch_check_points(filename, points):
    """
    Check multiple points against workspace.

    Args:
        filename: Path to workspace file
        points: List of [x, y, z] points

    Returns:
        List of bool results
    """
    workspace = load_workspace(filename)
    results = []
    for point in points:
        point_array = np.array(point)
        safe = is_point_in_hull(point_array, workspace['hull'])
        results.append(safe)
    return results


# Convenience functions for specific arms
def is_safe_arm1(x, y, z, workspace_file="arm1_workspace.pkl"):
    """Check if point is safe for arm1"""
    return is_safe_point(workspace_file, x, y, z)


def is_safe_arm2(x, y, z, workspace_file="arm2_workspace.pkl"):
    """Check if point is safe for arm2"""
    return is_safe_point(workspace_file, x, y, z)
