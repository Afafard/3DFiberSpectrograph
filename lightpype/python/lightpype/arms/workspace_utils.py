import json
import numpy as np

def load_calibration(calib_file='cube_calibration.json'):
    with open(calib_file, 'r') as f:
        data = json.load(f)

    world_origin = np.array(data["metadata"]["world_origin"])

    arms = {}
    for arm_name, arm_data in data.items():
        if arm_name in ["metadata", "world_coordinate_system", "geometric_constraints"]:
            continue

        calib_points = arm_data["calibration_points"]
        R = np.array(arm_data["rotation_matrix"])
        T = np.array(arm_data["translation_vector"])

        # Store for later use
        arms[arm_name] = {
            "R": R,
            "T": T,
            "calib_points": calib_points
        }

    return {
        "world_origin": world_origin,
        "arms": arms
    }
