import json
import numpy as np
from typing import Dict, Optional, List
from arm import RoArmController, world_to_arm_coords


class DualArmController:
    """
    Higher-level controller that manages two RoArmController instances.
    This class will handle world coordinate transformations using calibration data.
    """

    def __init__(self, config_file: str = "arm_config.json", calibration_file: str = "cube_calibration.json"):
        """
        Initialize the dual arm controller with configuration and calibration.

        Args:
            config_file: Path to arm configuration file
            calibration_file: Path to cube calibration file
        """
        self.config = self._load_config(config_file)
        self.calibration = self._load_calibration(calibration_file)

        # Create individual arm controllers
        self.left_arm = RoArmController(arm_type="left", config_file=config_file, calibration_file=calibration_file)
        self.right_arm = RoArmController(arm_type="right", config_file=config_file, calibration_file=calibration_file)

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {config_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def _load_calibration(self, calibration_file: str) -> Dict:
        """Load calibration data from JSON file."""
        try:
            with open(calibration_file, 'r') as f:
                calibration = json.load(f)
                return calibration
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration file {calibration_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in calibration file {calibration_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading calibration: {e}")

    def get_left_arm(self) -> RoArmController:
        """Get reference to left arm controller."""
        return self.left_arm

    def get_right_arm(self) -> RoArmController:
        """Get reference to right arm controller."""
        return self.right_arm

    def move_to_world_position(self, world_coords: tuple, arm_type: str = "left", speed: int = 50) -> Dict:
        """
        Move specified arm to a position in world coordinates.

        Args:
            world_coords: (x, y, z) target position in world coordinates
            arm_type: Which arm to move ("left" or "right")
            speed: Movement speed (1-100)

        Returns:
            Response from the arm
        """
        if arm_type not in ["left", "right"]:
            return {"error": "arm_type must be 'left' or 'right'"}

        # Get calibration data for the specified arm
        cal_data = self.calibration[arm_type]

        # Transform world coordinates to arm base frame
        arm_coords = world_to_arm_coords(
            world_coords,
            cal_data["rotation_matrix"],
            cal_data["translation_vector"]
        )

        # Move the arm to the transformed coordinates
        if arm_type == "left":
            return self.left_arm.coord_ctrl(arm_coords[0], arm_coords[1], arm_coords[2], speed=speed)
        else:
            return self.right_arm.coord_ctrl(arm_coords[0], arm_coords[1], arm_coords[2], speed=speed)

    def get_world_position(self, arm_type: str = "left") -> Optional[tuple]:
        """
        Get the current position of the specified arm in world coordinates.

        Args:
            arm_type: Which arm to query ("left" or "right")

        Returns:
            (x, y, z) position in world coordinates, or None if error
        """
        if arm_type not in ["left", "right"]:
            return None

        # Get current position from the arm
        if arm_type == "left":
            status = self.left_arm.get_status()
        else:
            status = self.right_arm.get_status()

        if "error" in status:
            return None

        # Extract arm coordinates from response
        arm_coords = None

        if "data" in status and "position" in status["data"]:
            arm_coords = status["data"]["position"]
        elif "x" in status and "y" in status and "z" in status:
            arm_coords = [status["x"], status["y"], status["z"]]
        elif "data" in status and isinstance(status["data"], dict):
            for key in ["position", "end_effector", "coords"]:
                if key in status["data"]:
                    arm_coords = status["data"][key]
                    break

        if not arm_coords or len(arm_coords) < 3:
            return None

        # Transform from arm coordinates to world coordinates
        cal_data = self.calibration[arm_type]

        # Inverse transformation: world_coords = R^T * arm_coords + T
        R_mat = np.array(cal_data["rotation_matrix"])
        T_vec = np.array(cal_data["translation_vector"])

        # Apply inverse transformation: world = R^T * arm + T
        arm_vec = np.array(arm_coords)
        world_vec = R_mat.T @ arm_vec + T_vec

        return tuple(world_vec.tolist())

    def synchronize_arms(self, world_position: tuple, speed: int = 50) -> Dict:
        """
        Move both arms to the same world position (useful for coordinated tasks).

        Args:
            world_position: (x, y, z) position in world coordinates
            speed: Movement speed for both arms

        Returns:
            Dictionary with responses from both arms
        """
        left_response = self.move_to_world_position(world_position, "left", speed)
        right_response = self.move_to_world_position(world_position, "right", speed)

        return {
            "left_arm": left_response,
            "right_arm": right_response
        }

    def coordinated_circle(self, center_x: float, center_y: float, z_height: float, radius: float,
                           speed: int = 30, steps: int = 16, duration_seconds: float = 5.0) -> Dict:
        """
        Execute a coordinated circle with both arms, positioned symmetrically around the center.

        Args:
            center_x: X coordinate of circle center
            center_y: Y coordinate of circle center
            z_height: Z height (constant) for the circles
            radius: Radius of each circle in mm
            speed: Movement speed (1-100)
            steps: Number of points to sample around each circle
            duration_seconds: Total time for completing one full circle

        Returns:
            Dictionary with responses from both arms
        """
        # Left arm moves on left side of center, right arm on right side
        left_center_x = center_x - radius / 2
        right_center_x = center_x + radius / 2

        # Execute circle on both arms simultaneously
        left_responses = self.left_arm.draw_circle(
            center_x=left_center_x,
            center_y=center_y,
            z_height=z_height,
            radius=radius / 2,
            speed=speed,
            steps=steps,
            duration_seconds=duration_seconds
        )

        right_responses = self.right_arm.draw_circle(
            center_x=right_center_x,
            center_y=center_y,
            z_height=z_height,
            radius=radius / 2,
            speed=speed,
            steps=steps,
            duration_seconds=duration_seconds
        )

        return {
            "left_arm": left_responses,
            "right_arm": right_responses
        }

    def coordinated_figure_8(self, center_x: float, center_y: float, z_height: float, radius: float,
                             speed: int = 25, steps: int = 16, duration_seconds: float = 8.0) -> Dict:
        """
        Execute a coordinated figure-8 with both arms, positioned symmetrically around the center.

        Args:
            center_x: X coordinate of figure-8 center
            center_y: Y coordinate of figure-8 center
            z_height: Z height (constant) for the figures
            radius: Radius of each loop in mm
            speed: Movement speed (1-100)
            steps: Number of points to sample for each loop
            duration_seconds: Total time for completing the figure-8

        Returns:
            Dictionary with responses from both arms
        """
        # Left arm moves on left side of center, right arm on right side
        left_center_x = center_x - radius / 2
        right_center_x = center_x + radius / 2

        # Execute figure-8 on both arms simultaneously
        left_responses = self.left_arm.draw_figure_8(
            center_x=left_center_x,
            center_y=center_y,
            z_height=z_height,
            radius=radius / 2,
            speed=speed,
            steps=steps,
            duration_seconds=duration_seconds
        )

        right_responses = self.right_arm.draw_figure_8(
            center_x=right_center_x,
            center_y=center_y,
            z_height=z_height,
            radius=radius / 2,
            speed=speed,
            steps=steps,
            duration_seconds=duration_seconds
        )

        return {
            "left_arm": left_responses,
            "right_arm": right_responses
        }

    def reset_both_arms(self) -> Dict:
        """
        Reset both arms to home position simultaneously.

        Returns:
            Dictionary with responses from both arms
        """
        left_response = self.left_arm.reset()
        right_response = self.right_arm.reset()

        return {
            "left_arm": left_response,
            "right_arm": right_response
        }

    def led_control_both(self, brightness: int = 80, pattern: str = "solid") -> Dict:
        """
        Control LED on both arms simultaneously.

        Args:
            brightness: LED brightness (0-100)
            pattern: Light pattern ("solid", "blink", "pulse", "off")

        Returns:
            Dictionary with responses from both arms
        """
        left_response = self.left_arm.led_ctrl(brightness=brightness, pattern=pattern)
        right_response = self.right_arm.led_ctrl(brightness=brightness, pattern=pattern)

        return {
            "left_arm": left_response,
            "right_arm": right_response
        }

    def get_both_statuses(self) -> Dict:
        """
        Get status from both arms simultaneously.

        Returns:
            Dictionary with statuses from both arms
        """
        left_status = self.left_arm.get_status()
        right_status = self.right_arm.get_status()

        return {
            "left_arm": left_status,
            "right_arm": right_status
        }