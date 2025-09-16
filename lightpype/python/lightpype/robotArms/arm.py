# roarm_control.py
"""
Robust Python module for controlling RoArm-M3 robotic arms via serial JSON protocol
"""

import serial
import json
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation
import threading
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArmState:
    """Represents the current state of the robotic arm"""
    x: float
    y: float
    z: float
    base_angle: float
    shoulder_angle: float
    elbow_angle: float
    tilt_angle: float
    roll_angle: float
    gripper_angle: float
    timestamp: float

    def to_dict(self):
        return asdict(self)


@dataclass
class CalibrationPoint:
    """Represents a calibration point with world and arm coordinates"""
    world_coords: Tuple[float, float, float]  # (x, y, z) in world frame
    arm_coords: Tuple[float, float, float]  # (x, y, z) in arm frame
    joint_angles: Tuple[float, float, float, float, float, float]  # (b, s, e, t, r, g)


class RoArmM3:
    """Control interface for RoArm-M3 robotic arm via JSON protocol"""

    def __init__(self, port: str, baudrate: int = 115200, name: str = "RoArm"):
        """
        Initialize connection to robotic arm

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            baudrate: Baud rate for communication
            name: Name for identification
        """
        self.port = port
        self.baudrate = baudrate
        self.name = name
        self.serial_conn = None
        self.is_connected = False
        self.response_queue = deque()
        self._response_lock = threading.Lock()
        self._listen_thread = None
        self._stop_listening = False
        self._command_lock = threading.Lock()

    def connect(self) -> bool:
        """Establish connection to the robotic arm"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.serial_conn.setRTS(False)
            self.serial_conn.setDTR(False)
            time.sleep(0.1)  # Allow connection to stabilize

            # Start listening thread
            self._stop_listening = False
            self._listen_thread = threading.Thread(target=self._listen_for_responses, name=f"{self.name}_listener")
            self._listen_thread.daemon = True
            self._listen_thread.start()

            self.is_connected = True
            logger.info(f"Connected to {self.name} on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close connection to the robotic arm"""
        self._stop_listening = True
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        logger.info(f"Disconnected from {self.name} on {self.port}")

    def _listen_for_responses(self):
        """Background thread to listen for responses from the arm"""
        while not self._stop_listening and self.is_connected:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        try:
                            data = json.loads(line)
                            with self._response_lock:
                                self.response_queue.append(data)
                        except json.JSONDecodeError:
                            # Non-JSON response, ignore
                            pass
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                logger.debug(f"Error in listener thread: {e}")
                pass

    def send_command(self, command: Union[dict, str], wait_for_response: bool = False,
                     timeout: float = 2.0) -> Optional[dict]:
        """
        Send command to the robotic arm

        Args:
            command: Command dictionary or JSON string
            wait_for_response: Whether to wait for a response
            timeout: Timeout in seconds

        Returns:
            Response dictionary or None
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to robotic arm")

        with self._command_lock:
            # Clear response queue before sending command
            with self._response_lock:
                self.response_queue.clear()

            # Format command
            if isinstance(command, dict):
                cmd_str = json.dumps(command)
            else:
                cmd_str = command

            # Send command
            logger.debug(f"Sending command to {self.name}: {cmd_str}")
            self.serial_conn.write((cmd_str + '\n').encode())

            if wait_for_response:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    with self._response_lock:
                        if self.response_queue:
                            response = self.response_queue.popleft()
                            logger.debug(f"Received response from {self.name}: {response}")
                            return response
                    time.sleep(0.01)
                logger.warning(f"Timeout waiting for response from {self.name}")
                return None

            return None

    def get_state(self) -> Optional[ArmState]:
        """
        Get current state of the arm

        Returns:
            Current arm state or None if failed
        """
        response = self.send_command({"T": 1051}, wait_for_response=True)
        if response:
            return ArmState(
                x=response.get('x', 0),
                y=response.get('y', 0),
                z=response.get('z', 0),
                base_angle=response.get('b', 0),
                shoulder_angle=response.get('s', 0),
                elbow_angle=response.get('e', 0),
                tilt_angle=response.get('t', 0),
                roll_angle=response.get('r', 0),
                gripper_angle=response.get('g', 0),
                timestamp=time.time()
            )
        return None

    def move_to_position(self, x: float, y: float, z: float,
                         wait_complete: bool = True, timeout: float = 10.0) -> bool:
        """
        Move arm to specified Cartesian position using COORDCTRL

        Args:
            x, y, z: Target position in mm
            wait_complete: Wait for movement to complete
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        command = {
            "T": 1052,  # COORDCTRL command
            "x": x,
            "y": y,
            "z": z
        }

        self.send_command(command)

        if wait_complete:
            start_time = time.time()
            while time.time() - start_time < timeout:
                state = self.get_state()
                if state:
                    # Check if we're close to target (within 1mm)
                    distance = np.sqrt((state.x - x) ** 2 + (state.y - y) ** 2 + (state.z - z) ** 2)
                    if distance < 1.0:
                        return True
                time.sleep(0.1)
            return False

        return True

    def move_to_joint_angles(self, base: float, shoulder: float, elbow: float,
                             tilt: float, roll: float, gripper: Optional[float] = None,
                             wait_complete: bool = True) -> bool:
        """
        Move arm to specified joint angles using AngleCtrl

        Args:
            base, shoulder, elbow, tilt, roll: Joint angles in radians
            gripper: Gripper angle (optional)
            wait_complete: Wait for movement to complete

        Returns:
            True if successful, False otherwise
        """
        command = {
            "T": 1001,  # AngleCtrl command
            "b": base,
            "s": shoulder,
            "e": elbow,
            "t": tilt,
            "r": roll
        }

        if gripper is not None:
            command["g"] = gripper

        self.send_command(command)
        return True  # For now, assume immediate execution

    def control_gripper(self, angle: float) -> bool:
        """
        Control gripper position

        Args:
            angle: Gripper angle (0 = closed, ~3.14 = open)

        Returns:
            True if successful
        """
        command = {
            "T": 1001,  # AngleCtrl command
            "g": angle
        }

        self.send_command(command)
        return True

    def grasp_object(self) -> bool:
        """Close gripper to grasp object"""
        return self.control_gripper(0.0)  # Fully closed

    def release_object(self) -> bool:
        """Open gripper to release object"""
        return self.control_gripper(3.14)  # Fully open

    def torque_lock(self, lock: bool = True) -> bool:
        """
        Enable/disable torque lock

        Args:
            lock: True to lock, False to unlock

        Returns:
            True if successful
        """
        command = {
            "T": 1043,  # Torque lock command
            "lock": 1 if lock else 0
        }

        self.send_command(command)
        return True

    def enable_defa(self, enable: bool = True) -> bool:
        """
        Enable/disable Dynamic External Force Adaptive Control

        Args:
            enable: True to enable, False to disable

        Returns:
            True if successful
        """
        command = {
            "T": 1044,  # DEFA command
            "enable": 1 if enable else 0
        }

        self.send_command(command)
        return True


class DualArmController:
    """Controller for managing two RoArm-M3 arms"""

    def __init__(self, left_port: str, right_port: str):
        """
        Initialize dual arm controller

        Args:
            left_port: Serial port for left arm
            right_port: Serial port for right arm
        """
        self.left_arm = RoArmM3(left_port, name="Left_Arm")
        self.right_arm = RoArmM3(right_port, name="Right_Arm")
        self.calibration_data = None
        self.world_origin = None  # Origin point in world coordinates
        self.transformation_matrices = {
            'left': None,
            'right': None
        }

    def connect(self) -> bool:
        """Connect to both arms"""
        left_success = self.left_arm.connect()
        right_success = self.right_arm.connect()
        return left_success and right_success

    def disconnect(self):
        """Disconnect from both arms"""
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def calibrate_arms(self, calibration_points: List[Tuple[float, float, float]]) -> bool:
        """
        Calibrate both arms to a common world coordinate system

        Args:
            calibration_points: List of world coordinates to calibrate at

        Returns:
            True if calibration successful
        """
        if len(calibration_points) < 4:
            raise ValueError("Need at least 4 calibration points")

        calibration_data = {
            'left_points': [],
            'right_points': [],
            'world_points': calibration_points.copy()
        }

        logger.info("Starting calibration process...")
        logger.info("Move each arm to the following positions:")

        for i, (wx, wy, wz) in enumerate(calibration_points):
            logger.info(f"\nPoint {i + 1}/{len(calibration_points)}: World({wx:.2f}, {wy:.2f}, {wz:.2f})")
            input("Press Enter when left arm is positioned, then move to next point...")

            # Get left arm position
            left_state = self.left_arm.get_state()
            if left_state:
                left_point = CalibrationPoint(
                    world_coords=(wx, wy, wz),
                    arm_coords=(left_state.x, left_state.y, left_state.z),
                    joint_angles=(
                        left_state.base_angle, left_state.shoulder_angle,
                        left_state.elbow_angle, left_state.tilt_angle,
                        left_state.roll_angle, left_state.gripper_angle
                    )
                )
                calibration_data['left_points'].append(left_point)

            input("Press Enter when right arm is positioned, then move to next point...")

            # Get right arm position
            right_state = self.right_arm.get_state()
            if right_state:
                right_point = CalibrationPoint(
                    world_coords=(wx, wy, wz),
                    arm_coords=(right_state.x, right_state.y, right_state.z),
                    joint_angles=(
                        right_state.base_angle, right_state.shoulder_angle,
                        right_state.elbow_angle, right_state.tilt_angle,
                        right_state.roll_angle, right_state.gripper_angle
                    )
                )
                calibration_data['right_points'].append(right_point)

        self.calibration_data = calibration_data
        self._compute_transformation_matrices()
        logger.info("Calibration complete!")
        return True

    def _compute_transformation_matrices(self):
        """Compute transformation matrices for coordinate mapping"""
        if not self.calibration_data:
            return

        # Extract points for transformation calculation
        left_arm_points = np.array([p.arm_coords for p in self.calibration_data['left_points']])
        right_arm_points = np.array([p.arm_coords for p in self.calibration_data['right_points']])
        world_points = np.array(self.calibration_data['world_points'])

        # Compute transformation matrices using least squares
        # This is a simplified approach - in practice, you might want to use more sophisticated methods
        try:
            # Add homogeneous coordinate for affine transformation
            left_arm_hom = np.hstack([left_arm_points, np.ones((left_arm_points.shape[0], 1))])
            right_arm_hom = np.hstack([right_arm_points, np.ones((right_arm_points.shape[0], 1))])
            world_hom = np.hstack([world_points, np.ones((world_points.shape[0], 1))])

            # Compute transformation matrices using pseudo-inverse
            self.transformation_matrices['left'] = np.linalg.pinv(left_arm_hom).dot(world_hom).T
            self.transformation_matrices['right'] = np.linalg.pinv(right_arm_hom).dot(world_hom).T

            logger.info("Transformation matrices computed successfully")
        except Exception as e:
            logger.error(f"Error computing transformation matrices: {e}")

    def save_calibration(self, filename: str):
        """Save calibration data to file"""
        if not self.calibration_data:
            raise ValueError("No calibration data to save")

        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'calibration_data': self.calibration_data,
                'transformation_matrices': self.transformation_matrices
            }, f)
        logger.info(f"Calibration saved to {filename}")

    def load_calibration(self, filename: str):
        """Load calibration data from file"""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.calibration_data = data['calibration_data']
            self.transformation_matrices = data['transformation_matrices']
        logger.info(f"Calibration loaded from {filename}")

    def move_left_to_world(self, x: float, y: float, z: float) -> bool:
        """Move left arm to world coordinates"""
        # Transform world coordinates to arm coordinates
        if self.transformation_matrices['left'] is not None:
            # Apply inverse transformation
            world_point = np.array([x, y, z, 1])
            arm_point = self.transformation_matrices['left'].dot(world_point)
            arm_x, arm_y, arm_z = arm_point[:3] / arm_point[3]  # Normalize homogeneous coordinate
        else:
            # If no calibration, pass through
            arm_x, arm_y, arm_z = x, y, z

        return self.left_arm.move_to_position(arm_x, arm_y, arm_z)

    def move_right_to_world(self, x: float, y: float, z: float) -> bool:
        """Move right arm to world coordinates"""
        # Transform world coordinates to arm coordinates
        if self.transformation_matrices['right'] is not None:
            # Apply inverse transformation
            world_point = np.array([x, y, z, 1])
            arm_point = self.transformation_matrices['right'].dot(world_point)
            arm_x, arm_y, arm_z = arm_point[:3] / arm_point[3]  # Normalize homogeneous coordinate
        else:
            # If no calibration, pass through
            arm_x, arm_y, arm_z = x, y, z

        return self.right_arm.move_to_position(arm_x, arm_y, arm_z)

    def move_both_to_world(self, left_pos: Tuple[float, float, float],
                           right_pos: Tuple[float, float, float]) -> bool:
        """
        Move both arms to world coordinates simultaneously

        Args:
            left_pos: (x, y, z) for left arm
            right_pos: (x, y, z) for right arm

        Returns:
            True if both movements successful
        """
        left_success = self.move_left_to_world(*left_pos)
        right_success = self.move_right_to_world(*right_pos)
        return left_success and right_success

    def grasp_with_left(self) -> bool:
        """Grasp with left arm gripper"""
        return self.left_arm.grasp_object()

    def release_with_left(self) -> bool:
        """Release with left arm gripper"""
        return self.left_arm.release_object()

    def grasp_with_right(self) -> bool:
        """Grasp with right arm gripper"""
        return self.right_arm.grasp_object()

    def release_with_right(self) -> bool:
        """Release with right arm gripper"""
        return self.right_arm.release_object()


# Scanning path planner for dual robotic arms
class ScanningPlanner:
    """Path planner for 3D scanning with dual robotic arms"""

    def __init__(self, controller: DualArmController, sample_center: Tuple[float, float, float]):
        """
        Initialize scanning planner

        Args:
            controller: Dual arm controller
            sample_center: (x, y, z) coordinates of sample center
        """
        self.controller = controller
        self.sample_center = np.array(sample_center)
        self.arm_reach = 250  # mm (approximate reach)
        self.safety_margin = 20  # mm safety margin

    def spherical_to_cartesian(self, r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian

        Args:
            r: Radial distance
            theta: Polar angle (from z-axis)
            phi: Azimuthal angle (from x-axis)

        Returns:
            (x, y, z) Cartesian coordinates
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return (x, y, z)

    def cartesian_to_spherical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            (r, theta, phi) spherical coordinates
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r) if r > 0 else 0
        phi = np.arctan2(y, x)
        return (r, theta, phi)

    def generate_scanning_positions(self,
                                    r_range: Tuple[float, float],
                                    theta_range: Tuple[float, float],
                                    phi_range: Tuple[float, float],
                                    r_steps: int = 10,
                                    theta_steps: int = 10,
                                    phi_steps: int = 10) -> List[Tuple[float, float, float]]:
        """
        Generate scanning positions in spherical coordinates

        Args:
            r_range: (min, max) radial distance
            theta_range: (min, max) polar angle (radians)
            phi_range: (min, max) azimuthal angle (radians)
            r_steps, theta_steps, phi_steps: Number of steps in each dimension

        Returns:
            List of (r, theta, phi) positions
        """
        r_min, r_max = r_range
        theta_min, theta_max = theta_range
        phi_min, phi_max = phi_range

        positions = []
        for r in np.linspace(r_min, r_max, r_steps):
            for theta in np.linspace(theta_min, theta_max, theta_steps):
                for phi in np.linspace(phi_min, phi_max, phi_steps):
                    positions.append((r, theta, phi))

        return positions

    def plan_hemispherical_scan(self,
                                r_min: float = 100, r_max: float = 200,
                                r_steps: int = 5,
                                theta_steps: int = 8,
                                phi_steps: int = 16) -> List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Plan a hemispherical scanning pattern for both arms

        Args:
            r_min, r_max: Radial distance range
            r_steps, theta_steps, phi_steps: Sampling resolution

        Returns:
            List of ((arm1_pos), (arm2_pos)) pairs
        """
        # Generate positions for upper hemisphere (theta: 0 to pi/2)
        positions = self.generate_scanning_positions(
            r_range=(r_min, r_max),
            theta_range=(0, np.pi / 2),
            phi_range=(0, 2 * np.pi),
            r_steps=r_steps,
            theta_steps=theta_steps,
            phi_steps=phi_steps
        )

        # Convert to world coordinates around sample
        world_positions = []
        for r, theta, phi in positions:
            # Convert to Cartesian relative to sample center
            dx, dy, dz = self.spherical_to_cartesian(r, theta, phi)
            world_x = self.sample_center[0] + dx
            world_y = self.sample_center[1] + dy
            world_z = self.sample_center[2] + dz
            world_positions.append((world_x, world_y, world_z))

        # For dual arm system, we'll pair positions:
        # Arm 1 covers one hemisphere, Arm 2 covers the other
        # This is a simplified approach - in reality, you'd want to optimize
        # for maximum coverage with minimal collision risk

        pairs = []
        half = len(world_positions) // 2

        # Arm 1 gets first half, Arm 2 gets second half
        for i in range(min(half, len(world_positions) - half)):
            arm1_pos = world_positions[i]
            arm2_pos = world_positions[half + i]
            pairs.append((arm1_pos, arm2_pos))

        return pairs

    def execute_scan(self, position_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]):
        """
        Execute the scanning sequence

        Args:
            position_pairs: List of ((arm1_pos), (arm2_pos)) pairs
        """
        logger.info(f"Executing scan with {len(position_pairs)} position pairs")

        for i, (arm1_pos, arm2_pos) in enumerate(position_pairs):
            logger.info(f"Moving to position pair {i + 1}/{len(position_pairs)}")
            logger.info(f"  Arm 1: ({arm1_pos[0]:.1f}, {arm1_pos[1]:.1f}, {arm1_pos[2]:.1f})")
            logger.info(f"  Arm 2: ({arm2_pos[0]:.1f}, {arm2_pos[1]:.1f}, {arm2_pos[2]:.1f})")

            # Move both arms simultaneously
            success = self.controller.move_both_to_world(arm1_pos, arm2_pos)

            if success:
                logger.info("  Movement successful")
                # Here you would trigger your spectrometer/light measurement
                # time.sleep(0.5)  # Allow time for measurement
            else:
                logger.warning("  Movement failed")

            # Add a small delay between movements
            # time.sleep(0.1)


# Example usage
if __name__ == "__main__":
    # Example of how to use the controller
    controller = DualArmController('/dev/ttyUSB0', '/dev/ttyUSB1')

    try:
        if controller.connect():
            logger.info("Both arms connected successfully")

            # Example: Move arms to positions
            controller.move_left_to_world(200, 50, -200)
            controller.move_right_to_world(200, -50, -200)

            # Example: Control grippers
            controller.left_arm.grasp_object()
            time.sleep(1)
            controller.left_arm.release_object()

        else:
            logger.error("Failed to connect to one or both arms")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        controller.disconnect()
