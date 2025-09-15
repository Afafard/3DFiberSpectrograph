import serial
import json
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import threading
from collections import deque


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


@dataclass
class CalibrationPoint:
    """Represents a calibration point with world and arm coordinates"""
    world_coords: Tuple[float, float, float]  # (x, y, z) in world frame
    arm_coords: Tuple[float, float, float]  # (x, y, z) in arm frame
    joint_angles: Tuple[float, float, float, float, float, float]  # (b, s, e, t, r, g)


class RoArmM3:
    """Control interface for RoArm-M3 robotic arm via JSON protocol"""

    def __init__(self, port: str, baudrate: int = 115200):
        """
        Initialize connection to robotic arm

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            baudrate: Baud rate for communication
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.response_queue = deque()
        self._response_lock = threading.Lock()
        self._listen_thread = None
        self._stop_listening = False

    def connect(self) -> bool:
        """Establish connection to the robotic arm"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.serial_conn.setRTS(False)
            self.serial_conn.setDTR(False)
            time.sleep(0.1)  # Allow connection to stabilize

            # Start listening thread
            self._stop_listening = False
            self._listen_thread = threading.Thread(target=self._listen_for_responses)
            self._listen_thread.daemon = True
            self._listen_thread.start()

            self.is_connected = True
            print(f"Connected to RoArm-M3 on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
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
        print(f"Disconnected from {self.port}")

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
            except:
                # Ignore errors in background thread
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

        # Clear response queue before sending command
        with self._response_lock:
            self.response_queue.clear()

        # Format command
        if isinstance(command, dict):
            cmd_str = json.dumps(command)
        else:
            cmd_str = command

        # Send command
        self.serial_conn.write((cmd_str + '\n').encode())

        if wait_for_response:
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self._response_lock:
                    if self.response_queue:
                        return self.response_queue.popleft()
                time.sleep(0.01)
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
                             tilt: float, roll: float, wait_complete: bool = True) -> bool:
        """
        Move arm to specified joint angles using AngleCtrl

        Args:
            base, shoulder, elbow, tilt, roll: Joint angles in radians
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


class DualArmController:
    """Controller for managing two RoArm-M3 arms"""

    def __init__(self, left_port: str, right_port: str):
        """
        Initialize dual arm controller

        Args:
            left_port: Serial port for left arm
            right_port: Serial port for right arm
        """
        self.left_arm = RoArmM3(left_port)
        self.right_arm = RoArmM3(right_port)
        self.calibration_data = None
        self.world_origin = None  # Origin point in world coordinates

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

        print("Starting calibration process...")
        print("Move each arm to the following positions:")

        for i, (wx, wy, wz) in enumerate(calibration_points):
            print(f"\nPoint {i + 1}/{len(calibration_points)}: World({wx:.2f}, {wy:.2f}, {wz:.2f})")
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
        print("Calibration complete!")
        return True

    def save_calibration(self, filename: str):
        """Save calibration data to file"""
        if not self.calibration_data:
            raise ValueError("No calibration data to save")

        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.calibration_data, f)
        print(f"Calibration saved to {filename}")

    def load_calibration(self, filename: str):
        """Load calibration data from file"""
        import pickle
        with open(filename, 'rb') as f:
            self.calibration_data = pickle.load(f)
        print(f"Calibration loaded from {filename}")

    def move_left_to_world(self, x: float, y: float, z: float) -> bool:
        """Move left arm to world coordinates"""
        # In a full implementation, this would transform world to arm coordinates
        # For now, we'll just pass through
        return self.left_arm.move_to_position(x, y, z)

    def move_right_to_world(self, x: float, y: float, z: float) -> bool:
        """Move right arm to world coordinates"""
        # In a full implementation, this would transform world to arm coordinates
        # For now, we'll just pass through
        return self.right_arm.move_to_position(x, y, z)

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


# Example usage
if __name__ == "__main__":
    # Example of how to use the controller
    controller = DualArmController('/dev/ttyUSB0', '/dev/ttyUSB1')

    try:
        if controller.connect():
            print("Both arms connected successfully")

            # Example: Move arms to positions
            controller.move_left_to_world(200, 50, -200)
            controller.move_right_to_world(200, -50, -200)

            # Example: Control grippers
            controller.left_arm.grasp_object()
            time.sleep(1)
            controller.left_arm.release_object()

        else:
            print("Failed to connect to one or both arms")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        controller.disconnect()