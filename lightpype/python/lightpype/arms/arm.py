import requests
import json
import serial
import time
from typing import Dict, Any, Union, Optional
import math


class RoArmController:
    def __init__(self, connection_type: str = "auto", http_ip: str = "192.168.1.255",
                 serial_port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        """
        Initialize the RoArm controller using T-code commands.

        Args:
            connection_type: "http", "serial", or "auto" (default)
            http_ip: IP address for HTTP connection
            serial_port: Serial port for USB connection
            baudrate: Baud rate for serial communication
        """
        self.connection_type = connection_type
        self.http_ip = http_ip
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_conn = None
        self.http_url = f"http://{http_ip}/cmd"
        self.http_headers = {"Content-Type": "application/json"}

        if connection_type == "serial":
            self._connect_serial()
        elif connection_type == "http":
            # Test HTTP connection
            try:
                response = requests.get(f"http://{http_ip}", timeout=2)
                if response.status_code != 200:
                    raise ConnectionError("HTTP connection failed")
            except requests.RequestException:
                raise ConnectionError("Could not connect to robot via HTTP")

    def _connect_serial(self):
        """Establish serial connection."""
        try:
            self.serial_conn = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=2
            )
            time.sleep(2)  # Wait for connection to establish
            # Flush input buffer
            self.serial_conn.flushInput()
        except serial.SerialException as e:
            raise ConnectionError(f"Serial connection failed: {e}")

    def _send_command_serial(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command via serial connection."""
        if not self.serial_conn or not self.serial_conn.is_open:
            self._connect_serial()

        try:
            # Flush input buffer before sending
            self.serial_conn.flushInput()

            # Send command
            json_cmd = json.dumps(command) + '\n'
            print(f"Sending command: {json_cmd.strip()}")
            self.serial_conn.write(json_cmd.encode())

            # Give time for processing
            time.sleep(0.1)

            # Read response
            response_lines = []
            start_time = time.time()

            while time.time() - start_time < 2:  # 2 second timeout
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode().strip()
                    if line:
                        response_lines.append(line)
                        # If we got a proper response, break
                        if line.startswith('{') and 'T' in line:
                            break
                time.sleep(0.01)

            if not response_lines:
                return {"status": "error", "error": {"code": 106, "message": "No response from device"}}

            response = response_lines[0]  # Take first response
            print(f"Raw response: {response}")

            # Try to parse response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"status": "success", "raw": response}

        except Exception as e:
            return {"status": "error", "error": {"code": 106, "message": f"Communication error: {str(e)}"}}

    def _send_command_http(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command via HTTP connection."""
        try:
            print(f"Sending HTTP command: {command}")
            response = requests.post(
                self.http_url,
                data=json.dumps(command),
                headers=self.http_headers,
                timeout=5
            )

            # Check if response contains valid JSON
            response_text = response.text.strip()
            print(f"Raw HTTP response: {response_text}")

            if not response_text:
                return {"status": "error", "error": {"code": 106, "message": "Empty response from device"}}

            # Try to parse as JSON first
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                return {"status": "success", "raw": response_text}

        except requests.RequestException as e:
            return {"status": "error", "error": {"code": 106, "message": f"HTTP request failed: {str(e)}"}}

    def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a T-code command to the robotic arm.

        Args:
            command: Dictionary containing the T-code command

        Returns:
            Response dictionary from the arm
        """
        if self.connection_type == "serial":
            return self._send_command_serial(command)
        elif self.connection_type == "http":
            return self._send_command_http(command)
        else:  # auto mode
            # Try HTTP first, fallback to serial
            try:
                result = self._send_command_http(command)
                if result.get("status") != "error":
                    return result
            except:
                pass

            return self._send_command_serial(command)

    # Basic Movement Commands
    def reset(self) -> Dict[str, Any]:
        """Move arm to initial position."""
        command = {"T": 100}
        return self.send_command(command)

    def move_single_joint_radians(self, joint: int, rad: float, spd: int = 0, acc: int = 10) -> Dict[str, Any]:
        """Move single joint using radians."""
        command = {
            "T": 101,
            "joint": joint,
            "rad": rad,
            "spd": spd,
            "acc": acc
        }
        return self.send_command(command)

    def move_all_joints_radians(self, base: float = 0, shoulder: float = 0, elbow: float = 1.57,
                                wrist: float = 0, roll: float = 0, hand: float = 1.57,
                                spd: int = 0, acc: int = 10) -> Dict[str, Any]:
        """Move all joints using radians."""
        command = {
            "T": 102,
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "roll": roll,
            "hand": hand,
            "spd": spd,
            "acc": acc
        }
        return self.send_command(command)

    def move_gripper_radians(self, cmd: float, spd: int = 0, acc: int = 0) -> Dict[str, Any]:
        """Move gripper/wrist joint using radians."""
        command = {
            "T": 106,
            "cmd": cmd,
            "spd": spd,
            "acc": acc
        }
        return self.send_command(command)

    def move_single_joint_degrees(self, joint: int, angle: float, spd: int = 10, acc: int = 10) -> Dict[str, Any]:
        """Move single joint using degrees."""
        command = {
            "T": 121,
            "joint": joint,
            "angle": angle,
            "spd": spd,
            "acc": acc
        }
        return self.send_command(command)

    def move_all_joints_degrees(self, b: float = 0, s: float = 0, e: float = 90,
                                t: float = 0, r: float = 0, h: float = 180,
                                spd: int = 10, acc: int = 10) -> Dict[str, Any]:
        """Move all joints using degrees."""
        command = {
            "T": 122,
            "b": b,
            "s": s,
            "e": e,
            "t": t,
            "r": r,
            "h": h,
            "spd": spd,
            "acc": acc
        }
        return self.send_command(command)

    # Cartesian Coordinate Commands
    def move_single_axis(self, axis: int, pos: float, spd: float = 0.25) -> Dict[str, Any]:
        """Move single axis (Inverse Kinematics)."""
        command = {
            "T": 103,
            "axis": axis,
            "pos": pos,
            "spd": spd
        }
        return self.send_command(command)

    def move_to_position(self, x: float, y: float, z: float,
                         t: float = 0, r: float = 0, g: float = 3.14,
                         spd: float = 0.25) -> Dict[str, Any]:
        """Move to Cartesian position (Inverse Kinematics)."""
        command = {
            "T": 104,
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "r": r,
            "g": g,
            "spd": spd
        }
        return self.send_command(command)

    def move_to_position_direct(self, x: float, y: float, z: float,
                                t: float = 0, r: float = 0, g: float = 3.14) -> Dict[str, Any]:
        """Move to Cartesian position directly (no interpolation)."""
        command = {
            "T": 1041,
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "r": r,
            "g": g
        }
        return self.send_command(command)

    # Feedback Commands
    def get_status(self) -> Dict[str, Any]:
        """Get current status including coordinates and joint angles."""
        command = {"T": 105}
        return self.send_command(command)

    # Continuous Movement Commands
    def continuous_move(self, m: int, axis: int, cmd: int, spd: int = 0) -> Dict[str, Any]:
        """Continuous movement control."""
        command = {
            "T": 123,
            "m": m,
            "axis": axis,
            "cmd": cmd,
            "spd": spd
        }
        return self.send_command(command)

    # Convenience Methods
    def move_joint(self, joint_id: int, angle: float, speed: int = 10) -> Dict[str, Any]:
        """Convenience method to move a joint by ID (1-6) to angle in degrees."""
        # Map joint IDs to the expected format
        joint_mapping = {
            1: 1,  # BASE_JOINT
            2: 2,  # SHOULDER_JOINT
            3: 3,  # ELBOW_JOINT
            4: 4,  # WRIST_JOINT
            5: 5,  # ROLL_JOINT
            6: 6  # EOAT_JOINT (gripper)
        }

        if joint_id not in joint_mapping:
            return {"status": "error", "error": {"code": 101, "message": "Invalid joint ID"}}

        return self.move_single_joint_degrees(joint_mapping[joint_id], angle, speed)

    def open_gripper(self, speed: int = 10) -> Dict[str, Any]:
        """Open gripper (decrease angle)."""
        # For gripper, decreasing angle opens it (from 180° to ~45°)
        return self.move_single_joint_degrees(6, 45, speed)

    def close_gripper(self, speed: int = 10) -> Dict[str, Any]:
        """Close gripper (increase angle to ~180°)."""
        return self.move_single_joint_degrees(6, 180, speed)

    # Connection Management
    def close(self):
        """Close serial connection if open."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage:
if __name__ == "__main__":
    # Auto-detect connection (tries HTTP first, then serial)
    with RoArmController(connection_type="serial", serial_port="/dev/ttyUSB0") as arm:
        print("=== RoArm Debug Session ===")

        # First, get current status
        print("\n1. Getting current status...")
        status = arm.get_status()
        print("Current status:", status)

        # Try to reset the arm
        print("\n2. Sending reset command...")
        reset_result = arm.reset()
        print("Reset result:", reset_result)

        # Wait for reset to complete
        print("Waiting 3 seconds for reset...")
        time.sleep(3)

        # Check status after reset
        print("\n3. Status after reset:")
        status = arm.get_status()
        print("Status:", status)

        # Try moving joint 1
        print("\n4. Moving joint 1 to 90 degrees...")
        move_result = arm.move_joint(1, 90, 20)
        print("Move result:", move_result)

        # Wait and check status
        print("Waiting 2 seconds...")
        time.sleep(2)
        status = arm.get_status()
        print("Status after move:", status)
