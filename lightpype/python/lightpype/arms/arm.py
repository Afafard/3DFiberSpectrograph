import json
import time
from typing import Dict, Any, Optional, Union
from enum import Enum


class Joint(Enum):
    BASE = 1
    SHOULDER = 2
    ELBOW = 3
    WRIST = 4
    ROLL = 5
    EOAT = 6


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3
    T = 4  # Pitch
    R = 5  # Roll
    G = 6  # Gripper (angle)


class MovementMode(Enum):
    ANGLE = 0
    CARTESIAN = 1


class CommandCode(Enum):
    # Movement Control
    MOVE_INIT = 100
    SINGLE_JOINT_RAD = 101
    JOINTS_RAD_CTRL = 102
    SINGLE_AXIS_CTRL = 103
    XYZT_GOAL_CTRL = 104
    SERVO_RAD_FEEDBACK = 105
    EOAT_HAND_CTRL = 106
    SINGLE_JOINT_ANGLE = 121
    JOINTS_ANGLE_CTRL = 122
    CONSTANT_CTRL = 123

    # Delay & Misc
    DELAY_MILLIS = 111

    # EOAT Torque
    EOAT_GRAB_TORQUE = 107

    # Reboot & Reset
    REBOOT = 600
    RESET_PID = 109

    # WiFi (optional)
    WIFI_ON_BOOT = 401
    SET_AP = 402
    SET_STA = 403

    # Feedback Response (read-only)
    FEEDBACK_RESPONSE = 1051


class RoArmM3:
    """
    Python interface for RoArm-M3-S robotic arm control via JSON commands.
    Supports serial or TCP/IP transport (extendable).
    """

    def __init__(self, transport):
        """
        Initialize with a transport object (e.g., Serial or socket).
        Transport must implement: send(data: str) -> None and recv() -> str
        """
        self.transport = transport

    def _send_command(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON command and wait for response (if blocking)."""
        json_cmd = json.dumps(cmd)
        self.transport.send(json_cmd + "\n")
        time.sleep(0.05)  # Small delay for command processing

        if cmd["T"] in {CommandCode.SERVO_RAD_FEEDBACK.value}:
            # These commands return data
            response = self.transport.recv()
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON response: {response}")
                return None
        else:
            # Non-feedback commands may not respond, or response is ignored
            return None

    def reset(self):
        """Move robotic arm to initial position (CMD_MOVE_INIT)."""
        self._send_command({"T": CommandCode.MOVE_INIT.value})

    def reboot(self):
        """Reboot the robotic arm controller."""
        self._send_command({"T": CommandCode.REBOOT.value})

    def move_to_initial(self):
        """Alias for reset()."""
        self.reset()

    # --- Single Joint Control (Radian) ---
    def move_joint_rad(self, joint: Union[Joint, int], rad: float, spd: int = 0, acc: int = 10):
        """
        Move a single joint by radians.
        joint: Joint enum or int (1-6)
        rad: angle in radians
        spd: speed in steps/s (0 = max)
        acc: acceleration (0-254, unit=100 steps/s²)
        """
        if isinstance(joint, Joint):
            joint = joint.value

        # Validate joint
        if not (1 <= joint <= 6):
            raise ValueError("Joint must be between 1 and 6")

        # Validate rad range (based on manual)
        rad_ranges = {
            1: (-3.14, 3.14),     # BASE
            2: (-1.57, 1.57),     # SHOULDER
            3: (-1.11, 3.14),     # ELBOW
            4: (-1.57, 1.57),     # WRIST
            5: (-3.14, 3.14),     # ROLL
            6: (1.08, 3.14),      # EOAT
        }
        min_rad, max_rad = rad_ranges[joint]
        if not (min_rad <= rad <= max_rad):
            raise ValueError(f"rad for joint {joint} must be in [{min_rad}, {max_rad}]")

        self._send_command({
            "T": CommandCode.SINGLE_JOINT_RAD.value,
            "joint": joint,
            "rad": rad,
            "spd": spd,
            "acc": acc
        })

    # --- All Joints Control (Radian) ---
    def move_joints_rad(self, base: float = 0.0, shoulder: float = 0.0, elbow: float = 1.57,
                        wrist: float = 0.0, roll: float = 0.0, hand: float = 1.57,
                        spd: int = 0, acc: int = 10):
        """
        Move all joints simultaneously in ra
        Default positions are near home (elbow at 90°, gripper closed).
        """
        # Validate ranges
        rad_ranges = {
            "base": (-3.14, 3.14),
            "shoulder": (-1.57, 1.57),
            "elbow": (-1.11, 3.14),
            "wrist": (-1.57, 1.57),
            "roll": (-3.14, 3.14),
            "hand": (1.08, 3.14)
        }

        for name, val in [("base", base), ("shoulder", shoulder), ("elbow", elbow),
                          ("wrist", wrist), ("roll", roll), ("hand", hand)]:
            min_r, max_r = rad_ranges[name]
            if not (min_r <= val <= max_r):
                raise ValueError(f"{name} radian value {val} out of range [{min_r}, {max_r}]")

        self._send_command({
            "T": CommandCode.JOINTS_RAD_CTRL.value,
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "roll": roll,
            "hand": hand,
            "spd": spd,
            "acc": acc
        })

    # --- Gripper Only (Radian) ---
    def set_gripper_rad(self, rad: float, spd: int = 0, acc: int = 10):
        """
        Control gripper (EOAT) joint independently.
        rad: 3.14 = closed, 1.08 = fully open
        """
        if not (1.08 <= rad <= 3.14):
            raise ValueError("Gripper angle must be between 1.08 and 3.14 radians")

        self._send_command({
            "T": CommandCode.EOAT_HAND_CTRL.value,
            "cmd": rad,
            "spd": spd,
            "acc": acc
        })

    # --- Single Joint Control (Angle System) ---
    def move_joint_angle(self, joint: Union[Joint, int], angle_deg: float, spd: int = 10, acc: int = 10):
        """
        Move a single joint by degrees (angle system).
        Use this if you prefer degrees over radians.
        """
        if isinstance(joint, Joint):
            joint = joint.value

        if not (1 <= joint <= 6):
            raise ValueError("Joint must be between 1 and 6")

        angle_ranges = {
            1: (-180, 180),   # BASE
            2: (-90, 90),     # SHOULDER
            3: (-45, 180),    # ELBOW
            4: (-90, 90),     # WRIST
            5: (-180, 180),   # ROLL
            6: (45, 180),     # EOAT
        }
        min_a, max_a = angle_ranges[joint]
        if not (min_a <= angle_deg <= max_a):
            raise ValueError(f"angle for joint {joint} must be in [{min_a}, {max_a}]°")

        self._send_command({
            "T": CommandCode.SINGLE_JOINT_ANGLE.value,
            "joint": joint,
            "angle": angle_deg,
            "spd": spd,
            "acc": acc
        })

    # --- All Joints Control (Angle System) ---
    def move_joints_angle(self, b: float = 0.0, s: float = 0.0, e: float = 90.0,
                          t: float = 0.0, r: float = 0.0, h: float = 180.0,
                          spd: int = 10, acc: int = 10):
        """
        Move all joints using angle system (degrees).
        Default: home position.
        """
        ranges = {
            "b": (-180, 180),
            "s": (-90, 90),
            "e": (-45, 180),
            "t": (-90, 90),
            "r": (-180, 180),
            "h": (45, 180)
        }

        for name, val in [("b", b), ("s", s), ("e", e), ("t", t), ("r", r), ("h", h)]:
            min_a, max_a = ranges[name]
            if not (min_a <= val <= max_a):
                raise ValueError(f"{name} angle must be in [{min_a}, {max_a}]°")

        self._send_command({
            "T": CommandCode.JOINTS_ANGLE_CTRL.value,
            "b": b, "s": s, "e": e, "t": t, "r": r, "h": h,
            "spd": spd, "acc": acc
        })

    # --- Cartesian Control (Inverse Kinematics) ---
    def move_to_xyz(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                    t: float = 0.0, r: float = 0.0, g: float = 3.14,
                    spd: float = 0.25):
        """
        Move end-effector to absolute XYZT position (blocking).
        Units: mm for x,y,z; radians for t,r,g.
        """
        self._send_command({
            "T": CommandCode.XYZT_GOAL_CTRL.value,
            "x": x, "y": y, "z": z, "t": t, "r": r, "g": g, "spd": spd
        })

    def move_to_xyz_fast(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                         t: float = 0.0, r: float = 0.0, g: float = 3.14):
        """
        Non-blocking Cartesian move (fastest speed, no interpolation).
        Use for rapid sequential commands.
        """
        self._send_command({
            "T": CommandCode.XYZT_DIRECT_CTRL.value,
            "x": x, "y": y, "z": z, "t": t, "r": r, "g": g
        })

    def move_axis(self, axis: Union[Axis, int], pos: float, spd: float = 0.25):
        """
        Move a single axis (X,Y,Z,T,R,G) using inverse kinematics.
        Useful for fine adjustments.
        """
        if isinstance(axis, Axis):
            axis = axis.value
        if not (1 <= axis <= 6):
            raise ValueError("Axis must be 1-6")

        self._send_command({
            "T": CommandCode.SINGLE_AXIS_CTRL.value,
            "axis": axis,
            "pos": pos,
            "spd": spd
        })

    # --- Continuous Movement ---
    def continuous_move(self, mode: MovementMode, axis: Union[Axis, Joint, int], cmd: int, spd: int = 5):
        """
        Start continuous movement.
        mode: MovementMode.ANGLE or CARTESIAN
        axis: Joint (1-6) for angle mode, Axis (1-6) for cartesian
        cmd: 0=STOP, 1=INCREASE, 2=DECREASE
        spd: speed coefficient (0-20 recommended)
        """
        if isinstance(axis, Joint):
            axis = axis.value
        elif isinstance(axis, Axis):
            axis = axis.value

        if not (1 <= axis <= 6):
            raise ValueError("Axis must be between 1 and 6")
        if cmd not in (0, 1, 2):
            raise ValueError("cmd must be 0 (STOP), 1 (INCREASE), or 2 (DECREASE)")

        self._send_command({
            "T": CommandCode.CONSTANT_CTRL.value,
            "m": mode.value,
            "axis": axis,
            "cmd": cmd,
            "spd": spd
        })

    # --- Get Feedback ---
    def get_feedback(self) -> Optional[Dict[str, Any]]:
        """
        Request current state: joint angles, end-effector position, and load.
        Returns a dict with keys:
          x,y,z,tit,b,s,e,t,r,g (positions in radians)
          tB,tS,tE,tT,tR (loads on joints)
        """
        response = self._send_command({"T": CommandCode.SERVO_RAD_FEEDBACK.value})
        return response

    # --- Gripper Torque ---
    def set_gripper_torque(self, torque: int = 200):
        """
        Set gripper grab torque (0-255).
        Higher = stronger grip.
        """
        if not (0 <= torque <= 255):
            raise ValueError("Torque must be between 0 and 255")
        self._send_command({
            "T": CommandCode.EOAT_GRAB_TORQUE.value,
            "tor": torque
        })

    # --- Delay ---
    def delay_ms(self, ms: int):
        """
        Insert a delay in milliseconds (blocks command stream).
        Useful for synchronizing actions.
        """
        if ms < 0:
            raise ValueError("Delay must be non-negative")
        self._send_command({
            "T": CommandCode.DELAY_MILLIS.value,
            "cmd": ms
        })

    # --- Reset PID ---
    def reset_pid(self):
        """Reset all joint PID parameters to default."""
        self._send_command({"T": CommandCode.RESET_PID.value})

    # --- WiFi Commands (Optional) ---
    def set_wifi_ap(self, ssid: str = "RoArm-M3", password: str = "12345678"):
        """Set WiFi as Access Point."""
        self._send_command({
            "T": CommandCode.SET_AP.value,
            "ssid": ssid,
            "password": password
        })

    def set_wifi_sta(self, ssid: str, password: str):
        """Connect to WiFi station."""
        self._send_command({
            "T": CommandCode.SET_STA.value,
            "ssid": ssid,
            "password": password
        })

    def reboot_wifi(self):
        """Reboot WiFi module (CMD_WIFI_ON_BOOT)."""
        self._send_command({"T": CommandCode.WIFI_ON_BOOT.value, "cmd": 3})


if __name__ == '__main__':

    import serial

    class SerialTransport:
        def __init__(self, port: str, baudrate=115200):
            self.ser = serial.Serial(port, baudrate, timeout=1)

        def send(self, data: str):
            self.ser.write(data.encode('utf-8'))

        def recv(self) -> str:
            line = self.ser.readline().decode('utf-8').strip()
            return line

    # Usage:
    arm = RoArmM3(SerialTransport("/dev/ttyUSB0"))
    time.sleep(1)

    # Move to home position
    arm.reset()
    time.sleep(1)
    feedback = arm.get_feedback()

    print(feedback)
    # # Open gripper fully
    # arm.set_gripper_rad(1.08)
    #
    # # Move elbow to 90° (in radians)
    # arm.move_joint_rad(Joint.ELBOW, 1.57)
    #
    # # Move end-effector to (200, 0, 150) mm
    # arm.move_to_xyz(x=200, y=0, z=150)
    #
    # # Get current position
    # feedback = arm.get_feedback()
    # if feedback:
    #     print(f"Current X: {feedback['x']:.2f} mm, Y: {feedback['y']:.2f} mm")
    #     print(f"Gripper angle: {feedback['g']:.3f} rad")
    #
    # # Reboot
    # arm.reboot()
