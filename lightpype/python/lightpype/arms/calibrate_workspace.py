#!/usr/bin/env python3
"""
Advanced Dual Arm Calibration for Cube Spectroscopy Setup

This script calibrates two RoArm-M3 Pro robotic arms mounted on opposite sides of an 0.8m cube,
facing inward, to establish a shared world coordinate system for spectral reflectance measurements.
The arms operate on a sample platform centered on a turntable at the cube's center.

Key geometric constraints:
- Cube: 0.8m x 0.8m x 0.8m
- Arms mounted on horizontal frame beams ~15cm above floor, halfway along 0.8m sides
- Sample turntable center at cube center (0,0,0) in world coordinates
- Turntable surface: ~1" thick (~2.54cm), so sample plane is at Z ≈ 0.15m + 0.0254m = ~0.175m
- Arms are 5DOF with ~1m horizontal reach, mounted at same height (Z_arm ≈ 0.15m)
- Arms face inward along X-axis; Y-axis is left/right from arm perspective

Calibration goal: Map each arm's local coordinate system to a shared world frame where:
- Origin (0,0,0) = center of sample on turntable
- Z-axis: vertical upward (positive)
- X-axis: forward from each arm toward cube center
- Y-axis: left/right perpendicular to X (right-handed system)

We'll use the 9 reference points per arm to compute a transformation from each arm's local frame to world.
We assume the arms are rigidly mounted, so we can use point correspondences to compute a 3D rigid transformation.

We'll collect:
- Arm's end-effector coordinates (from T:1051) for each reference point
- World coordinates of each reference point (predefined geometrically)

Then we solve: Minimize || W_i - (R * A_i + T) ||^2 for each point i, where:
  W_i = world coordinates of reference point i
  A_i = arm's measured end-effector position for that point
  R = rotation matrix (3x3)
  T = translation vector (3x1)

We'll use Kabsch algorithm for optimal rigid transformation.

After calibration, we save:
- World origin (0,0,0) = sample center
- Transformation matrices for left and right arms
- Joint limits (from config)
- DEFA enabled by default for position maintenance

Note: The serial communication uses T:1051 (real-time status) and T-codes for control.
We now use the exact protocol observed in working example: T:1051 streams position data.
Control commands use T:210 (torque), T:112 (DEFA), T:110 (CoordCtrl), T:101 (AngleCtrl).
"""

import serial
import json
import time
import numpy as np
from scipy.spatial import ConvexHull
import pickle
import argparse
import sys
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading
import queue
import select

# --- CONFIGURATION ---

BAUDRATE = 115200
TIMEOUT = 3.0  # Increased for robustness
COMMAND_TIMEOUT = 5.0  # Increased from 3.0 to 5.0
RESPONSE_WAIT = 0.1

# Joint limits (radians) — from AngleCtrl docs
JOINT_LIMITS = {
    1: (-np.pi, np.pi),  # Base — 360°
    2: (-np.pi / 2, np.pi / 2),  # Shoulder — 180°
    3: (-np.pi / 2, np.pi),  # Elbow — varies by design (180° to 360° range)
    4: (-np.pi / 2, np.pi / 2),  # Wrist1
    5: (-np.pi, np.pi),  # Wrist2 — 360°
    6: (1.08, np.pi),  # Gripper — 135° open to closed
}

# Workspace bounds (mm) — from COORDCTRL docs, adjusted for 1m reach
WORKSPACE_BOUNDS = {
    'x': (-500, 500),  # Extended to match ~1m reach
    'y': (-500, 500),
    'z': (0, 300)  # Z from floor to ~30cm
}

# DEFA recovery settings (using T-codes now, not DefaCtrl)
DEFA_RECOVERY_DELAY = 2.0  # Seconds after disturbance before auto-recovery (if DEFA is on)
DEFA_SENSITIVITY = 50  # Default sensitivity (1-100)

# Point collection
MIN_POINT_DISTANCE_MM = 5.0
MAX_RECORDING_FREQ_HZ = 30

# Reference points — in WORLD coordinates (mm) relative to cube center (0,0,0)
# All positions are in millimeters
WORLD_REFERENCE_POINTS = {
    "sample_center": (0, 0, 175),  # Sample center on turntable: Z = 15cm (arm height) + 2.54cm (turntable thickness)
    "left_arm_bottom_left": (-400, -400, 150),  # Arm mounting plane bottom inside corner (left) - for left arm
    "left_arm_bottom_right": (-400, 400, 150),  # Arm mounting plane bottom inside corner (right) - for left arm
    "left_arm_top_beam_corner": (-400, 0, 800),  # Above arm box support beam crossing inside corner (Z=0.8m)
    "turntable_edge_left": (0, -150, 175),  # Turntable edge left: Y=-150mm (radius=150mm)
    "turntable_edge_right": (0, 150, 175),  # Turntable edge right
    "cross_beam_left": (-400, 0, 800),  # Cross beam halfway mark left (top beam) — same as top corner for this arm
    "cross_beam_right": (400, 0, 800),  # Cross beam halfway mark right (top beam) — opposite arm's side
    "folded_position": (0, 0, 150)  # Folded position: directly below arm base (X=0,Y=0,Z=arm_height)
}

# For the right arm, we use the same world points but note:
# The right arm is mounted at (400, 0, 150) facing left (-X)
# So when the right arm moves to "sample_center", it's still (0,0,175)
# The "left" and "right" directions for the right arm are mirrored.

# We'll collect 9 points per arm. The naming is consistent with world geometry.
REFERENCE_POINT_NAMES = [
    "Sample center on turntable",
    "Arm mounting plane bottom inside corner (left)",
    "Arm mounting plane bottom inside corner (right)",
    "Above arm box support beam crossing inside corner",
    "Turntable edge left",
    "Turntable edge right",
    "Cross beam halfway mark (left side)",  # For left arm: this is (-400,0,800); for right arm it's the same point
    "Cross beam halfway mark (right side)",  # For left arm: this is (400,0,800); for right arm it's the same point
    "Folded position (effector close to base)"
]


# --- COMMUNICATION LAYER — Robust T-Code Command System ---

def send_command(ser: serial.Serial, cmd: dict) -> Optional[dict]:
    """Send JSON command and wait for response. Returns parsed response or None."""
    try:
        cmd_str = json.dumps(cmd) + '\n'
        ser.write(cmd_str.encode('utf-8'))
        time.sleep(0.1)  # Slight delay after write

        start_time = time.time()
        while time.time() - start_time < COMMAND_TIMEOUT:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        resp = json.loads(line)
                        return resp
                    except json.JSONDecodeError:
                        continue
            time.sleep(0.01)
        print(f"⚠️  Timeout waiting for: {cmd}")
        return None
    except Exception as e:
        print(f"❌ Command error: {e}")
        return None


def send_torque_ctrl(ser: serial.Serial, enable: bool) -> bool:
    """Send TorqueCtrl using T-code protocol (T:210)."""
    cmd = {"T": 210, "cmd": 0 if not enable else 1}
    print(f"🔧 Torque {'OFF' if not enable else 'ON'}")
    resp = send_command(ser, cmd)
    if not resp:
        print(f"❌ TorqueCtrl failed: No response")
        return False
    if "status" in resp and resp["status"] == "success":
        print(f"✅ Torque state confirmed: {'ON' if enable else 'OFF'}")
        return True
    print(f"❌ TorqueCtrl failed: {resp}")
    return False


def send_defa_ctrl(ser: serial.Serial, enable: bool) -> bool:
    """Send DEFA (Dynamic External Force Adaptive Control) using T-code protocol (T:112)."""
    sensitivity = DEFA_SENSITIVITY
    cmd = {
        "T": 112,
        "mode": 1 if enable else 0,
        "b": sensitivity,  # base
        "s": sensitivity,  # shoulder
        "e": sensitivity,  # elbow
        "t": sensitivity,  # wrist1
        "r": sensitivity,  # wrist2
        "h": sensitivity  # gripper (as per observed T:1051 output)
    }
    print(f"🧠 DEFA {'ON' if enable else 'OFF'} (sensitivity: {sensitivity})")
    resp = send_command(ser, cmd)
    if not resp:
        print(f"❌ DEFA failed: No response")
        return False
    if "status" in resp and resp["status"] == "success":
        print(f"✅ DEFA state confirmed: {'ON' if enable else 'OFF'}")
        return True
    print(f"❌ DEFA failed: {resp}")
    return False


def send_coord_ctrl(ser: serial.Serial, x: float, y: float, z: float, speed: int = 50) -> bool:
    """Send CoordCtrl using T-code protocol (T:110)."""
    cmd = {"T": 110, "x": int(x), "y": int(y), "z": int(z), "spd": speed}
    print(f"🎯 COORDCTRL: ({x:.1f}, {y:.1f}, {z:.1f}) @ speed {speed}")
    resp = send_command(ser, cmd)
    if not resp or "status" not in resp or resp["status"] != "success":
        print(f"❌ CoordCtrl failed: {resp}")
        return False
    print("✅ COORDCTRL accepted")
    return True


def send_angle_ctrl(ser: serial.Serial, joint_id: int, angle_rad: float, speed: int = 50) -> bool:
    """Send AngleCtrl using T-code protocol (T:101)."""
    if joint_id not in JOINT_LIMITS:
        print(f"❌ Invalid joint ID: {joint_id}")
        return False

    min_ang, max_ang = JOINT_LIMITS[joint_id]
    if not (min_ang <= angle_rad <= max_ang):
        print(f"❌ AngleCtrl joint {joint_id}: {angle_rad:.3f} rad out of bounds [{min_ang:.2f}, {max_ang:.2f}]")
        return False

    cmd = {"T": 101, "joint": joint_id, "rad": angle_rad, "spd": speed}
    print(f"🪜 ANGLECTRL: joint {joint_id} = {np.degrees(angle_rad):.1f}° ({angle_rad:.3f} rad)")
    resp = send_command(ser, cmd)
    if not resp or "status" not in resp or resp["status"] != "success":
        print(f"❌ AngleCtrl failed: {resp}")
        return False
    print("✅ ANGLECTRL accepted")
    return True


def send_get_status(ser: serial.Serial) -> Optional[dict]:
    """Fetch current status including joint angles, position, torque. Uses T:1051 streaming."""
    # We don't send a command to trigger T:1051 — it streams continuously when enabled.
    # Instead, we rely on the background thread to collect data.
    # This function is a placeholder — actual status comes from read_status_continuously
    return None


def reset_arm_state(ser: serial.Serial) -> bool:
    """Reset using official T-codes for clean state."""
    print("🔧 Resetting arm (T:603 + T:604)...")
    # Try reset boot mission
    cmd = {"T": 603}
    print(f"Sending: {cmd}")
    resp1 = send_command(ser, cmd)
    time.sleep(0.5)

    # Try clear NVS
    cmd = {"T": 604}
    print(f"Sending: {cmd}")
    resp2 = send_command(ser, cmd)

    # Wait for reset to complete
    time.sleep(3.0)

    if resp1 or resp2:
        print("✅ Reset commands sent")
        return True
    else:
        print("❌ Reset failed — no responses received")
        return False


# --- DATA STRUCTURES ---

@dataclass
class ReferencePoint:
    world_coords: Tuple[float, float, float]  # (x, y, z) in world frame
    arm_coords: Tuple[float, float, float]  # (x, y, z) in arm frame
    joint_angles: Tuple[float, float, float, float, float, float]  # (b,s,e,t,r,g) in radians
    target_orientation: Optional[Tuple[float, float, float]] = None  # Vector gripper should point to


@dataclass
class WorkspaceCalibration:
    left_arm: Dict[str, Any]
    right_arm: Dict[str, Any]
    world_origin: Tuple[float, float, float]  # (0,0,0) — sample center
    sample_origin: Tuple[float, float, float]  # Same as world origin
    transformation_matrices: Dict[str, np.ndarray]  # left and right arm to world
    workspace_bounds: Dict[str, Dict[str, float]]
    defa_enabled: bool = True
    joint_limits: Dict[int, Tuple[float, float]] = None


# --- CORE CALIBRATION LOGIC ---

def collect_reference_points(ser: serial.Serial, arm_name: str, num_points: int = 9) -> List[ReferencePoint]:
    """Collect reference points with live joint-angle feedback and DEFA for orientation fixation."""
    reference_points = []
    print(f"\n=== {arm_name} Reference Point Collection ===")

    # 🔒 SAFETY: Disable DEFA during manual positioning to avoid auto-recovery interference
    send_defa_ctrl(ser, False)
    # 🔧 Ensure torque is off for manual movement
    send_torque_ctrl(ser, False)
    set_arm_led(ser, True)

    print("⚠️  DEFA OFF — You may now manually position the arm.")
    time.sleep(1)

    # Start background thread to collect T:1051 status messages
    data_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=read_status_continuously, args=(ser, data_queue, stop_event), daemon=True)
    reader_thread.start()

    try:
        for i in range(num_points):
            print(f"\n--- Point {i + 1}/{num_points}: {REFERENCE_POINT_NAMES[i]} ---")
            print("Move the arm to this position. Live joint angles and end-effector will update below.")
            print("Press Enter to record.")

            last_displayed = None
            try:
                while True:
                    if select.select([sys.stdin], [], [], 0.05) == ([sys.stdin], [], []):
                        sys.stdin.readline()
                        break

                    try:
                        data = data_queue.get_nowait()
                        pos = data['position']
                        angles = data['angles']
                        current = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Angles: {[int(np.degrees(a)) for a in angles]}"
                        if last_displayed != current:
                            print(f"\r  -> {current}", end='', flush=True)
                            last_displayed = current
                    except queue.Empty:
                        pass

            finally:
                # We don't stop the thread here — it runs until end of collection
                pass

            # Capture final state from queue (last known good)
            try:
                data = data_queue.get(timeout=1.0)
                pos = data['position']
                angles = data['angles']
            except queue.Empty:
                print(f"\n❌ Failed to read final state. Using zeros.")
                pos = [0, 0, 0]
                angles = (0.0,) * 6

            # Record point: arm_coords is the measured position from the arm
            world_coord = WORLD_REFERENCE_POINTS[list(WORLD_REFERENCE_POINTS.keys())[i]]
            reference_points.append(ReferencePoint(
                world_coords=world_coord,
                arm_coords=tuple(pos),
                joint_angles=angles,
                target_orientation=None  # Will be computed later
            ))

            print(f"\n  -> Recorded: Pos={pos}, Angles={[int(np.degrees(a)) for a in angles]}°")

            if i < num_points - 1:
                print("\nMove to next position.")
                time.sleep(0.5)
                input("Press Enter to continue...")

    finally:
        stop_event.set()
        reader_thread.join(timeout=1.0)
        set_arm_led(ser, False)
        send_torque_ctrl(ser, False)  # Safety: torque off

    return reference_points


def read_status_continuously(ser: serial.Serial, data_queue: queue.Queue, stop_event: threading.Event):
    """Continuously read T:1051 messages from serial port and update latest state."""
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                if line.startswith('{') and line.endswith('}'):
                    try:
                        resp = json.loads(line)
                        if "T" in resp and resp["T"] == 1051:
                            # Extract position data
                            x = float(resp.get("x", 0))
                            y = float(resp.get("y", 0))
                            z = float(resp.get("z", 0))

                            # Extract joint angles from T:1051 fields
                            b = float(resp.get("b", 0))
                            s = float(resp.get("s", 0))
                            e = float(resp.get("e", 0))
                            t = float(resp.get("t", 0))
                            r = float(resp.get("r", 0))
                            g = float(resp.get("g", 0))

                            # Update latest state in queue
                            data = {
                                'position': [x, y, z],
                                'angles': (b, s, e, t, r, g)
                            }
                            try:
                                data_queue.get_nowait()
                            except queue.Empty:
                                pass
                            data_queue.put(data)
                    except (json.JSONDecodeError, ValueError) as e:
                        continue
            time.sleep(0.01)
        except Exception as e:
            print(f"⚠️ Status read error: {e}")
            time.sleep(0.1)


def collect_workspace_points(ser: serial.Serial, arm_name: str) -> np.ndarray:
    """Collect workspace points with DEFA off (manual movement)."""
    print(f"\n=== {arm_name} Workspace Collection ===")
    print("Move the arm through its full range. Press Enter to stop.")

    send_defa_ctrl(ser, False)
    send_torque_ctrl(ser, False)
    set_arm_led(ser, True)

    points = []
    last_point = None

    # Start background thread to collect T:1051 status messages
    data_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=read_status_continuously, args=(ser, data_queue, stop_event), daemon=True)
    reader_thread.start()

    try:
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                sys.stdin.readline()
                break

            try:
                data = data_queue.get_nowait()
                pos = data['position']
                current_point = np.array(pos)

                if last_point is None or np.linalg.norm(current_point - last_point) >= MIN_POINT_DISTANCE_MM:
                    points.append(pos)
                    last_point = current_point
                    print(f"\rPoints collected: {len(points)} - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                          end='', flush=True)

            except queue.Empty:
                pass

            time.sleep(0.05)
    finally:
        stop_event.set()
        reader_thread.join(timeout=1.0)
        set_arm_led(ser, False)
        send_torque_ctrl(ser, False)

    print(f"\n✅ Collected {len(points)} points.")
    return np.array(points)


def compute_transformation_matrices(left_refs: List[ReferencePoint], right_refs: List[ReferencePoint]) -> Dict[
    str, Any]:
    """Compute rigid transformation matrices from arm coordinates to world coordinates using Kabsch algorithm.

    We have 9 corresponding point pairs per arm: (arm_coords_i, world_coords_i)

    For each arm, we compute a 3D rigid transformation:
        world_point = R * arm_point + T

    Steps:
    1. Compute centroids of both point sets
    2. Subtract centroids to center the points
    3. Compute covariance matrix H = A^T * B
    4. SVD of H: H = U * S * V^T
    5. Compute rotation matrix R = V * U^T (with correction for reflection)
    6. Compute translation T = centroid_world - R * centroid_arm

    We assume the arm's end-effector position is measured at the same point (gripper tip) for all points.
    """

    # Convert lists to numpy arrays
    left_arm_points = np.array([p.arm_coords for p in left_refs])  # shape: (9,3)
    left_world_points = np.array([p.world_coords for p in left_refs])  # shape: (9,3)

    right_arm_points = np.array([p.arm_coords for p in right_refs])  # shape: (9,3)
    right_world_points = np.array([p.world_coords for p in right_refs])  # shape: (9,3)

    def kabsch_algorithm(A, B):
        """
        A: source points (arm coordinates), shape (n,3)
        B: target points (world coordinates), shape (n,3)

        Returns: R (rotation matrix 3x3), T (translation vector 3x1)
        """
        # Step 1: Compute centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # Step 2: Center the points
        A_centered = A - centroid_A
        B_centered = B - centroid_B

        # Step 3: Compute covariance matrix H
        H = A_centered.T @ B_centered

        # Step 4: SVD of H
        U, S, Vt = np.linalg.svd(H)

        # Step 5: Compute rotation matrix
        R = Vt.T @ U.T

        # Handle reflection (determinant should be +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Step 6: Compute translation
        T = centroid_B - R @ centroid_A

        return R, T

    # Compute transformations for left and right arms
    R_left, T_left = kabsch_algorithm(left_arm_points, left_world_points)
    R_right, T_right = kabsch_algorithm(right_arm_points, right_world_points)

    # Save transformation matrices
    return {
        'left': {'R': R_left, 'T': T_left},
        'right': {'R': R_right, 'T': T_right},
        'world_origin': (0.0, 0.0, 0.0),  # Sample center is origin
        'sample_origin': (0.0, 0.0, 175)  # Sample center in world (Z=175mm)
    }


def compute_workspace_bounds(points: np.ndarray) -> Dict[str, float]:
    """Compute axis-aligned bounds."""
    if len(points) == 0:
        return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_min': 0, 'z_max': 0}

    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    return {
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'z_min': float(z_min),
        'z_max': float(z_max)
    }


def calibrate_dual_arm_workspace(left_port: str, right_port: str,
                                 output_file: str, num_ref_points: int = 9) -> bool:
    """Main calibration routine with adaptive control and safety."""
    print("🚀 Starting dual arm adaptive calibration for cube spectroscopy setup...")
    left_ser = None
    right_ser = None

    try:
        print(f"Opening left arm port: {left_port}")
        left_ser = serial.Serial(left_port, BAUDRATE, timeout=TIMEOUT)
        # Configure port to avoid RTS/CTS issues
        left_ser.setRTS(False)
        left_ser.setDTR(False)

        print(f"Opening right arm port: {right_port}")
        right_ser = serial.Serial(right_port, BAUDRATE, timeout=TIMEOUT)
        right_ser.setRTS(False)
        right_ser.setDTR(False)

        time.sleep(2)

        # Reset both arms
        reset_arm_state(left_ser)
        reset_arm_state(right_ser)
        time.sleep(3)  # Increased wait after reset

        # Phase 1: Collect reference points
        print("\n" + "=" * 60)
        left_refs = collect_reference_points(left_ser, "Left Arm", num_ref_points)
        right_refs = collect_reference_points(right_ser, "Right Arm", num_ref_points)

        # Phase 2: Collect workspace points
        left_points = collect_workspace_points(left_ser, "Left Arm")
        right_points = collect_workspace_points(right_ser, "Right Arm")

        # Phase 3: Compute transformations and bounds
        transforms = compute_transformation_matrices(left_refs, right_refs)
        left_bounds = compute_workspace_bounds(left_points)
        right_bounds = compute_workspace_bounds(right_points)

        # ✅ NEW: Save DEFA state and joint limits for future use
        calibration = WorkspaceCalibration(
            left_arm={
                'reference_points': [asdict(p) for p in left_refs],
                'workspace_points': left_points.tolist(),
                'bounds': left_bounds,
                'hull': ConvexHull(left_points) if len(left_points) >= 4 else None
            },
            right_arm={
                'reference_points': [asdict(p) for p in right_refs],
                'workspace_points': right_points.tolist(),
                'bounds': right_bounds,
                'hull': ConvexHull(right_points) if len(right_points) >= 4 else None
            },
            world_origin=(0.0, 0.0, 0.0),  # Sample center is origin
            sample_origin=(0.0, 0.0, 175),  # Sample center at Z=175mm
            transformation_matrices={
                'left': transforms['left']['R'],
                'right': transforms['right']['R']
            },
            workspace_bounds={
                'left': left_bounds,
                'right': right_bounds
            },
            defa_enabled=True,  # Always enable DEFA for adaptive control
            joint_limits=JOINT_LIMITS.copy()  # Save limits for future safety checks
        )

        with open(output_file, 'wb') as f:
            pickle.dump(calibration, f)

        print(f"\n✅ ✅ ✅ Calibration completed successfully!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Left arm points: {len(left_points)}")
        print(f"📊 Right arm points: {len(right_points)}")
        print("🧠 DEFA enabled — arms will auto-recover to target orientation after disturbance!")

        # Print transformation matrices for verification
        print("\n🔧 Transformation Matrices (R):")
        print("Left Arm R:\n", transforms['left']['R'])
        print("\nLeft Arm T:", transforms['left']['T'])
        print("\nRight Arm R:\n", transforms['right']['R'])
        print("\nRight Arm T:", transforms['right']['T'])

        return True

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if left_ser and left_ser.is_open:
            send_defa_ctrl(left_ser, False)
            send_torque_ctrl(left_ser, False)
            set_arm_led(left_ser, False)
            left_ser.close()
        if right_ser and right_ser.is_open:
            send_defa_ctrl(right_ser, False)
            send_torque_ctrl(right_ser, False)
            set_arm_led(right_ser, False)
            right_ser.close()


# --- HELPER FUNCTIONS ---

def set_arm_led(ser: serial.Serial, state: bool):
    """Control LED via official command. Use T:120 for LED control."""
    cmd = {"T": 120, "brightness": 100 if state else 0}
    send_command(ser, cmd)


# --- INTERACTIVE MODE ---

def interactive_calibration():
    """Interactive mode with port auto-detection."""
    print("=" * 60)
    print("🤖 Advanced Dual Arm Adaptive Calibration for Cube Spectroscopy")
    print("=" * 60)

    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    if len(ports) < 2:
        print("❌ Need at least 2 serial ports.")
        for p in ports:
            print(f"  - {p.device} ({p.description})")
        return False

    print("\n✅ Available ports:")
    for i, p in enumerate(ports):
        print(f"  {i + 1}. {p.device} — {p.description}")

    def get_port(prompt: str, other=None):
        while True:
            try:
                choice = input(f"\n{prompt} (1-{len(ports)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(ports):
                    port = ports[idx].device
                    if other and port == other:
                        print("❌ Cannot select same port for both arms!")
                        continue
                    return port
                else:
                    print("❌ Invalid selection.")
            except ValueError:
                print("❌ Enter a number.")

    left_port = get_port("Select LEFT arm port (facing right)")
    right_port = get_port("Select RIGHT arm port (facing left)", left_port)

    output_file = input("\nOutput file (e.g., cube_calibration.pkl, Enter for default): ").strip()
    if not output_file:
        output_file = 'cube_calibration.pkl'
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'

    if os.path.exists(output_file):
        resp = input(f"⚠️  {output_file} exists. Overwrite? (y/N): ").strip().lower()
        if resp not in ['y', 'yes']:
            print("🛑 Cancelled.")
            return False

    while True:
        try:
            num_points = input("Number of reference points (7-9, default=9): ").strip()
            if not num_points:
                num_ref_points = 9
                break
            num_ref_points = int(num_points)
            if 7 <= num_ref_points <= 9:
                break
            else:
                print("❌ Must be between 7 and 9.")
        except ValueError:
            print("❌ Enter a number.")

    print("\n" + "=" * 60)
    print("⚙️  Calibration Parameters:")
    print(f"   Left Arm Port: {left_port}")
    print(f"  Right Arm Port: {right_port}")
    print(f"     Output File: {output_file}")
    print(f"  Ref Points: {num_ref_points}")
    print("🧠 DEFA enabled — arms will auto-recover to target orientation!")
    print("=" * 60)

    ready = input("\nReady? (y/N): ").strip().lower()
    if ready not in ['y', 'yes']:
        print("🛑 Cancelled.")
        return False

    print("\n🚀 Starting calibration...")
    success = calibrate_dual_arm_workspace(left_port, right_port, output_file, num_ref_points)
    return success


# --- MAIN ---

def main():
    if len(sys.argv) == 1:
        success = interactive_calibration()
        sys.exit(0 if success else 1)
    else:
        parser = argparse.ArgumentParser(
            description="Advanced dual-arm calibration for cube spectroscopy setup with T:1051 streaming",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python calibrate_workspace.py /dev/ttyUSB0 /dev/ttyUSB1 cube_calibration.pkl
  python calibrate_workspace.py --points 9

After calibration:
- Load the .pkl file to access joint limits, workspace bounds, and DEFA state.
- Use CoordCtrl or AngleCtrl with saved data to move arms while preserving orientation.
            """
        )
        parser.add_argument("left_port", help="Serial port for left arm")
        parser.add_argument("right_port", help="Serial port for right arm")
        parser.add_argument("output", help="Output .pkl file")
        parser.add_argument("--points", type=int, default=9, help="Number of reference points")

        args = parser.parse_args()
        success = calibrate_dual_arm_workspace(args.left_port, args.right_port, args.output, args.points)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
