#!/usr/bin/env python3
"""
Geometric Cube-Based Calibration for Dual Arm Spectroscopy Setup

This routine leverages the known cube geometry to systematically calibrate
both arms by tracing the physical boundaries they can reach, then computing
the optimal transformation to the shared world coordinate system.
"""

import serial
import json
import time
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict, Any, Optional
import queue
import select
import sys
import os
from pathlib import Path
import threading
import asyncio

# Import LED Manager with enhanced feedback
try:
    from lightpype.python.lightpype.gpio_control.led_manager import LEDManager

    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False
    print("LED Manager not available - running without LED feedback")

# Import existing utilities
try:
    from workspace_utils import save_checkpoint, load_checkpoint, delete_checkpoint

    WORKSPACE_UTILS_AVAILABLE = True
except ImportError:
    WORKSPACE_UTILS_AVAILABLE = False
    print("workspace_utils not available - using local implementations")

# --- GLOBAL LED COMMAND QUEUE ---
led_command_queue = queue.Queue()

# --- CONFIGURATION ---
BAUDRATE = 115200
TIMEOUT = 3.0
COMMAND_TIMEOUT = 5.0
DEFAULT_LEFT_PORT = "/dev/ttyUSB1"
DEFAULT_RIGHT_PORT = "/dev/ttyUSB0"
DEFAULT_OUTPUT_FILE = "cube_calibration.json"

# --- GEOMETRIC CONSTANTS ---
CUBE_SIZE = 800  # mm inner dimension
CUBE_HALF = CUBE_SIZE / 2
TURN_TABLE_HEIGHT = 57.15  # mm above cube bottom
TURN_TABLE_RADIUS = 174.625  # mm
ARM_MOUNT_HEIGHT = 165.1  # mm above cube bottom
CROSS_BEAM_HEIGHT = 342.85  # mm above cube bottom

# World coordinate system definition - now defined by shared points only
WORLD_ORIGIN = np.array([0, 0, TURN_TABLE_HEIGHT])  # Center of sample

# Remove fixed arm mounting positions - these will be determined by calibration
# LEFT_ARM_X = -350  # mm
# RIGHT_ARM_X = 317.5  # mm


# --- LOCAL IMPLEMENTATIONS FOR WORKSPACE UTILS ---
def save_checkpoint_local(output_file: str, left_refs: List[Dict] = None, right_refs: List[Dict] = None,
                          left_points: np.ndarray = None, right_points: np.ndarray = None) -> str:
    """Local implementation when workspace_utils not available"""
    checkpoint = {
        'left_refs': left_refs if left_refs is not None else None,
        'right_refs': right_refs if right_refs is not None else None,
        'left_points': left_points.tolist() if left_points is not None else None,
        'right_points': right_points.tolist() if right_points is not None else None,
        'timestamp': time.time()
    }

    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"💾 Checkpoint saved to: {checkpoint_file}")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint {checkpoint_file}: {e}")

    return checkpoint_file


def load_checkpoint_local(filename: str) -> Dict[str, Any]:
    """Local implementation when workspace_utils not available"""
    checkpoint_file = filename.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"🔍 Loading checkpoint from: {checkpoint_file}")

            # Convert lists back to numpy arrays where needed
            if checkpoint.get('left_points') is not None:
                checkpoint['left_points'] = np.array(checkpoint['left_points'])
            if checkpoint.get('right_points') is not None:
                checkpoint['right_points'] = np.array(checkpoint['right_points'])

            return checkpoint
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint {checkpoint_file}: {e}")
            return {}
    print("ℹ️  No checkpoint found")
    return {}


def delete_checkpoint_local(filename: str):
    """Local implementation when workspace_utils not available"""
    checkpoint_file = filename.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"🗑️  Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"⚠️  Failed to delete checkpoint {checkpoint_file}: {e}")


# Use appropriate functions
if WORKSPACE_UTILS_AVAILABLE:
    save_checkpoint_fn = save_checkpoint
    load_checkpoint_fn = load_checkpoint
    delete_checkpoint_fn = delete_checkpoint
else:
    save_checkpoint_fn = save_checkpoint_local
    load_checkpoint_fn = load_checkpoint_local
    delete_checkpoint_fn = delete_checkpoint_local


# --- DATA STRUCTURES ---

class CalibrationPoint:
    """Represents a single calibration measurement"""

    def __init__(self, world_coords, arm_coords, joint_angles, timestamp=None):
        self.world_coords = tuple(world_coords)
        self.arm_coords = tuple(arm_coords)
        self.joint_angles = tuple(joint_angles)
        self.timestamp = timestamp or time.time()

    def to_dict(self):
        return {
            'world_coords': self.world_coords,
            'arm_coords': self.arm_coords,
            'joint_angles': self.joint_angles,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data['world_coords'],
            data['arm_coords'],
            data['joint_angles'],
            data['timestamp']
        )


# --- ARM COMMUNICATION ---

def send_tcode_command(ser: serial.Serial, t_code: int, **params) -> Optional[Dict]:
    """Send T-code command and wait for response"""
    cmd = {"T": t_code, **params}
    try:
        cmd_str = json.dumps(cmd) + '\n'
        ser.write(cmd_str.encode('utf-8'))
        time.sleep(0.1)

        start_time = time.time()
        while time.time() - start_time < COMMAND_TIMEOUT:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            time.sleep(0.01)
        return None
    except Exception as e:
        print(f"Command error: {e}")
        return None


def read_arm_position(ser: serial.Serial) -> Optional[Tuple[List[float], List[float]]]:
    """Read current arm position and joint angles"""
    data = send_tcode_command(ser, 1051)  # Real-time status
    if data and data.get("T") == 1051:
        pos = [data.get('x', 0.0), data.get('y', 0.0), data.get('z', 0.0)]
        angles = [
            data.get('b', 0.0), data.get('s', 0.0), data.get('e', 0.0),
            data.get('t', 0.0), data.get('r', 0.0), data.get('g', 0.0)
        ]
        return pos, angles
    return None


def set_arm_torque(ser: serial.Serial, enable: bool):
    """Enable/disable arm torque"""
    send_tcode_command(ser, 210, cmd=1 if enable else 0)


def set_arm_defa(ser: serial.Serial, enable: bool):
    """Enable/disable DEFA mode"""
    if enable:
        send_tcode_command(ser, 112, mode=1, b=50, s=50, e=50, t=50, r=50, h=50)
    else:
        send_tcode_command(ser, 112, mode=0)


def read_status_continuously(ser: serial.Serial, data_queue: queue.Queue, stop_event: threading.Event):
    """Background thread to read T:1051 status messages."""
    buffer = ""
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                buffer += chunk

                lines = buffer.split('\n')
                buffer = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('{') and line.endswith('}'):
                        try:
                            data = json.loads(line)
                            if data.get("T") == 1051:
                                pos = [data.get('x', 0.0), data.get('y', 0.0), data.get('z', 0.0)]
                                angles = [
                                    data.get('b', 0.0),
                                    data.get('s', 0.0),
                                    data.get('e', 0.0),
                                    data.get('t', 0.0),
                                    data.get('r', 0.0),
                                    data.get('g', 0.0)
                                ]
                                if len(pos) == 3 and len(angles) == 6:
                                    data_queue.put({
                                        'position': pos,
                                        'angles': tuple(angles)
                                    })
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
            time.sleep(0.01)
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(0.1)


# --- GEOMETRIC CALIBRATION ROUTINES ---

def generate_boundary_points(arm_side: str) -> List[Tuple[str, np.ndarray]]:
    """
    Generate systematic boundary points based on cube geometry
    Now focused only on shared reference points that BOTH arms can reach.
    This eliminates assumptions about arm base positions and removes unreachable points.
    """
    # Only include points that BOTH arms can physically reach
    reachable_points = [
        # Turntable center - primary reference point (both arms can reach)
        ("Sample center on turntable", np.array([0, 0, TURN_TABLE_HEIGHT])),

        # Turntable edge points (front/back — accessible from both sides)
        ("Turntable front edge (Y+)", np.array([0, TURN_TABLE_RADIUS, TURN_TABLE_HEIGHT])),
        ("Turntable back edge (Y-)", np.array([0, -TURN_TABLE_RADIUS, TURN_TABLE_HEIGHT])),

        # Cross beam center points — top reference (accessible from either side)
        ("Cross beam front center", np.array([0, CUBE_HALF - 30, CROSS_BEAM_HEIGHT])),
        ("Cross beam back center", np.array([0, -(CUBE_HALF - 30), CROSS_BEAM_HEIGHT])),
    ]

    # Return the filtered list — no arm-specific logic needed, since all points are shared and reachable by both
    return reachable_points

def trace_boundary_path(ser: serial.Serial, arm_name: str, arm_side: str, boundary_points: List[Tuple[str, np.ndarray]],
                        output_file: str) -> List[CalibrationPoint]:
    """
    Guide user through tracing physical boundaries systematically with real-time feedback
    Now focused on the 7 shared reference points only.
    """
    print(f"\n{'=' * 60}")
    print(f"BINDARY TRACING FOR {arm_name.upper()}")
    print(f"{'=' * 60}")
    print("You will move the arm to trace the 7 shared reference points.")
    print("These are common locations that both arms can reach and form the world coordinate system.")
    print("At each point, position the end-effector at the target location,")
    print("then press Enter to record the measurement.")
    print("\nReal-time position feedback will be shown below.")
    print("Type 'skip' to skip a point, or 'done' to finish early.")

    calibration_points = []

    # Set up continuous position reading
    data_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=read_status_continuously,
        args=(ser, data_queue, stop_event),
        daemon=True
    )
    reader_thread.start()

    try:
        for i, (description, world_pos) in enumerate(boundary_points):
            print(f"\n--- Point {i + 1}/{len(boundary_points)}: {description} ---")
            print("*** SHARED REFERENCE POINT - This point is used for both arms' calibration ***")

            print(f"Target world position: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f}) mm")
            print("Position arm at target and press Enter to record (or 'skip'/'done'):")

            last_displayed = None
            while True:
                # Check for user input
                if select.select([sys.stdin], [], [], 0.05) == ([sys.stdin], [], []):
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == 'skip':
                        print("Point skipped.")
                        break
                    elif user_input == 'done':
                        print("Boundary tracing completed early.")
                        stop_event.set()
                        reader_thread.join(timeout=1.0)
                        return calibration_points
                    elif user_input == '':
                        # Record current position
                        arm_data = read_arm_position(ser)
                        if arm_data:
                            pos, angles = arm_data
                            cal_point = CalibrationPoint(
                                world_coords=world_pos,
                                arm_coords=pos,
                                joint_angles=angles
                            )
                            calibration_points.append(cal_point)
                            print(f"Recorded: Arm({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

                            # Save checkpoint after each point
                            save_checkpoint_fn(
                                output_file,
                                left_refs=[p.to_dict() for p in calibration_points] if arm_side == "left" else None,
                                right_refs=[p.to_dict() for p in calibration_points] if arm_side == "right" else None
                            )
                        else:
                            print("Failed to read arm position. Try again.")
                            continue
                        break
                    else:
                        print("Press Enter to record, 'skip' to skip, or 'done' to finish:")

                # Show real-time position feedback
                try:
                    data = data_queue.get_nowait()
                    pos = data['position']
                    angles = data['angles']
                    current = f"Current: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f} mm"
                    if last_displayed != current:
                        print(f"\r{current}", end='', flush=True)
                        last_displayed = current
                except queue.Empty:
                    pass

            if i < len(boundary_points) - 1:
                print("\nMove to next position...")
                time.sleep(0.5)

    finally:
        stop_event.set()
        reader_thread.join(timeout=1.0)

    return calibration_points


def compute_rigid_transform(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal rigid transformation (R, T) that minimizes ||B - (R*A + T)||²
    Using Kabsch algorithm
    """
    # Center the point sets
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    # Subtract centroids
    a_centered = points_a - centroid_a
    b_centered = points_b - centroid_b

    # Compute covariance matrix
    h = a_centered.T @ b_centered

    # SVD to find rotation
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    # Compute translation
    t = centroid_b - r @ centroid_a

    return r, t


def calibrate_arm_from_boundaries(ser: serial.Serial, arm_name: str, arm_side: str, output_file: str) -> Dict[str, Any]:
    """
    Perform complete calibration for one arm using boundary tracing
    Now uses only the 7 shared reference points to determine transformation.
    Arm base position is no longer assumed - it's calculated from the shared points.
    """
    # LED indication for which arm is being calibrated
    if LED_AVAILABLE:
        led_command_queue.put(("set_calibration_state",))

    print(f"\n{'=' * 60}")
    print(f"STARTING {arm_name.upper()} BOUNDARY-BASED CALIBRATION")
    print(f"{'=' * 60}")

    # Generate boundary points based on geometry - now only the 7 shared reference points
    boundary_points = generate_boundary_points(arm_side)
    print(f"Generated {len(boundary_points)} shared reference points for calibration")

    # Show detailed instructions
    print(f"\n{arm_name.upper()} SHARED REFERENCE POINTS:")
    for i, (desc, coords) in enumerate(boundary_points):
        print(f"  {i + 1:2d}. ✓ {desc}: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}) mm")

    # Trace boundaries
    calibration_points = trace_boundary_path(ser, arm_name, arm_side, boundary_points, output_file)

    if len(calibration_points) < 4:
        raise ValueError(f"Insufficient calibration points for {arm_name}: {len(calibration_points)}")

    # Extract arrays for transformation computation
    world_points = np.array([p.world_coords for p in calibration_points])
    arm_points = np.array([p.arm_coords for p in calibration_points])

    # Compute transformation
    rotation_matrix, translation_vector = compute_rigid_transform(arm_points, world_points)

    # Compute accuracy metrics
    transformed_points = (rotation_matrix @ arm_points.T).T + translation_vector
    errors = np.linalg.norm(transformed_points - world_points, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"\n{arm_name} Calibration Results:")
    print(f"  Points used: {len(calibration_points)}")
    print(f"  RMSE: {rmse:.3f} mm")
    print(f"  Max error: {np.max(errors):.3f} mm")

    return {
        'calibration_points': [p.to_dict() for p in calibration_points],
        'rotation_matrix': rotation_matrix.tolist(),
        'translation_vector': translation_vector.tolist(),
        'rmse': float(rmse),
        'max_error': float(np.max(errors)),
        'num_points': len(calibration_points)
    }


# --- LED COMMAND PROCESSOR ---

async def process_led_commands():
    """Async task to process LED commands."""
    if not LED_AVAILABLE:
        # Keep running but do nothing if LED not available
        while True:
            await asyncio.sleep(1)
        return

    # Initialize LED manager
    try:
        gpio_config_path = Path(__file__).parent.parent.joinpath('gpio_control', 'gpio_control.json')
        if not gpio_config_path.exists():
            gpio_config_path = Path('.').resolve().parent.joinpath('gpio_control', 'gpio_control.json')

        led_manager = LEDManager(config_file=gpio_config_path)
        led_manager.set_idle_state()
    except Exception as e:
        print(f"LED Manager initialization failed: {e}")
        # Keep running but do nothing
        while True:
            await asyncio.sleep(1)
        return

    try:
        while True:
            try:
                cmd = led_command_queue.get_nowait()
                if cmd[0] == "set_calibration_state":
                    led_manager.set_calibration_state()
                elif cmd[0] == "set_system_ready":
                    led_manager.set_system_ready()
                elif cmd[0] == "set_idle_state":
                    led_manager.set_idle_state()
                elif cmd[0] == "set_error_state":
                    led_manager.set_error_state()
                else:
                    print(f"Unknown LED command: {cmd}")
            except queue.Empty:
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"LED processing error: {e}")
    finally:
        try:
            led_manager.cleanup()
        except:
            pass


# --- MAIN CALIBRATION ROUTINE ---

async def cube_geometry_calibration_async(
        left_port: str = DEFAULT_LEFT_PORT,
        right_port: str = DEFAULT_RIGHT_PORT,
        output_file: str = DEFAULT_OUTPUT_FILE
) -> bool:
    """
    Main calibration routine using cube geometry constraints (async version)
    Now based on 7 shared reference points that eliminate assumptions about arm base positions.
    """
    print("CUBE GEOMETRY-BASED DUAL ARM CALIBRATION")
    print("=" * 60)
    print("This routine will calibrate both arms using only 7 shared reference points.")
    print("These points form a common world coordinate system that both arms can reach.")
    print("No assumptions are made about arm base positions - they will be determined by calibration.")
    print("=" * 60)

    left_ser = None
    right_ser = None

    try:
        # Open serial connections
        print(f"\nConnecting to left arm: {left_port}")
        left_ser = serial.Serial(left_port, BAUDRATE, timeout=TIMEOUT)
        left_ser.setRTS(False)
        left_ser.setDTR(False)

        print(f"Connecting to right arm: {right_port}")
        right_ser = serial.Serial(right_port, BAUDRATE, timeout=TIMEOUT)
        right_ser.setRTS(False)
        right_ser.setDTR(False)

        time.sleep(2)

        # Load checkpoint if exists
        checkpoint_file = output_file.replace('.json', '.npz')
        checkpoint = load_checkpoint_fn(checkpoint_file)
        left_results = checkpoint.get('left_refs') if checkpoint else None
        right_results = checkpoint.get('right_refs') if checkpoint else None

        # Convert checkpoint data back to CalibrationPoint objects if needed
        if left_results and isinstance(left_results[0], dict):
            left_results = [CalibrationPoint.from_dict(p) for p in left_results]

        if right_results and isinstance(right_results[0], dict):
            right_results = [CalibrationPoint.from_dict(p) for p in right_results]

        # Calibrate left arm
        if left_results is None or (isinstance(left_results, list) and len(left_results) == 0):
            print("\n--- LEFT ARM CALIBRATION ---")
            print("Instructions for LEFT ARM:")
            print("• You will trace the 7 shared reference points")
            print("• These are common locations that both arms can reach and define the world coordinate system")
            print("• The calibration will determine your arm's position relative to these points")
            set_arm_defa(left_ser, False)
            set_arm_torque(left_ser, False)
            left_results = calibrate_arm_from_boundaries(left_ser, "Left Arm", "left", output_file)
        else:
            print("Using cached left arm calibration from checkpoint")

        # Calibrate right arm
        if right_results is None or (isinstance(right_results, list) and len(right_results) == 0):
            print("\n--- RIGHT ARM CALIBRATION ---")
            print("Instructions for RIGHT ARM:")
            print("• You will trace the same 7 shared reference points")
            print("• These are common locations that both arms can reach and define the world coordinate system")
            print("• The calibration will determine your arm's position relative to these points")
            set_arm_defa(right_ser, False)
            set_arm_torque(right_ser, False)
            right_results = calibrate_arm_from_boundaries(right_ser, "Right Arm", "right", output_file)
        else:
            print("Using cached right arm calibration from checkpoint")

        # Convert results to proper format if they're dictionaries
        if isinstance(left_results, dict) and 'calibration_points' in left_results:
            left_calibration_points = [CalibrationPoint.from_dict(p) for p in left_results['calibration_points']]
        else:
            left_calibration_points = left_results if isinstance(left_results, list) else []

        if isinstance(right_results, dict) and 'calibration_points' in right_results:
            right_calibration_points = [CalibrationPoint.from_dict(p) for p in right_results['calibration_points']]
        else:
            right_calibration_points = right_results if isinstance(right_results, list) else []

        # Create final calibration data
        calibration_data = {
            "metadata": {
                "calibration_method": "cube_geometry_shared_points",
                "timestamp": time.time(),
                "world_origin": WORLD_ORIGIN.tolist()
            },
            "left_arm": {
                'calibration_points': [p.to_dict() for p in left_calibration_points] if left_calibration_points else [],
                'rotation_matrix': left_results['rotation_matrix'] if isinstance(left_results, dict) else [[1, 0, 0],
                                                                                                           [0, 1, 0],
                                                                                                           [0, 0, 1]],
                'translation_vector': left_results['translation_vector'] if isinstance(left_results, dict) else [0, 0,
                                                                                                                 0],
                'rmse': left_results['rmse'] if isinstance(left_results, dict) else 0,
                'max_error': left_results['max_error'] if isinstance(left_results, dict) else 0,
                'num_points': len(left_calibration_points) if left_calibration_points else 0
            },
            "right_arm": {
                'calibration_points': [p.to_dict() for p in
                                       right_calibration_points] if right_calibration_points else [],
                'rotation_matrix': right_results['rotation_matrix'] if isinstance(right_results, dict) else [[1, 0, 0],
                                                                                                             [0, 1, 0],
                                                                                                             [0, 0, 1]],
                'translation_vector': right_results['translation_vector'] if isinstance(right_results, dict) else [0, 0,
                                                                                                                   0],
                'rmse': right_results['rmse'] if isinstance(right_results, dict) else 0,
                'max_error': right_results['max_error'] if isinstance(right_results, dict) else 0,
                'num_points': len(right_calibration_points) if right_calibration_points else 0
            },
            "world_coordinate_system": {
                "origin": WORLD_ORIGIN.tolist(),
                "description": "Origin at sample center on turntable, defined by 7 shared reference points",
                "x_axis": "Left/Right (Left negative, Right positive)",
                "y_axis": "Front/Back (Front positive, Back negative)",
                "z_axis": "Vertical (Up positive)"
            },
            "geometric_constraints": {
                "cube_size_mm": CUBE_SIZE,
                "turntable_height_mm": TURN_TABLE_HEIGHT,
                "arm_mount_height_mm": ARM_MOUNT_HEIGHT,
                "cross_beam_height_mm": CROSS_BEAM_HEIGHT,
                "turntable_radius_mm": TURN_TABLE_RADIUS
            }
        }

        # Save calibration
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"\n{'=' * 60}")
        print("CALIBRATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(
            f"Left arm: {calibration_data['left_arm']['num_points']} points, RMSE: {calibration_data['left_arm']['rmse']:.3f} mm")
        print(
            f"Right arm: {calibration_data['right_arm']['num_points']} points, RMSE: {calibration_data['right_arm']['rmse']:.3f} mm")
        print(f"Calibration saved to: {output_file}")
        print("Note: Arm base positions are now determined by calibration, not assumed.")

        # LED indication of completion
        if LED_AVAILABLE:
            led_command_queue.put(("set_system_ready",))

        # Clean up checkpoint
        delete_checkpoint_fn(checkpoint_file)

        return True

    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        if LED_AVAILABLE:
            led_command_queue.put(("set_error_state",))
        return False

    finally:
        # Clean up serial connections
        if left_ser and left_ser.is_open:
            set_arm_defa(left_ser, False)
            set_arm_torque(left_ser, False)
            left_ser.close()
        if right_ser and right_ser.is_open:
            set_arm_defa(right_ser, False)
            set_arm_torque(right_ser, False)
            right_ser.close()

        # LED back to idle
        if LED_AVAILABLE:
            led_command_queue.put(("set_idle_state",))


def cube_geometry_calibration(
        left_port: str = DEFAULT_LEFT_PORT,
        right_port: str = DEFAULT_RIGHT_PORT,
        output_file: str = DEFAULT_OUTPUT_FILE
) -> bool:
    """
    Synchronous wrapper for the async calibration routine
    """
    # Create event loop and run the async function
    try:
        # For Python 3.7+
        return asyncio.run(cube_geometry_calibration_async(left_port, right_port, output_file))
    except AttributeError:
        # For older Python versions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(cube_geometry_calibration_async(left_port, right_port, output_file))
        finally:
            loop.close()


# --- TRANSFORMATION UTILITIES ---

def load_calibration_transforms(calibration_file: str) -> Dict[str, np.ndarray]:
    """Load transformation matrices from calibration file"""
    with open(calibration_file, 'r') as f:
        data = json.load(f)

    transforms = {}
    for arm in ['left_arm', 'right_arm']:
        arm_data = data[arm]
        transforms[arm] = {
            'rotation': np.array(arm_data['rotation_matrix']),
            'translation': np.array(arm_data['translation_vector'])
        }

    return transforms


def transform_arm_to_world(arm_coords: np.ndarray, transform: Dict[str, np.ndarray]) -> np.ndarray:
    """Transform arm coordinates to world coordinates"""
    return (transform['rotation'] @ arm_coords.T).T + transform['translation']


def transform_world_to_arm(world_coords: np.ndarray, transform: Dict[str, np.ndarray]) -> np.ndarray:
    """Transform world coordinates to arm coordinates"""
    centered = world_coords - transform['translation']
    return (transform['rotation'].T @ centered.T).T


# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    # Run with default settings for PyCharm
    print("Starting Cube Geometry-Based Dual Arm Calibration with defaults...")
    print(f"Left port: {DEFAULT_LEFT_PORT}")
    print(f"Right port: {DEFAULT_RIGHT_PORT}")
    print(f"Output file: {DEFAULT_OUTPUT_FILE}")

    # Start LED processor in background if available
    if LED_AVAILABLE:
        try:
            # Create event loop for LED processing
            led_loop = asyncio.new_event_loop()
            import threading

            led_thread = threading.Thread(target=lambda: led_loop.run_until_complete(process_led_commands()),
                                          daemon=True)
            led_thread.start()
            led_command_queue.put(("set_idle_state",))
        except Exception as e:
            print(f"Could not start LED processor: {e}")

    # Run main calibration
    success = cube_geometry_calibration(DEFAULT_LEFT_PORT, DEFAULT_RIGHT_PORT, DEFAULT_OUTPUT_FILE)

    print(f"\nCalibration {'succeeded' if success else 'failed'}")
    exit(0 if success else 1)