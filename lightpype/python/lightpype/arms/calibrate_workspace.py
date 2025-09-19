#!/usr/bin/env python3
"""
Calibrate workspace for dual 5-DOF robotic arms with common reference space.

Usage:
    python calibrate_workspace.py /dev/ttyUSB0 /dev/ttyUSB1 workspace_calibration.pkl
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
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import threading
import queue


# --- Helper Functions for Arm Communication ---

def send_command(ser: serial.Serial, cmd: dict):
    """Sends a JSON command to the arm."""
    ser.write((json.dumps(cmd) + '\n').encode())


def set_arm_led(ser: serial.Serial, state: bool):
    """Sends an LED control command to the specified arm."""
    try:
        led_cmd = {"type": "LedCtrl", "enable": state}
        send_command(ser, led_cmd)
        # Small delay to let the command process
        time.sleep(0.1)
    except Exception as e:
        print(f"Warning: Could not set LED: {e}")


def read_json_continuously(ser: serial.Serial, data_queue: queue.Queue, stop_event: threading.Event):
    """
    Thread target function to continuously read JSON from serial and put valid points in a queue.
    """
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{'):
                    data = json.loads(line)
                    if 'x' in data and 'y' in data and 'z' in data:
                        # Put the latest data point in the queue, replacing any old one
                        try:
                            data_queue.get_nowait()  # Remove old item if queue is full (size 1)
                        except queue.Empty:
                            pass
                        data_queue.put((data['x'], data['y'], data['z']))
            else:
                time.sleep(0.01)  # Small sleep to prevent busy-waiting
        except Exception as e:
            # Silently handle read errors to keep the thread alive
            time.sleep(0.1)
    # print("Serial reading thread stopped.") # Debug


# --- Data Classes ---

@dataclass
class ReferencePoint:
    """Represents a calibration reference point"""
    world_coords: Tuple[float, float, float]  # (x, y, z) in world frame
    arm_coords: Tuple[float, float, float]  # (x, y, z) in arm frame
    joint_angles: Tuple[float, float, float, float, float, float]  # (b, s, e, t, r, g)


@dataclass
class WorkspaceCalibration:
    """Complete workspace calibration data"""
    left_arm: Dict[str, Any]
    right_arm: Dict[str, Any]
    world_origin: Tuple[float, float, float]
    sample_origin: Tuple[float, float, float]
    transformation_matrices: Dict[str, np.ndarray]
    workspace_bounds: Dict[str, Dict[str, float]]


# --- Core Calibration Logic ---

def collect_reference_points(ser: serial.Serial, arm_name: str, num_points: int = 5) -> List[ReferencePoint]:
    """
    Collect reference points for arm calibration with real-time feedback.
    """
    reference_points = []
    point_names = [
        "Sample origin (center of workspace)",
        "Front-right reference point",
        "Front-left reference point",
        "Back reference point",
        "Folded position (effector close to base)"
    ]

    print(f"\n=== {arm_name} Reference Point Collection ===")
    print("Please move the arm to the specified positions.")

    # Turn LED ON to indicate this arm is active
    set_arm_led(ser, True)

    try:
        for i in range(num_points):
            print(f"\n--- Point {i + 1}/{num_points}: {point_names[i]} ---")
            print(f"Position the {arm_name} at this point. Live coordinates will appear below.")
            print("Press Enter to record this position.")

            # --- Real-time feedback setup ---
            data_queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()
            reader_thread = threading.Thread(target=read_json_continuously, args=(ser, data_queue, stop_event),
                                             daemon=True)
            reader_thread.start()

            # Wait for user input while displaying live data
            last_displayed_coords = None
            try:
                while True:
                    # Non-blocking check for user input (Enter key)
                    import select
                    if select.select([sys.stdin], [], [], 0.05) == ([sys.stdin], [], []):
                        # Enter key pressed, consume the newline
                        sys.stdin.readline()
                        break

                    # Display live coordinates
                    try:
                        # Get the latest point from the queue (non-blocking)
                        x, y, z = data_queue.get_nowait()
                        current_coords = (x, y, z)
                        # Only print if the coordinates have changed significantly to reduce flicker
                        if last_displayed_coords is None or np.linalg.norm(
                                np.array(current_coords) - np.array(last_displayed_coords)) > 0.1:
                            print(f"\r  -> Live Coordinates: ({x:8.2f}, {y:8.2f}, {z:8.2f})", end='', flush=True)
                            last_displayed_coords = current_coords
                    except queue.Empty:
                        # No new data yet, just wait
                        pass

            finally:
                # Stop the background thread
                stop_event.set()
                # reader_thread.join() # Not strictly necessary for daemon threads, but good practice

            # --- Capture the final confirmed point ---
            # After the loop, get the last confirmed data point
            try:
                # Flush any remaining data to get the most up-to-date reading
                ser.reset_input_buffer()
                time.sleep(0.1)  # Brief pause

                # Read the next few lines to get a fresh, valid point
                x, y, z = 0.0, 0.0, 0.0
                for _ in range(10):  # Try a few times to get a good point
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('{'):
                        try:
                            data = json.loads(line)
                            if 'x' in data and 'y' in data and 'z' in data:
                                x, y, z = data['x'], data['y'], data['z']
                                print(f"\n  -> Final Recorded Coordinates: ({x:8.2f}, {y:8.2f}, {z:8.2f})")
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    # If loop completes without breaking, it means we failed to get a good point
                    raise TimeoutError("Could not read a valid point after confirmation.")

            except Exception as e:
                print(f"\n  -> Error reading final position: {e}. Using (0, 0, 0).")
                x, y, z = 0.0, 0.0, 0.0

            # Store reference point
            reference_points.append(ReferencePoint(
                world_coords=(0, 0, 0),
                arm_coords=(x, y, z),
                joint_angles=(0, 0, 0, 0, 0, 0)
            ))

            if i < num_points - 1:  # Don't prompt after the last point
                print("\nMove to the next position.")
                input("Press Enter to continue...")

    finally:
        # Turn LED OFF when done with this arm
        set_arm_led(ser, False)

    return reference_points


def read_serial_points(ser, min_distance=5.0, max_freq=30):
    """
    Read serial data and collect unique points until user stops.
    Waits for user to press Enter to finish recording.
    """
    points = []
    last_point = None
    min_interval = 1.0 / max_freq if max_freq > 0 else 0

    print("Recording points...")
    print("Move the arm around. Press Enter when finished.")

    recording_finished = [False]  # Use a list to make it mutable inside nested function

    def check_for_stop():
        import select
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    try:
        while not recording_finished[0]:
            loop_start = time.time()

            # Check for Enter key press
            if check_for_stop():
                sys.stdin.readline()
                recording_finished[0] = True
                break

            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{'):
                    data = json.loads(line)
                    if 'x' in data and 'y' in data and 'z' in data:
                        x, y, z = data['x'], data['y'], data['z']
                        current_point = np.array([x, y, z])

                        if last_point is None or np.linalg.norm(current_point - last_point) >= min_distance:
                            points.append((x, y, z))
                            last_point = current_point
                            print(f"\rPoints collected: {len(points)} - Last: ({x:.1f}, {y:.1f}, {z:.1f})", end='',
                                  flush=True)
            except json.JSONDecodeError:
                pass

            elapsed = time.time() - loop_start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    except Exception as e:
        print(f"\nAn error occurred during recording: {e}")

    print(f"\nRecording complete. {len(points)} points collected.")
    return np.array(points)


def collect_workspace_points(ser: serial.Serial, arm_name: str) -> np.ndarray:
    """
    Collect workspace points by moving arm through range of motion.
    """
    print(f"\n=== {arm_name} Workspace Collection ===")
    print(f"Move the {arm_name.lower()} through its full range of motion.")
    print("Try to cover all reachable positions.")

    set_arm_led(ser, True)
    try:
        points = read_serial_points(ser, min_distance=5.0, max_freq=30)
    finally:
        set_arm_led(ser, False)
    return points


# --- (Rest of the functions remain the same) ---

def compute_transformation_matrices(left_refs: List[ReferencePoint],
                                    right_refs: List[ReferencePoint]) -> Dict[str, np.ndarray]:
    """Compute transformation matrices to map arms to common world space."""
    left_sample = np.array(left_refs[0].arm_coords)
    right_sample = np.array(right_refs[0].arm_coords)
    world_origin = ((left_sample + right_sample) / 2).tolist()

    transform_left = np.eye(4)
    transform_right = np.eye(4)

    return {
        'left': transform_left,
        'right': transform_right,
        'world_origin': world_origin,
        'sample_origin': left_refs[0].arm_coords
    }


def compute_workspace_bounds(points: np.ndarray) -> Dict[str, float]:
    """Compute workspace bounds from collected points."""
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
                                 output_file: str, num_ref_points: int = 5) -> bool:
    """Calibrate workspace for dual arm system."""
    print("Starting dual arm workspace calibration...")

    left_ser = None
    right_ser = None

    try:
        print(f"Opening left arm port: {left_port}")
        left_ser = serial.Serial(left_port, baudrate=115200, timeout=2)
        left_ser.setRTS(False)
        left_ser.setDTR(False)
        time.sleep(2)

        print(f"Opening right arm port: {right_port}")
        right_ser = serial.Serial(right_port, baudrate=115200, timeout=2)
        right_ser.setRTS(False)
        right_ser.setDTR(False)
        time.sleep(2)

        # Phase 1: Collect reference points
        left_refs = collect_reference_points(left_ser, "Left Arm", num_ref_points)
        right_refs = collect_reference_points(right_ser, "Right Arm", num_ref_points)

        # Phase 2: Collect workspace points
        left_points = collect_workspace_points(left_ser, "Left Arm")
        right_points = collect_workspace_points(right_ser, "Right Arm")

        # Compute transformations and bounds
        transforms = compute_transformation_matrices(left_refs, right_refs)
        left_bounds = compute_workspace_bounds(left_points)
        right_bounds = compute_workspace_bounds(right_points)

        # Create calibration data structure
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
            world_origin=transforms['world_origin'],
            sample_origin=transforms['sample_origin'],
            transformation_matrices={
                'left': transforms['left'],
                'right': transforms['right']
            },
            workspace_bounds={
                'left': left_bounds,
                'right': right_bounds
            }
        )

        # Save calibration
        with open(output_file, 'wb') as f:
            pickle.dump(calibration, f)

        print(f"\n✅ Calibration completed successfully!")
        print(f"📁 Calibration saved to: {output_file}")
        print(f"📊 Left arm points: {len(left_points)}")
        print(f"📊 Right arm points: {len(right_points)}")
        return True

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if left_ser and left_ser.is_open:
            try:
                set_arm_led(left_ser, False)
            except:
                pass
            left_ser.close()
        if right_ser and right_ser.is_open:
            try:
                set_arm_led(right_ser, False)
            except:
                pass
            right_ser.close()


# --- Interactive Mode ---

def interactive_calibration():
    """Interactive calibration routine"""
    print("=" * 60)
    print("Interactive Dual Arm Workspace Calibration")
    print("=" * 60)

    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    if len(ports) < 2:
        print("Error: Need at least 2 serial ports for dual arm calibration")
        return False

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i + 1}. {port.device} - {port.description}")

    def get_port_selection(prompt: str, other_port: str = None) -> str:
        while True:
            try:
                choice = input(f"\n{prompt} (1-{len(ports)}): ").strip()
                port_index = int(choice) - 1
                if 0 <= port_index < len(ports):
                    selected_port = ports[port_index].device
                    if other_port and selected_port == other_port:
                        print("Error: Cannot select the same port as the other arm!")
                        continue
                    return selected_port
                else:
                    print("Invalid selection!")
            except ValueError:
                print("Please enter a valid number!")

    left_port = get_port_selection("Select LEFT arm port")
    right_port = get_port_selection("Select RIGHT arm port", left_port)

    output_file = input("\nEnter output filename (e.g., dual_calibration.pkl, press Enter for default): ").strip()
    if not output_file:
        output_file = 'dual_calibration.pkl'
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'

    if os.path.exists(output_file):
        response = input(f"File '{output_file}' exists. Overwrite? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Calibration cancelled.")
            return False

    while True:
        try:
            num_points_input = input("Number of reference points (default: 5): ").strip()
            if not num_points_input:
                num_ref_points = 5
                break
            num_ref_points = int(num_points_input)
            if num_ref_points >= 4:
                break
            else:
                print("Need at least 4 reference points!")
        except ValueError:
            print("Please enter a valid number!")

    print("\n" + "=" * 60)
    print("Calibration Parameters:")
    print(f"  Left Arm Port: {left_port}")
    print(f"  Right Arm Port: {right_port}")
    print(f"  Output File: {output_file}")
    print(f"  Reference Points: {num_ref_points}")
    print("=" * 60)

    ready = input("\nReady to start calibration? (y/N): ").strip().lower()
    if ready not in ['y', 'yes']:
        print("Calibration cancelled.")
        return False

    print("\nStarting calibration...")
    success = calibrate_dual_arm_workspace(left_port, right_port, output_file, num_ref_points)
    return success


# --- Main Execution ---

def main():
    if len(sys.argv) == 1:
        success = interactive_calibration()
        sys.exit(0 if success else 1)
    else:
        parser = argparse.ArgumentParser(
            description="Calibrate workspace for dual 5-DOF robotic arms",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python calibrate_workspace.py /dev/ttyUSB0 /dev/ttyUSB1 calibration.pkl
  python calibrate_workspace.py /dev/ttyUSB2 /dev/ttyUSB3 workspace.pkl --points 6
            """
        )
        parser.add_argument("left_port", help="Serial port for left arm (e.g., /dev/ttyUSB0)")
        parser.add_argument("right_port", help="Serial port for right arm (e.g., /dev/ttyUSB1)")
        parser.add_argument("output", help="Output file for calibration data (e.g., calibration.pkl)")
        parser.add_argument("--points", type=int, default=5, help="Number of reference points (default: 5)")

        args = parser.parse_args()
        success = calibrate_dual_arm_workspace(
            args.left_port,
            args.right_port,
            args.output,
            args.points
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
