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
from scipy.spatial.transform import Rotation

from select import select


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


def flush_serial_buffer(ser):
    """Flush serial buffer by reading all available data."""
    ser.reset_input_buffer()
    time.sleep(1)  # Wait for the buffer to clear

    # Read and discard any leftover data
    while ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        print(f"Discarding: {line}")  # Debugging statement

def read_serial_point(ser, timeout=5) -> Tuple[float, float, float]:
    """Read a single point from serial with a timeout."""
    start_time = time.time()
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    return data['x'], data['y'], data['z']
                except Exception as e:
                    print(f"Error parsing data: {e}")
                    pass  # Skip malformed JSON

        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout waiting for valid serial data")



def read_serial_points(ser, min_distance=5.0, max_freq=30, stop_condition=None):
    """
    Read serial data and collect unique points until stop condition.

    Args:
        ser: Serial connection
        min_distance: Minimum distance between consecutive points (mm)
        max_freq: Maximum sampling frequency (Hz)
        stop_condition: Function that returns True to stop recording
    """
    points = []
    last_point = None
    min_interval = 1.0 / max_freq if max_freq > 0 else 0

    print("Recording points...")
    print("Press Ctrl+C to stop recording\n")

    try:
        while True:
            if stop_condition and stop_condition():
                break

            loop_start = time.time()

            try:
                x, y, z = read_serial_point(ser)
                current_point = np.array([x, y, z])

                # Check distance from last point
                if last_point is None or np.linalg.norm(current_point - last_point) >= min_distance:
                    points.append((x, y, z))
                    last_point = current_point
                    print(f"\rPoints collected: {len(points)} - Last: ({x:.1f}, {y:.1f}, {z:.1f})", end='',
                          flush=True)
            except Exception:
                pass  # Skip errors

            # Enforce max frequency
            elapsed = time.time() - loop_start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    except KeyboardInterrupt:
        print("\nRecording stopped by user")

    print(f"\nRecording complete. {len(points)} points collected.")
    return np.array(points)

def collect_reference_points(ser, arm_name: str, num_points: int = 5) -> List[ReferencePoint]:
    """
    Collect reference points for arm calibration.

    Args:
        ser: Serial connection
        arm_name: Name of the arm (for prompts)
        num_points: Number of reference points to collect

    Returns:
        List of ReferencePoint objects
    """
    reference_points = []

    print(f"\n=== {arm_name} Reference Point Collection ===")
    print(f"Move the {arm_name.lower()} to {num_points} reference points in order:")
    print("1. Sample origin (center of workspace)")
    print("2. Front-right reference point")
    print("3. Front-left reference point")
    print("4. Back reference point")
    print("5. Folded position (effector close to base)")

    # Flush the serial buffer
    ser.reset_input_buffer()
    time.sleep(1)  # Wait for the buffer to clear

    for i in range(num_points):
        point_names = [
            "Sample origin",
            "Front-right reference",
            "Front-left reference",
            "Back reference",
            "Folded position"
        ]

        print(f"\nPoint {i + 1}/{num_points}: {point_names[i]}")
        print("Position the {arm_name} and press Enter when ready...".format(arm_name=arm_name))

        # Continuously display current arm position
        try:
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('{'):
                    data = json.loads(line)
                    x, y, z = data['x'], data['y'], data['z']
                    print(f"\rCurrent Coordinates: ({x:.2f}, {y:.2f}, {z:.2f})", end='', flush=True)

                    # Check if the user has pressed Enter to confirm the position
                    if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                        input()  # Consume the newline character
                        break

        except Exception as e:
            print(f"Error parsing data: {e}")
            x, y, z = (0.0, 0.0, 0.0)  # Default coordinates in case of error

        # Store reference point
        reference_points.append(ReferencePoint(
            world_coords=(0, 0, 0),  # Will be set later
            arm_coords=(x, y, z),
            joint_angles=(0, 0, 0, 0, 0, 0)  # Simplified - could read joint angles
        ))

    return reference_points

def collect_workspace_points(ser, arm_name: str) -> np.ndarray:
    """
    Collect workspace points by moving arm through range of motion.

    Args:
        ser: Serial connection
        arm_name: Name of the arm (for prompts)

    Returns:
        Array of (x, y, z) points
    """
    print(f"\n=== {arm_name} Workspace Collection ===")
    print(f"Move the {arm_name.lower()} through its full range of motion")
    print("Try to cover all reachable positions")
    print("Press Ctrl+C when done\n")

    points = read_serial_points(ser, min_distance=5.0, max_freq=30)
    return points


def compute_transformation_matrices(left_refs: List[ReferencePoint],
                                    right_refs: List[ReferencePoint]) -> Dict[str, np.ndarray]:
    """
    Compute transformation matrices to map arms to common world space.

    Args:
        left_refs: Left arm reference points
        right_refs: Right arm reference points

    Returns:
        Dictionary with transformation matrices
    """
    # Use first reference point (sample origin) to establish world coordinates
    left_sample = np.array(left_refs[0].arm_coords)
    right_sample = np.array(right_refs[0].arm_coords)

    # Compute approximate world origin (midpoint between arms at sample)
    world_origin = ((left_sample + right_sample) / 2).tolist()

    # For this simplified implementation, we'll create identity transforms
    # In practice, you'd compute actual transformation matrices using all reference points
    transform_left = np.eye(4)  # Identity transform for now
    transform_right = np.eye(4)  # Identity transform for now

    return {
        'left': transform_left,
        'right': transform_right,
        'world_origin': world_origin,
        'sample_origin': left_refs[0].arm_coords  # Using left arm sample as origin
    }


def compute_workspace_bounds(points: np.ndarray) -> Dict[str, float]:
    """
    Compute workspace bounds from collected points.

    Args:
        points: Array of (x, y, z) points

    Returns:
        Dictionary with min/max bounds
    """
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
    """
    Calibrate workspace for dual arm system.

    Args:
        left_port: Serial port for left arm
        right_port: Serial port for right arm
        output_file: Output filename
        num_ref_points: Number of reference points to collect

    Returns:
        True if successful
    """
    print("Starting dual arm workspace calibration...")

    left_ser = None
    right_ser = None

    try:
        # Open serial connections
        print(f"Opening left arm port: {left_port}")
        left_ser = serial.Serial(left_port, baudrate=115200, timeout=4)
        left_ser.setRTS(False)
        left_ser.setDTR(False)
        time.sleep(2)

        print(f"Opening right arm port: {right_port}")
        right_ser = serial.Serial(right_port, baudrate=115200, timeout=4)
        right_ser.setRTS(False)
        right_ser.setDTR(False)
        time.sleep(2)

        # Phase 1: Collect reference points
        left_refs = collect_reference_points(left_ser, "Left Arm", num_ref_points)
        right_refs = collect_reference_points(right_ser, "Right Arm", num_ref_points)

        # Phase 2: Collect workspace points
        left_points = collect_workspace_points(left_ser, "Left Arm")
        right_points = collect_workspace_points(right_ser, "Right Arm")

        # Compute transformations
        transforms = compute_transformation_matrices(left_refs, right_refs)

        # Compute workspace bounds
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
        return False
    finally:
        # Clean up serial connections
        if left_ser and left_ser.is_open:
            left_ser.close()
        if right_ser and right_ser.is_open:
            right_ser.close()


def interactive_calibration():
    """Interactive calibration routine"""
    print("=" * 60)
    print("Interactive Dual Arm Workspace Calibration")
    print("=" * 60)

    # Get serial ports
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    if len(ports) < 2:
        print("Error: Need at least 2 serial ports for dual arm calibration")
        return False

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i + 1}. {port.device} - {port.description}")

    # Select left arm port
    while True:
        try:
            choice = input(f"\nSelect LEFT arm port (1-{len(ports)}): ").strip()
            port_index = int(choice) - 1
            if 0 <= port_index < len(ports):
                left_port = ports[port_index].device
                break
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")

    # Select right arm port
    while True:
        try:
            choice = input(f"\nSelect RIGHT arm port (1-{len(ports)}): ").strip()
            port_index = int(choice) - 1
            if 0 <= port_index < len(ports) and ports[port_index].device != left_port:
                right_port = ports[port_index].device
                break
            else:
                print("Invalid selection or same as left arm!")
        except ValueError:
            print("Please enter a valid number!")

    # Get output filename
    while True:
        output_file = input("\nEnter output filename (e.g., dual_calibration.pkl, press Enter for default): ").strip()
        if not output_file:
            output_file = 'dual_calibration.pkl'
        if not output_file.endswith('.pkl'):
            output_file += '.pkl'
        break


    # Check if file exists
    if os.path.exists(output_file):
        response = input(f"File '{output_file}' exists. Overwrite? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Calibration cancelled.")
            return False

    # Get number of reference points
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


def main():
    # Check if command line arguments are provided
    if len(sys.argv) == 1:
        # No arguments provided - run interactive mode
        success = interactive_calibration()
        sys.exit(0 if success else 1)
    else:
        # Arguments provided - run command line mode
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
