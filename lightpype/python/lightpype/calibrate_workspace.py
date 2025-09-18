#!/usr/bin/env python3
"""
Calibrate workspace for a single 5-DOF robotic arm.

Usage:
    python calibrate_workspace.py /dev/ttyUSB0 arm1_workspace.pkl
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


def read_serial_points(ser, duration=15, min_distance=5.0, max_freq=30):
    """
    Read serial data and collect unique points.

    Args:
        ser: Serial connection
        duration: Recording duration in seconds
        min_distance: Minimum distance between consecutive points (mm)
        max_freq: Maximum sampling frequency (Hz)
    """
    points = []
    last_point = None
    start_time = time.time()
    min_interval = 1.0 / max_freq if max_freq > 0 else 0

    print(f"Move the arm to define workspace boundaries...")
    print(f"Recording for {duration} seconds")
    print("Press Ctrl+C to stop early\n")

    try:
        while time.time() - start_time < duration:
            loop_start = time.time()

            line = ser.readline().decode('utf-8').strip()
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    x, y, z = data['x'], data['y'], data['z']
                    current_point = np.array([x, y, z])

                    # Check distance from last point
                    if last_point is None or np.linalg.norm(current_point - last_point) >= min_distance:
                        points.append((x, y, z))
                        last_point = current_point
                        print(f"\rPoints collected: {len(points)} - Last: ({x:.1f}, {y:.1f}, {z:.1f})", end='',
                              flush=True)

                except Exception as e:
                    pass  # Skip malformed JSON

            # Enforce max frequency
            elapsed = time.time() - loop_start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    except KeyboardInterrupt:
        print("\nRecording stopped by user")

    print(f"\nRecording complete. {len(points)} points collected.")
    return np.array(points)


def calibrate_arm(port, output_file, duration=15, min_distance=5.0, max_freq=30):
    """
    Calibrate single arm workspace.

    Args:
        port: Serial port (e.g., /dev/ttyUSB0)
        output_file: Output filename for workspace data
        duration: Recording duration in seconds
        min_distance: Minimum distance between points (mm)
        max_freq: Maximum sampling frequency (Hz)
    """
    print(f"Opening serial port: {port}")

    try:
        with serial.Serial(port, baudrate=115200, timeout=1) as ser:
            ser.setRTS(False)
            ser.setDTR(False)
            time.sleep(2)  # Wait for connection to stabilize

            points = read_serial_points(ser, duration, min_distance, max_freq)

    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        return False
    except Exception as e:
        print(f"Error during recording: {e}")
        return False

    if len(points) < 4:
        print("Error: Not enough unique points recorded. Need at least 4 points.")
        return False

    try:
        # Compute convex hull
        hull = ConvexHull(points)

        # Save workspace data
        workspace_data = {
            'points': points,
            'hull': hull,
            'num_points': len(points),
            'num_vertices': len(hull.vertices),
            'volume': hull.volume,
            'timestamp': time.time()
        }

        with open(output_file, "wb") as f:
            pickle.dump(workspace_data, f)

        print(f"\nWorkspace saved to {output_file}")
        print(f"  - Points recorded: {len(points)}")
        print(f"  - Hull vertices: {len(hull.vertices)}")
        print(f"  - Workspace volume: {hull.volume:.2f} mm³")
        return True

    except Exception as e:
        print(f"Error saving workspace: {e}")
        return False


def interactive_calibration():
    """
    Interactive calibration routine for when no command line arguments are provided.
    """
    print("=" * 60)
    print("Interactive Robotic Arm Workspace Calibration")
    print("=" * 60)

    # Get serial port
    print("\nAvailable serial ports:")
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("No serial ports found!")
        return False

    for i, port in enumerate(ports):
        print(f"  {i + 1}. {port.device} - {port.description}")

    while True:
        try:
            choice = input(f"\nSelect port (1-{len(ports)}): ").strip()
            port_index = int(choice) - 1
            if 0 <= port_index < len(ports):
                port = ports[port_index].device
                break
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")

    # Get output filename
    while True:
        output_file = input("\nEnter output filename (e.g., arm1_workspace.pkl): ").strip()
        if output_file:
            if not output_file.endswith('.pkl'):
                output_file += '.pkl'
            break
        print("Filename cannot be empty!")

    # Check if file exists
    if os.path.exists(output_file):
        response = input(f"File '{output_file}' exists. Overwrite? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Calibration cancelled.")
            return False

    # Get recording duration
    while True:
        try:
            duration_input = input("Recording duration in seconds (default: 15): ").strip()
            if not duration_input:
                duration = 15
                break
            duration = int(duration_input)
            if duration > 0:
                break
            else:
                print("Duration must be positive!")
        except ValueError:
            print("Please enter a valid number!")

    # Get minimum distance
    while True:
        try:
            distance_input = input("Minimum distance between points (mm, default: 5.0): ").strip()
            if not distance_input:
                min_distance = 5.0
                break
            min_distance = float(distance_input)
            if min_distance > 0:
                break
            else:
                print("Distance must be positive!")
        except ValueError:
            print("Please enter a valid number!")

    # Get maximum frequency
    while True:
        try:
            freq_input = input("Maximum sampling frequency (Hz, default: 30): ").strip()
            if not freq_input:
                max_freq = 30
                break
            max_freq = int(freq_input)
            if max_freq > 0:
                break
            else:
                print("Frequency must be positive!")
        except ValueError:
            print("Please enter a valid number!")

    print("\n" + "=" * 60)
    print("Calibration Parameters:")
    print(f"  Serial Port: {port}")
    print(f"  Output File: {output_file}")
    print(f"  Duration: {duration} seconds")
    print(f"  Minimum Distance: {min_distance} mm")
    print(f"  Maximum Frequency: {max_freq} Hz")
    print("=" * 60)

    ready = input("\nReady to start calibration? (y/N): ").strip().lower()
    if ready not in ['y', 'yes']:
        print("Calibration cancelled.")
        return False

    print("\nStarting calibration...")
    success = calibrate_arm(port, output_file, duration, min_distance, max_freq)

    if success:
        print("\n✅ Calibration completed successfully!")
        print(f"📁 Workspace saved to: {output_file}")
    else:
        print("\n❌ Calibration failed!")

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
            description="Calibrate workspace for a single 5-DOF robotic arm",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python calibrate_workspace.py /dev/ttyUSB0 arm1_workspace.pkl
  python calibrate_workspace.py /dev/ttyUSB1 arm2_workspace.pkl --duration 20 --distance 10
            """
        )

        parser.add_argument("port", help="Serial port (e.g., /dev/ttyUSB0)")
        parser.add_argument("output", help="Output file for workspace data (e.g., arm1_workspace.pkl)")
        parser.add_argument("--duration", type=int, default=15, help="Recording duration in seconds (default: 15)")
        parser.add_argument("--distance", type=float, default=5.0,
                            help="Minimum distance between points (mm, default: 5.0)")
        parser.add_argument("--freq", type=int, default=30, help="Maximum sampling frequency (Hz, default: 30)")

        args = parser.parse_args()

        success = calibrate_arm(
            args.port,
            args.output,
            duration=args.duration,
            min_distance=args.distance,
            max_freq=args.freq
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
