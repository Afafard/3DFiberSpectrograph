#!/usr/bin/env python3
"""
Real-time 3D visualization of dual robotic arms with common coordinate space.

Usage:
    python visualize_arms.py /dev/ttyUSB0 /dev/ttyUSB1 calibration.pkl
"""

import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import serial
import threading
import pickle
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import argparse


class SerialReader(QObject):
    """Thread-safe serial reader that emits data signals"""
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, port: str, parent=None):
        super().__init__(parent)
        self.port = port
        self.ser = None
        self.running = False
        self.thread = None

    def start(self):
        """Start the serial reader thread"""
        try:
            self.ser = serial.Serial(self.port, baudrate=115200, timeout=1)
            self.ser.setRTS(False)
            self.ser.setDTR(False)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            self.error_occurred.emit(f"Failed to open {self.port}: {str(e)}")

    def stop(self):
        """Stop the serial reader"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _read_loop(self):
        """Background thread to read serial data"""
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line.startswith('{'):
                        data = json.loads(line)
                        self.data_received.emit(data)
                else:
                    # Small delay to prevent excessive CPU usage
                    threading.Event().wait(0.01)
            except json.JSONDecodeError:
                pass  # Skip malformed JSON
            except Exception as e:
                self.error_occurred.emit(f"Serial error on {self.port}: {str(e)}")
                break


@dataclass
class ArmPosition:
    """Represents current arm position"""
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


class ArmVisualizer(QMainWindow):
    """3D visualization window for dual robotic arms"""

    def __init__(self, left_port: str, right_port: str, calibration_file: str):
        super().__init__()
        self.setWindowTitle("Dual Robotic Arm Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Load calibration data
        self.calibration = self._load_calibration(calibration_file)

        # Initialize serial readers
        self.left_reader = SerialReader(left_port)
        self.right_reader = SerialReader(right_port)

        # Current arm positions
        self.left_position = None
        self.right_position = None

        # Setup UI
        self._setup_ui()

        # Start serial readers
        self.left_reader.data_received.connect(self._update_left_arm)
        self.right_reader.data_received.connect(self._update_right_arm)
        self.left_reader.error_occurred.connect(self._handle_error)
        self.right_reader.error_occurred.connect(self._handle_error)

        self.left_reader.start()
        self.right_reader.start()

        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_visualization)
        self.timer.start(50)  # Update at 20 FPS

    def _load_calibration(self, filename: str) -> Dict:
        """Load calibration data from file"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
            return None

    def _setup_ui(self):
        """Setup the user interface"""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 3D visualization
        self._setup_3d_view()
        main_layout.addWidget(self.view, stretch=3)

        # Control panel
        self._setup_control_panel()
        main_layout.addWidget(self.control_panel, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _setup_3d_view(self):
        """Setup the 3D visualization view"""
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('3D Arm Visualization')
        self.view.setCameraPosition(distance=800, elevation=30, azimuth=45)

        # Add coordinate axes
        axes = gl.GLAxisItem()
        axes.setSize(x=200, y=200, z=200)
        self.view.addItem(axes)

        # Add sample origin marker
        if self.calibration:
            sample_origin = self.calibration.sample_origin
            origin_point = gl.GLScatterPlotItem(
                pos=np.array([sample_origin]),
                color=(1, 0, 0, 1),
                size=15,
                pxMode=True
            )
            self.view.addItem(origin_point)

            # Add sample coordinate system
            self._add_coordinate_system(sample_origin, size=50)

        # Add workspace bounds if available
        self._add_workspace_bounds()

        # Create arm visualization items
        self.left_arm_item = gl.GLLinePlotItem(
            color=(0, 1, 0, 1),  # Green for left arm
            width=3,
            antialias=True
        )
        self.view.addItem(self.left_arm_item)

        self.right_arm_item = gl.GLLinePlotItem(
            color=(0, 0, 1, 1),  # Blue for right arm
            width=3,
            antialias=True
        )
        self.view.addItem(self.right_arm_item)

        # Add end effectors
        self.left_ee = gl.GLScatterPlotItem(
            color=(0, 1, 0, 1),
            size=10,
            pxMode=True
        )
        self.view.addItem(self.left_ee)

        self.right_ee = gl.GLScatterPlotItem(
            color=(0, 0, 1, 1),
            size=10,
            pxMode=True
        )
        self.view.addItem(self.right_ee)

    def _setup_control_panel(self):
        """Setup the control panel"""
        self.control_panel = QFrame()
        self.control_panel.setFrameStyle(QFrame.StyledPanel)
        panel_layout = QVBoxLayout(self.control_panel)

        # Title
        title_label = QLabel("Arm Status")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        panel_layout.addWidget(title_label)

        # Left arm status
        left_label = QLabel("Left Arm")
        left_label.setFont(QFont("Arial", 12, QFont.Bold))
        panel_layout.addWidget(left_label)

        self.left_status = QLabel("Not connected")
        self.left_status.setFont(QFont("Monospace"))
        panel_layout.addWidget(self.left_status)

        # Right arm status
        right_label = QLabel("Right Arm")
        right_label.setFont(QFont("Arial", 12, QFont.Bold))
        panel_layout.addWidget(right_label)

        self.right_status = QLabel("Not connected")
        self.right_status.setFont(QFont("Monospace"))
        panel_layout.addWidget(self.right_status)

        # Workspace info
        workspace_label = QLabel("Workspace Info")
        workspace_label.setFont(QFont("Arial", 12, QFont.Bold))
        panel_layout.addWidget(workspace_label)

        self.workspace_info = QLabel("No workspace data")
        self.workspace_info.setFont(QFont("Monospace"))
        panel_layout.addWidget(self.workspace_info)

        # Calibration info
        if self.calibration:
            cal_label = QLabel("Calibration Info")
            cal_label.setFont(QFont("Arial", 12, QFont.Bold))
            panel_layout.addWidget(cal_label)

            cal_info = QLabel(
                f"Sample Origin: {self._format_coords(self.calibration.sample_origin)}\n"
                f"World Origin: {self._format_coords(self.calibration.world_origin)}"
            )
            cal_info.setFont(QFont("Monospace"))
            panel_layout.addWidget(cal_info)

        panel_layout.addStretch()

    def _add_coordinate_system(self, origin: Tuple[float, float, float], size: float = 50):
        """Add a coordinate system visualization"""
        origin = np.array(origin)

        # X-axis (red)
        x_points = np.array([origin, origin + [size, 0, 0]])
        x_axis = gl.GLLinePlotItem(pos=x_points, color=(1, 0, 0, 1), width=2)
        self.view.addItem(x_axis)

        # Y-axis (green)
        y_points = np.array([origin, origin + [0, size, 0]])
        y_axis = gl.GLLinePlotItem(pos=y_points, color=(0, 1, 0, 1), width=2)
        self.view.addItem(y_axis)

        # Z-axis (blue)
        z_points = np.array([origin, origin + [0, 0, size]])
        z_axis = gl.GLLinePlotItem(pos=z_points, color=(0, 0, 1, 1), width=2)
        self.view.addItem(z_axis)

    def _add_workspace_bounds(self):
        """Add workspace bounds visualization"""
        if not self.calibration:
            return

        # Add left arm workspace bounds
        if 'left_arm' in self.calibration and 'bounds' in self.calibration['left_arm']:
            bounds = self.calibration['left_arm']['bounds']
            self._add_bounding_box(bounds, (0, 1, 0, 0.2))  # Semi-transparent green

        # Add right arm workspace bounds
        if 'right_arm' in self.calibration and 'bounds' in self.calibration['right_arm']:
            bounds = self.calibration['right_arm']['bounds']
            self._add_bounding_box(bounds, (0, 0, 1, 0.2))  # Semi-transparent blue

    def _add_bounding_box(self, bounds: Dict[str, float], color: Tuple[float, float, float, float]):
        """Add a bounding box visualization"""
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        z_min, z_max = bounds['z_min'], bounds['z_max']

        # Define box vertices
        vertices = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])

        # Define box edges
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges
        ])

        # Create line segments
        lines = []
        for edge in edges:
            lines.append([vertices[edge[0]], vertices[edge[1]]])

        # Add all lines
        for line in lines:
            line_item = gl.GLLinePlotItem(pos=np.array(line), color=color, width=1)
            self.view.addItem(line_item)

    def _update_left_arm(self, data: dict):
        """Update left arm position data"""
        try:
            self.left_position = ArmPosition(
                x=data.get('x', 0),
                y=data.get('y', 0),
                z=data.get('z', 0),
                base_angle=data.get('b', 0),
                shoulder_angle=data.get('s', 0),
                elbow_angle=data.get('e', 0),
                tilt_angle=data.get('t', 0),
                roll_angle=data.get('r', 0),
                gripper_angle=data.get('g', 0),
                timestamp=data.get('timestamp', 0)
            )
        except Exception as e:
            self.statusBar().showMessage(f"Error parsing left arm data: {e}")

    def _update_right_arm(self, data: dict):
        """Update right arm position data"""
        try:
            self.right_position = ArmPosition(
                x=data.get('x', 0),
                y=data.get('y', 0),
                z=data.get('z', 0),
                base_angle=data.get('b', 0),
                shoulder_angle=data.get('s', 0),
                elbow_angle=data.get('e', 0),
                tilt_angle=data.get('t', 0),
                roll_angle=data.get('r', 0),
                gripper_angle=data.get('g', 0),
                timestamp=data.get('timestamp', 0)
            )
        except Exception as e:
            self.statusBar().showMessage(f"Error parsing right arm data: {e}")

    def _update_visualization(self):
        """Update the 3D visualization"""
        # Update left arm visualization
        if self.left_position:
            self._visualize_arm(self.left_position, self.left_arm_item, self.left_ee, (0, 1, 0))
            self._update_status_display(self.left_position, self.left_status, "Left")

        # Update right arm visualization
        if self.right_position:
            self._visualize_arm(self.right_position, self.right_arm_item, self.right_ee, (0, 0, 1))
            self._update_status_display(self.right_position, self.right_status, "Right")

    def _visualize_arm(self, position: ArmPosition, arm_item: gl.GLLinePlotItem,
                       ee_item: gl.GLScatterPlotItem, color: Tuple[float, float, float]):
        """Visualize a single arm"""
        # For now, just show end effector position
        # In a more advanced implementation, you could draw the full arm geometry
        ee_pos = np.array([[position.x, position.y, position.z]])
        ee_item.setData(pos=ee_pos, color=(*color, 1))

        # Draw simple line from origin to end effector for now
        if self.calibration:
            origin = np.array(self.calibration.sample_origin)
            ee_pos_array = np.array([origin, [position.x, position.y, position.z]])
            arm_item.setData(pos=ee_pos_array, color=(*color, 0.7))

    def _update_status_display(self, position: ArmPosition, label: QLabel, arm_name: str):
        """Update the status display for an arm"""
        status_text = (
            f"X: {position.x:7.2f} mm\n"
            f"Y: {position.y:7.2f} mm\n"
            f"Z: {position.z:7.2f} mm\n"
            f"Base:  {np.degrees(position.base_angle):6.1f}°\n"
            f"Shoulder: {np.degrees(position.shoulder_angle):6.1f}°\n"
            f"Elbow: {np.degrees(position.elbow_angle):6.1f}°\n"
            f"Tilt:  {np.degrees(position.tilt_angle):6.1f}°\n"
            f"Roll:  {np.degrees(position.roll_angle):6.1f}°\n"
            f"Gripper: {np.degrees(position.gripper_angle):6.1f}°"
        )
        label.setText(status_text)

    def _format_coords(self, coords: Tuple[float, float, float]) -> str:
        """Format coordinates for display"""
        return f"({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})"

    def _handle_error(self, error_msg: str):
        """Handle serial errors"""
        self.statusBar().showMessage(error_msg)

    def closeEvent(self, event):
        """Handle window close event"""
        self.left_reader.stop()
        self.right_reader.stop()
        self.timer.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 3D visualization of dual robotic arms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_arms.py /dev/ttyUSB0 /dev/ttyUSB1 calibration.pkl
        """
    )

    parser.add_argument("left_port", help="Serial port for left arm (e.g., /dev/ttyUSB0)")
    parser.add_argument("right_port", help="Serial port for right arm (e.g., /dev/ttyUSB1)")
    parser.add_argument("calibration", help="Calibration file (e.g., calibration.pkl)")

    args = parser.parse_args()

    app = QApplication(sys.argv)
    visualizer = ArmVisualizer(args.left_port, args.right_port, args.calibration)
    visualizer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
