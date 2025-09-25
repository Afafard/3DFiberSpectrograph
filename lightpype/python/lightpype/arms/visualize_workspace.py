#!/usr/bin/env python3
"""
Enhanced 3D Visualization for Dual Robotic Arm Calibration System

Updated to work with new JSON calibration format and 11-point calibration system.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox,
                             QGroupBox, QCheckBox, QMessageBox, QDoubleSpinBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import json
import time

# Import workspace utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from workspace_utils import is_point_in_hull  # We'll use this for hull checks


class ArmKinematicsModel:
    """Handles the kinematic modeling of 5-DOF robotic arms"""

    def __init__(self, arm_type="left", base_position=(0, 0, 0)):
        self.arm_type = arm_type
        self.base_position = np.array(base_position, dtype=float)

        # Arm dimensions (mm) - updated to match RoArm-M3 Pro specs
        self.arm_dimensions = {
            'base_height': 126.01,
            'arm1_length': 236.82,
            'pivot_length': 30.0,
            'arm2_length': 316.15
        }

        # Joint lengths (segments between joints)
        self.joint_lengths = [
            self.arm_dimensions['base_height'],
            self.arm_dimensions['arm1_length'],
            self.arm_dimensions['pivot_length'],
            self.arm_dimensions['arm2_length']
        ]

        # Current joint angles (radians)
        self.joint_angles = [0.0] * 5

        # End effector position
        self.end_effector = np.array([0.0, 0.0, 0.0])

        # Arm segments for visualization
        self.segments = []

        # Gripper state
        self.gripper_closed = False

    def set_joint_angles(self, angles):
        """Set joint angles and update end effector position"""
        if len(angles) != 5:
            raise ValueError("Must provide exactly 5 joint angles")

        self.joint_angles = angles
        self._calculate_end_effector()

    def _calculate_end_effector(self):
        """Calculate end effector position using forward kinematics"""
        # Initialize with base position
        current_pos = self.base_position.copy()

        # Apply transformations based on arm type
        if self.arm_type == "left":
            # Left arm faces right (positive X direction)
            base_axis = np.array([0, 0, 1])  # Base rotates around Z
            shoulder_axis = np.array([0, 1, 0])  # Shoulder rotates around Y
            elbow_axis = np.array([0, 1, 0])  # Elbow rotates around Y
            wrist1_axis = np.array([1, 0, 0])  # Wrist1 rotates around X
            wrist2_axis = np.array([0, 0, 1])  # Wrist2 rotates around Z
        else:  # right arm
            # Right arm faces left (negative X direction)
            base_axis = np.array([0, 0, 1])  # Base rotates around Z
            shoulder_axis = np.array([0, 1, 0])  # Shoulder rotates around Y
            elbow_axis = np.array([0, 1, 0])  # Elbow rotates around Y
            wrist1_axis = np.array([1, 0, 0])  # Wrist1 rotates around X
            wrist2_axis = np.array([0, 0, 1])  # Wrist2 rotates around Z

        # Apply rotations and translations sequentially
        segments = [current_pos.copy()]

        # Joint 1: Base rotation (around Z axis)
        rot_matrix = self._rotation_matrix(base_axis, self.joint_angles[0])
        current_pos += rot_matrix @ np.array([0, 0, self.joint_lengths[0]])
        segments.append(current_pos.copy())

        # Joint 2: Shoulder rotation (around Y axis)
        rot_matrix = self._rotation_matrix(shoulder_axis, self.joint_angles[1])
        current_pos += rot_matrix @ np.array([self.joint_lengths[1], 0, 0])
        segments.append(current_pos.copy())

        # Joint 3: Elbow rotation (around Y axis)
        rot_matrix = self._rotation_matrix(elbow_axis, self.joint_angles[2])
        current_pos += rot_matrix @ np.array([self.joint_lengths[2], 0, 0])
        segments.append(current_pos.copy())

        # Joint 4: Wrist1 rotation (around X axis)
        rot_matrix = self._rotation_matrix(wrist1_axis, self.joint_angles[3])
        current_pos += rot_matrix @ np.array([self.joint_lengths[3], 0, 0])
        segments.append(current_pos.copy())

        # Joint 5: Wrist2 rotation (no translation, just orientation)
        segments.append(current_pos.copy())

        self.end_effector = current_pos
        self.segments = segments

    def _rotation_matrix(self, axis, angle):
        """Create rotation matrix around arbitrary axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis

        return np.array([
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c]
        ])

    def get_segments(self):
        """Get all arm segments as numpy array for visualization"""
        return np.array(self.segments)

    def get_end_effector(self):
        """Get end effector position"""
        return self.end_effector.copy()

    def toggle_gripper(self):
        """Toggle gripper state"""
        self.gripper_closed = not self.gripper_closed
        return self.gripper_closed


class WorkspaceVisualizer(QMainWindow):
    """Main visualization window for dual robotic arm calibration"""

    def __init__(self, calibration_file="cube_calibration.json"):
        super().__init__()

        self.calibration_file = calibration_file
        self.workspace_data = None
        # Updated base positions to match our calibration geometry
        self.left_arm_model = ArmKinematicsModel("left", (-374.6, 0, 165.1))  # Left arm mounted on left wall
        self.right_arm_model = ArmKinematicsModel("right", (374.6, 0, 165.1))  # Right arm mounted on right wall

        # Calibration point mapping for overlap detection
        self.calibration_points = {}

        # Visualization modes
        self.visualization_modes = [
            "Default View",
            "Arm Kinematics Only",
            "Calibration Points Only",
            "Workspace Bounds Only",
            "Overlap Analysis",
            "Turntable Focused"
        ]
        self.current_mode = 0

        # Arm visibility states
        self.show_left_arm = True
        self.show_right_arm = True

        # Setup UI and load data
        self.init_ui()
        self.load_calibration_data()

        # Start real-time update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_realtime_positions)
        self.timer.start(100)  # Update every 100ms

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Dual Robotic Arm Calibration Visualization")
        self.setGeometry(100, 100, 1600, 900)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Create matplotlib 3D figure
        self.fig = Figure(figsize=(12, 10), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.fig)

        # Create 3D axis
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#444444')
        self.ax.yaxis.pane.set_edgecolor('#444444')
        self.ax.zaxis.pane.set_edgecolor('#444444')
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        # Set labels with high contrast colors
        self.ax.set_xlabel('X (mm)', color='#ffffff')
        self.ax.set_ylabel('Y (mm)', color='#ffffff')
        self.ax.set_zlabel('Z (mm)', color='#ffffff')
        self.ax.set_title('Dual Robotic Arm Calibration Visualization', color='#ffffff', pad=20)

        # Set tick colors
        self.ax.tick_params(axis='x', colors='#cccccc')
        self.ax.tick_params(axis='y', colors='#cccccc')
        self.ax.tick_params(axis='z', colors='#cccccc')

        # Add controls panel on the right
        control_panel = QWidget()
        control_panel.setStyleSheet("background-color: #2d2d2d; color: #ffffff;")
        control_panel.setFixedWidth(400)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(10, 10, 10, 10)

        # Mode selection
        mode_group = QGroupBox("Visualization Mode")
        mode_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(10)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.visualization_modes)
        self.mode_combo.currentIndexChanged.connect(self.change_visualization_mode)
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: #ffffff;
                selection-background-color: #555555;
            }
        """)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)

        # Arm visibility controls
        arm_group = QGroupBox("Arm Visibility")
        arm_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        arm_layout = QVBoxLayout()
        arm_layout.setSpacing(10)

        self.left_arm_checkbox = QCheckBox("Show Left Arm (Spectrometer)")
        self.left_arm_checkbox.setChecked(True)
        self.left_arm_checkbox.stateChanged.connect(self.toggle_left_arm)
        self.left_arm_checkbox.setStyleSheet("QCheckBox { spacing: 10px; color: #3498db; }")
        arm_layout.addWidget(self.left_arm_checkbox)

        self.right_arm_checkbox = QCheckBox("Show Right Arm (Illuminator)")
        self.right_arm_checkbox.setChecked(True)
        self.right_arm_checkbox.stateChanged.connect(self.toggle_right_arm)
        self.right_arm_checkbox.setStyleSheet("QCheckBox { spacing: 10px; color: #e74c3c; }")
        arm_layout.addWidget(self.right_arm_checkbox)

        arm_group.setLayout(arm_layout)
        control_layout.addWidget(arm_group)

        # Calibration points controls
        calib_group = QGroupBox("Calibration Points")
        calib_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        calib_layout = QVBoxLayout()
        calib_layout.setSpacing(10)

        self.show_calib_points_checkbox = QCheckBox("Show Calibration Points")
        self.show_calib_points_checkbox.setChecked(True)
        self.show_calib_points_checkbox.stateChanged.connect(self.toggle_calibration_points)
        self.show_calib_points_checkbox.setStyleSheet("QCheckBox { spacing: 10px; }")
        calib_layout.addWidget(self.show_calib_points_checkbox)

        self.show_overlap_checkbox = QCheckBox("Highlight Overlaps")
        self.show_overlap_checkbox.setChecked(True)
        self.show_overlap_checkbox.stateChanged.connect(self.update_calibration_points)
        self.show_overlap_checkbox.setStyleSheet("QCheckBox { spacing: 10px; }")
        calib_layout.addWidget(self.show_overlap_checkbox)

        self.overlap_threshold_slider = QSlider(Qt.Horizontal)
        self.overlap_threshold_slider.setMinimum(1)
        self.overlap_threshold_slider.setMaximum(50)
        self.overlap_threshold_slider.setValue(10)
        self.overlap_threshold_slider.valueChanged.connect(self.update_calibration_points)
        self.overlap_threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3d3d3d;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #555555;
                border: 1px solid #777777;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        calib_layout.addWidget(self.overlap_threshold_slider)

        overlap_label = QLabel(f"Overlap Threshold: {self.overlap_threshold_slider.value()} mm")
        overlap_label.setStyleSheet("QLabel { color: #cccccc; }")
        self.overlap_threshold_slider.valueChanged.connect(
            lambda: overlap_label.setText(f"Overlap Threshold: {self.overlap_threshold_slider.value()} mm")
        )
        calib_layout.addWidget(overlap_label)

        calib_group.setLayout(calib_layout)
        control_layout.addWidget(calib_group)

        # Workspace controls
        workspace_group = QGroupBox("Workspace")
        workspace_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        workspace_layout = QVBoxLayout()
        workspace_layout.setSpacing(10)

        self.show_left_hull_checkbox = QCheckBox("Show Left Arm Workspace")
        self.show_left_hull_checkbox.setChecked(True)
        self.show_left_hull_checkbox.stateChanged.connect(self.toggle_left_hull)
        self.show_left_hull_checkbox.setStyleSheet("QCheckBox { spacing: 10px; color: #3498db; }")
        workspace_layout.addWidget(self.show_left_hull_checkbox)

        self.show_right_hull_checkbox = QCheckBox("Show Right Arm Workspace")
        self.show_right_hull_checkbox.setChecked(True)
        self.show_right_hull_checkbox.stateChanged.connect(self.toggle_right_hull)
        self.show_right_hull_checkbox.setStyleSheet("QCheckBox { spacing: 10px; color: #e74c3c; }")
        workspace_layout.addWidget(self.show_right_hull_checkbox)

        self.hull_transparency_slider = QSlider(Qt.Horizontal)
        self.hull_transparency_slider.setMinimum(10)
        self.hull_transparency_slider.setMaximum(100)
        self.hull_transparency_slider.setValue(30)
        self.hull_transparency_slider.valueChanged.connect(self.update_hull_transparency)
        self.hull_transparency_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3d3d3d;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #555555;
                border: 1px solid #777777;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        workspace_layout.addWidget(self.hull_transparency_slider)

        transparency_label = QLabel(f"Hull Transparency: {self.hull_transparency_slider.value()}%")
        transparency_label.setStyleSheet("QLabel { color: #cccccc; }")
        self.hull_transparency_slider.valueChanged.connect(
            lambda: transparency_label.setText(f"Hull Transparency: {self.hull_transparency_slider.value()}%")
        )
        workspace_layout.addWidget(transparency_label)

        workspace_group.setLayout(workspace_layout)
        control_layout.addWidget(workspace_group)

        # Manual control panel
        manual_group = QGroupBox("Manual Control")
        manual_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        manual_layout = QVBoxLayout()
        manual_layout.setSpacing(10)

        # Position controls
        position_layout = QGridLayout()
        position_layout.setSpacing(10)

        # X position
        x_label = QLabel("X (mm):")
        x_label.setStyleSheet("QLabel { color: #ffffff; }")
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-500, 500)
        self.x_spinbox.setSingleStep(10)
        self.x_spinbox.setValue(0)
        self.x_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        position_layout.addWidget(x_label, 0, 0)
        position_layout.addWidget(self.x_spinbox, 0, 1)

        # Y position
        y_label = QLabel("Y (mm):")
        y_label.setStyleSheet("QLabel { color: #ffffff; }")
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-500, 500)
        self.y_spinbox.setSingleStep(10)
        self.y_spinbox.setValue(0)
        self.y_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        position_layout.addWidget(y_label, 1, 0)
        position_layout.addWidget(self.y_spinbox, 1, 1)

        # Z position
        z_label = QLabel("Z (mm):")
        z_label.setStyleSheet("QLabel { color: #ffffff; }")
        self.z_spinbox = QDoubleSpinBox()
        self.z_spinbox.setRange(0, 800)
        self.z_spinbox.setSingleStep(10)
        self.z_spinbox.setValue(150)
        self.z_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        position_layout.addWidget(z_label, 2, 0)
        position_layout.addWidget(self.z_spinbox, 2, 1)

        manual_layout.addLayout(position_layout)

        # Control buttons
        control_button_layout = QGridLayout()
        control_button_layout.setSpacing(10)

        # Left arm controls
        left_label = QLabel("Left Arm (Spectrometer):")
        left_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")
        control_button_layout.addWidget(left_label, 0, 0, 1, 2)

        self.left_move_button = QPushButton("Move Left Arm")
        self.left_move_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.left_move_button.clicked.connect(lambda: self.move_arm("left"))
        control_button_layout.addWidget(self.left_move_button, 1, 0, 1, 2)

        self.left_gripper_button = QPushButton("Open Gripper (Left)")
        self.left_gripper_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.left_gripper_button.clicked.connect(lambda: self.toggle_gripper("left"))
        control_button_layout.addWidget(self.left_gripper_button, 2, 0, 1, 2)

        # Right arm controls
        right_label = QLabel("Right Arm (Illuminator):")
        right_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        control_button_layout.addWidget(right_label, 3, 0, 1, 2)

        self.right_move_button = QPushButton("Move Right Arm")
        self.right_move_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.right_move_button.clicked.connect(lambda: self.move_arm("right"))
        control_button_layout.addWidget(self.right_move_button, 4, 0, 1, 2)

        self.right_gripper_button = QPushButton("Open Gripper (Right)")
        self.right_gripper_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.right_gripper_button.clicked.connect(lambda: self.toggle_gripper("right"))
        control_button_layout.addWidget(self.right_gripper_button, 5, 0, 1, 2)

        manual_layout.addLayout(control_button_layout)

        # Rotation controls
        rotation_layout = QGridLayout()
        rotation_layout.setSpacing(10)

        rot_label = QLabel("Rotation (degrees):")
        rot_label.setStyleSheet("QLabel { color: #ffffff; }")
        rotation_layout.addWidget(rot_label, 0, 0, 1, 2)

        # Base rotation
        base_label = QLabel("Base:")
        base_label.setStyleSheet("QLabel { color: #ffffff; }")
        self.base_spinbox = QDoubleSpinBox()
        self.base_spinbox.setRange(-180, 180)
        self.base_spinbox.setSingleStep(5)
        self.base_spinbox.setValue(0)
        self.base_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        rotation_layout.addWidget(base_label, 1, 0)
        rotation_layout.addWidget(self.base_spinbox, 1, 1)

        # Wrist rotation
        wrist_label = QLabel("Wrist:")
        wrist_label.setStyleSheet("QLabel { color: #ffffff; }")
        self.wrist_spinbox = QDoubleSpinBox()
        self.wrist_spinbox.setRange(-180, 180)
        self.wrist_spinbox.setSingleStep(5)
        self.wrist_spinbox.setValue(0)
        self.wrist_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        rotation_layout.addWidget(wrist_label, 2, 0)
        rotation_layout.addWidget(self.wrist_spinbox, 2, 1)

        manual_layout.addLayout(rotation_layout)

        manual_group.setLayout(manual_layout)
        control_layout.addWidget(manual_group)

        # Real-time position display
        realtime_group = QGroupBox("Real-time Positions")
        realtime_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        realtime_layout = QVBoxLayout()
        realtime_layout.setSpacing(10)

        self.left_pos_label = QLabel("Left Arm: Position (0, 0, 0)")
        self.left_pos_label.setStyleSheet("QLabel { color: #3498db; font-family: monospace; font-size: 12px; }")
        realtime_layout.addWidget(self.left_pos_label)

        self.right_pos_label = QLabel("Right Arm: Position (0, 0, 0)")
        self.right_pos_label.setStyleSheet("QLabel { color: #e74c3c; font-family: monospace; font-size: 12px; }")
        realtime_layout.addWidget(self.right_pos_label)

        realtime_group.setLayout(realtime_layout)
        control_layout.addWidget(realtime_group)

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_camera)
        self.reset_view_button.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        button_layout.addWidget(self.reset_view_button)

        self.save_image_button = QPushButton("Save Screenshot")
        self.save_image_button.clicked.connect(self.save_screenshot)
        self.save_image_button.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        button_layout.addWidget(self.save_image_button)

        control_layout.addLayout(button_layout)

        # Add spacer to push controls to top
        control_layout.addStretch()

        # Add matplotlib canvas and control panel to main layout
        main_layout.addWidget(self.canvas, 7)
        main_layout.addWidget(control_panel, 3)

        # Initialize the plot
        self.init_plot()

    def init_plot(self):
        """Initialize the 3D plot with basic elements"""
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])

        # Add coordinate system indicator
        self._add_coordinate_axes()

        # Add aluminum cube (2020 extrusion frame)
        self._add_aluminum_cube()

        # Add turntable
        self._add_turntable()

        # Sample center point
        self.sample_center_point = None

        # Arm visualization elements
        self.left_arm_lines = []
        self.right_arm_lines = []
        self.left_gripper_lines = []
        self.right_gripper_lines = []

        # Calibration point visualization
        self.calibration_point_items = {}

        # Hull meshes for workspace bounds
        self.left_hull_collection = None
        self.right_hull_collection = None

        # Initial plot limits
        self.ax.set_xlim(-500, 500)
        self.ax.set_ylim(-500, 500)
        self.ax.set_zlim(0, 800)

        # Set initial view
        self.ax.view_init(elev=20, azim=45)

        # Update plot
        self.fig.tight_layout()
        self.canvas.draw()

    def load_calibration_data(self):
        """Load calibration data from JSON file"""
        try:
            if not os.path.exists(self.calibration_file):
                QMessageBox.warning(self, "File Not Found",
                                    f"Calibration file '{self.calibration_file}' not found. Using default settings.")
                # Initialize with default positions
                self._initialize_default_positions()
                return

            with open(self.calibration_file, 'r') as f:
                self.workspace_data = json.load(f)

            # Extract left and right arm data
            left_arm_data = self.workspace_data.get('left_arm', {})
            right_arm_data = self.workspace_data.get('right_arm', {})

            # Load workspace points
            self.left_workspace_points = np.array(left_arm_data.get('workspace_points', []))
            self.right_workspace_points = np.array(right_arm_data.get('workspace_points', []))

            # Load reference points
            self.left_ref_points = left_arm_data.get('calibration_points', [])
            self.right_ref_points = right_arm_data.get('calibration_points', [])

            # Extract transformation matrices - Updated to match new format
            self.transformation_matrices = {
                'left': {
                    'rotation': np.array(left_arm_data.get('rotation_matrix', [])),
                    'translation': np.array(left_arm_data.get('translation_vector', []))
                },
                'right': {
                    'rotation': np.array(right_arm_data.get('rotation_matrix', [])),
                    'translation': np.array(right_arm_data.get('translation_vector', []))
                }
            }

            # Extract world origin and sample center
            self.world_origin = np.array(self.workspace_data.get('metadata', {}).get('world_origin', [0, 0, 0]))
            self.sample_center = np.array(self.workspace_data.get('metadata', {}).get('world_origin', [0, 0, 57.15]))

            # Build calibration point dictionary for overlap detection
            self._build_calibration_point_map()

            # Update visualization elements
            self.update_all_visualizations()

            print(f"Successfully loaded calibration data from {self.calibration_file}")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Calibration",
                                 f"Failed to load calibration data: {str(e)}")
            print(f"Error loading calibration data: {e}")
            # Initialize with default positions
            self._initialize_default_positions()

    def _initialize_default_positions(self):
        """Initialize with default arm positions"""
        # Set default joint angles for both arms
        default_angles = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_arm_model.set_joint_angles(default_angles)
        self.right_arm_model.set_joint_angles(default_angles)

        # Build default calibration points map
        self._build_default_calibration_point_map()

        # Update visualization
        self.update_all_visualizations()

    def _build_calibration_point_map(self):
        """Build calibration point map for overlap detection using our 11-point system"""
        self.calibration_points = {}

        # List of point names from our calibration
        point_names = [
            "Sample center on turntable",
            "Turntable front extremum (Y+)",
            "Turntable back extremum (Y-)",
            "Cross beam front (mid-height, Y=+374.6)",
            "Cross beam back (mid-height, Y=-374.6)",
            "Turntable left extremum (X-)",
            "Left arm mount corner — rear interior",
            "Left arm mount corner — front interior",
            "Turntable right extremum (X+)",
            "Right arm mount corner — rear interior",
            "Right arm mount corner — front interior"
        ]

        # Process left arm reference points
        for i, ref_point in enumerate(self.left_ref_points):
            if not ref_point:
                continue

            world_coords = tuple(ref_point['world_coords'])
            self.calibration_points[world_coords] = {
                'name': point_names[i],
                'left_arm': ref_point.get('arm_coords', None),
                'right_arm': None,
                'shared': False
            }

        # Process right arm reference points
        for i, ref_point in enumerate(self.right_ref_points):
            if not ref_point:
                continue

            world_coords = tuple(ref_point['world_coords'])
            if world_coords in self.calibration_points:
                # This point is shared (should be points 0-4)
                self.calibration_points[world_coords]['right_arm'] = ref_point.get('arm_coords', None)
                self.calibration_points[world_coords]['shared'] = True
            else:
                # This is a right-arm-only point (points 8-10)
                self.calibration_points[world_coords] = {
                    'name': point_names[i],
                    'left_arm': None,
                    'right_arm': ref_point.get('arm_coords', None),
                    'shared': False
                }

    def _build_default_calibration_point_map(self):
        """Build default calibration point map for demonstration"""
        # Create some default points for visualization
        default_points = [
            ((0, 0, 57.15), "Sample center on turntable", True),  # Shared point
            ((0, 174.625, 57.15), "Turntable front", True),  # Shared point
            ((0, -174.625, 57.15), "Turntable back", True),  # Shared point
            ((0, 374.6, 342.85), "Cross beam front", True),  # Shared point
            ((0, -374.6, 342.85), "Cross beam back", True),  # Shared point
            ((-174.625, 0, 57.15), "Turntable left", False),  # Left only
            ((-374.6, -374.6, 165.1), "Left rear corner", False),  # Left only
            ((-374.6, 374.6, 165.1), "Left front corner", False),  # Left only
            ((174.625, 0, 57.15), "Turntable right", False),  # Right only
            ((374.6, -374.6, 165.1), "Right rear corner", False),  # Right only
            ((374.6, 374.6, 165.1), "Right front corner", False)  # Right only
        ]

        self.calibration_points = {}
        for i, (coords, name, shared) in enumerate(default_points):
            world_coords = tuple(coords)
            self.calibration_points[world_coords] = {
                'name': name,
                'left_arm': coords if not shared or i % 2 == 0 else None,
                'right_arm': coords if not shared or i % 2 == 1 else None,
                'shared': shared
            }

    def _add_coordinate_axes(self):
        """Add 3D coordinate axes to the scene"""
        # X axis (red)
        self.ax.quiver(0, 0, 0, 200, 0, 0, color='#ff5555', arrow_length_ratio=0.1, linewidth=3)
        # Y axis (green)
        self.ax.quiver(0, 0, 0, 0, 200, 0, color='#55ff55', arrow_length_ratio=0.1, linewidth=3)
        # Z axis (blue)
        self.ax.quiver(0, 0, 0, 0, 0, 200, color='#5555ff', arrow_length_ratio=0.1, linewidth=3)

        # Labels
        self.ax.text(210, 0, 0, 'X', color='#ff5555', fontsize=12, weight='bold')
        self.ax.text(0, 210, 0, 'Y', color='#55ff55', fontsize=12, weight='bold')
        self.ax.text(0, 0, 210, 'Z', color='#5555ff', fontsize=12, weight='bold')

    def _add_aluminum_cube(self):
        """Add cube representing 2020 aluminum extrusion frame (800mm)"""
        size = 800

        # Define cube vertices
        vertices = np.array([
            [size / 2, size / 2, -size / 2], [-size / 2, size / 2, -size / 2],
            [-size / 2, -size / 2, -size / 2], [size / 2, -size / 2, -size / 2],
            [size / 2, size / 2, size / 2], [-size / 2, size / 2, size / 2],
            [-size / 2, -size / 2, size / 2], [size / 2, -size / 2, size / 2]
        ])

        # Define cube edges (12 edges)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
        ]

        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                         color='#aaaaaa', linewidth=1.5, alpha=0.8)

        # Add mounting frame at Z=165.1mm (arms mounting plane)
        mounting_height = 165.1
        mounting_vertices = np.array([
            [size / 2, size / 2, mounting_height], [-size / 2, size / 2, mounting_height],
            [-size / 2, -size / 2, mounting_height], [size / 2, -size / 2, mounting_height]
        ])

        mounting_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0]  # mounting frame
        ]

        for edge in mounting_edges:
            start = mounting_vertices[edge[0]]
            end = mounting_vertices[edge[1]]
            self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                         color='#ffaa00', linewidth=2, alpha=0.9)

        # Add horizontal crossbars at Z=342.85mm (cross beam)
        cross_height = 342.85
        cross_vertices = np.array([
            [size / 2, 0, cross_height], [-size / 2, 0, cross_height]
        ])

        self.ax.plot([cross_vertices[0][0], cross_vertices[1][0]],
                     [cross_vertices[0][1], cross_vertices[1][1]],
                     [cross_vertices[0][2], cross_vertices[1][2]],
                     color='#ff55ff', linewidth=3, alpha=0.9)

        # Add vertical supports for cross beam
        self.ax.plot([0, 0], [0, 0], [cross_height, size / 2],
                     color='#ff55ff', linewidth=2, alpha=0.9)

        # Add arm mounting brackets
        # Left arm bracket
        left_bracket_points = [
            [-size / 2, -1.5 * 25.4, mounting_height],  # 1.5" from center
            [-size / 2, 1.5 * 25.4, mounting_height]  # 1.5" from center
        ]
        self.ax.plot([left_bracket_points[0][0], left_bracket_points[1][0]],
                     [left_bracket_points[0][1], left_bracket_points[1][1]],
                     [left_bracket_points[0][2], left_bracket_points[1][2]],
                     color='#00ff00', linewidth=5, alpha=0.9)

        # Right arm bracket
        right_bracket_points = [
            [size / 2, -1.5 * 25.4, mounting_height],  # 1.5" from center
            [size / 2, 1.5 * 25.4, mounting_height]  # 1.5" from center
        ]
        self.ax.plot([right_bracket_points[0][0], right_bracket_points[1][0]],
                     [right_bracket_points[0][1], right_bracket_points[1][1]],
                     [right_bracket_points[0][2], right_bracket_points[1][2]],
                     color='#00ff00', linewidth=5, alpha=0.9)

    def _add_turntable(self):
        """Add turntable representation"""
        # Turntable is a cylinder with radius 174.625mm at Z=57.15mm
        n_points = 60
        theta = np.linspace(0, 2 * np.pi, n_points)
        radius = 174.625
        z_height = 57.15

        # Create circle points for turntable edge
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        z_circle = np.full_like(x_circle, z_height)

        # Turntable edge
        self.ax.plot(x_circle, y_circle, z_circle, color='#ffaa00', linewidth=3, alpha=0.9)

        # Turntable center
        self.ax.scatter([0], [0], [z_height], color='#ffff00', s=150, marker='o', alpha=0.9)

        # Add turntable surface
        r = np.linspace(0, radius, 20)
        theta_grid, r_grid = np.meshgrid(theta, r)
        x_surface = r_grid * np.cos(theta_grid)
        y_surface = r_grid * np.sin(theta_grid)
        z_surface = np.full_like(x_surface, z_height)

        # Only draw a subset of the surface for better performance
        self.ax.plot_surface(x_surface[::4, ::4], y_surface[::4, ::4], z_surface[::4, ::4],
                             color='#ffaa00', alpha=0.2, linewidth=0)

    def update_all_visualizations(self):
        """Update all visualization elements"""
        self.update_arm_visualization()
        self.update_calibration_points()
        self.update_hull_visualizations()

    def update_arm_visualization(self):
        """Update the 3D visualization of both arms"""

        # Clear existing arm lines
        for line in self.left_arm_lines:
            line.remove()
        for line in self.right_arm_lines:
            line.remove()
        for line in self.left_gripper_lines:
            line.remove()
        for line in self.right_gripper_lines:
            line.remove()

        self.left_arm_lines = []
        self.right_arm_lines = []
        self.left_gripper_lines = []
        self.right_gripper_lines = []

        if not self.show_left_arm and not self.show_right_arm:
            self.canvas.draw()
            return

        # Create arm segments for left arm
        if self.show_left_arm:
            segments = self.left_arm_model.get_segments()
            if len(segments) > 1:
                # Draw each segment with different colors
                segment_colors = ['#3498db', '#2980b9', '#1f618d', '#154360']  # Blue gradient
                for i in range(len(segments) - 1):
                    x_vals = [segments[i][0], segments[i + 1][0]]
                    y_vals = [segments[i][1], segments[i + 1][1]]
                    z_vals = [segments[i][2], segments[i + 1][2]]

                    # Color gradient from base to end effector
                    color = segment_colors[min(i, len(segment_colors) - 1)]

                    line = self.ax.plot(x_vals, y_vals, z_vals,
                                        color=color, linewidth=6, alpha=0.9)[0]
                    self.left_arm_lines.append(line)

                # Add end effector point
                end_point = self.ax.scatter(segments[-1][0], segments[-1][1], segments[-1][2],
                                            color='#3498db', s=150, marker='o', alpha=0.9)
                self.left_arm_lines.append(end_point)

                # Add gripper visualization (spectrometer)
                self._draw_gripper(segments[-1], "left", "spectrometer")

        # Create arm segments for right arm
        if self.show_right_arm:
            segments = self.right_arm_model.get_segments()
            if len(segments) > 1:
                # Draw each segment with different colors
                segment_colors = ['#e74c3c', '#c0392b', '#922b21', '#641e16']  # Red gradient
                for i in range(len(segments) - 1):
                    x_vals = [segments[i][0], segments[i + 1][0]]
                    y_vals = [segments[i][1], segments[i + 1][1]]
                    z_vals = [segments[i][2], segments[i + 1][2]]

                    # Color gradient from base to end effector
                    color = segment_colors[min(i, len(segment_colors) - 1)]

                    line = self.ax.plot(x_vals, y_vals, z_vals,
                                        color=color, linewidth=6, alpha=0.9)[0]
                    self.right_arm_lines.append(line)

                # Add end effector point
                end_point = self.ax.scatter(segments[-1][0], segments[-1][1], segments[-1][2],
                                            color='#e74c3c', s=150, marker='o', alpha=0.9)
                self.right_arm_lines.append(end_point)

                # Add gripper visualization (illuminator)
                self._draw_gripper(segments[-1], "right", "illuminator")

        # Update position labels
        left_pos = self.left_arm_model.get_end_effector()
        right_pos = self.right_arm_model.get_end_effector()

        self.left_pos_label.setText(f"Left Arm: ({left_pos[0]:.1f}, {left_pos[1]:.1f}, {left_pos[2]:.1f})")
        self.right_pos_label.setText(f"Right Arm: ({right_pos[0]:.1f}, {right_pos[1]:.1f}, {right_pos[2]:.1f})")

        self.canvas.draw()

    def _draw_gripper(self, position, arm_type, device_type):
        """Draw gripper visualization at end effector position"""
        if arm_type == "left":
            color = '#3498db'
            lines = self.left_gripper_lines
        else:
            color = '#e74c3c'
            lines = self.right_gripper_lines

        # Draw device representation at end effector
        x, y, z = position
        size = 20

        # Draw device cube
        device_vertices = np.array([
            [x - size / 2, y - size / 2, z - size / 2], [x + size / 2, y - size / 2, z - size / 2],
            [x + size / 2, y + size / 2, z - size / 2], [x - size / 2, y + size / 2, z - size / 2],
            [x - size / 2, y - size / 2, z + size / 2], [x + size / 2, y - size / 2, z + size / 2],
            [x + size / 2, y + size / 2, z + size / 2], [x - size / 2, y + size / 2, z + size / 2]
        ])

        device_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
        ]

        for edge in device_edges:
            start = device_vertices[edge[0]]
            end = device_vertices[edge[1]]
            line = self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                                color=color, linewidth=2, alpha=0.8)[0]
            lines.append(line)

        # Add device label
        label = self.ax.text(x, y, z + size / 2 + 10,
                             f"{device_type.capitalize()}",
                             color=color, fontsize=10, weight='bold')
        lines.append(label)

    def update_calibration_points(self):
        """Update calibration points visualization with overlap detection"""

        # Clear existing calibration point items
        for item in self.calibration_point_items.values():
            if isinstance(item, list):
                for sub_item in item:
                    try:
                        sub_item.remove()
                    except:
                        pass
            else:
                try:
                    item.remove()
                except:
                    pass

        self.calibration_point_items = {}

        if not self.show_calib_points_checkbox.isChecked():
            self.canvas.draw()
            return

        # Create visualization for each calibration point
        threshold = self.overlap_threshold_slider.value()

        for world_coords, point_info in self.calibration_points.items():
            # Position of the calibration point in world coordinates
            pos = np.array(world_coords)

            if point_info['shared'] and self.show_overlap_checkbox.isChecked():
                # Shared point - highlight with special color
                color = '#ffff00'  # Bright yellow

                # Create a larger point for shared points
                item = self.ax.scatter(pos[0], pos[1], pos[2],
                                       color=color, s=250, marker='*', alpha=0.9)

                # Add label
                label = self.ax.text(pos[0], pos[1], pos[2] + 20,
                                     f"Shared: {point_info['name']}",
                                     color='#ffff00', fontsize=10, weight='bold')

                self.calibration_point_items[world_coords] = [item, label]
            else:
                # Point belongs to only one arm
                if point_info['left_arm'] is not None:
                    # Left arm point - blue color
                    color = '#3498db'
                    item = self.ax.scatter(pos[0], pos[1], pos[2],
                                           color=color, s=120, marker='o', alpha=0.8)

                    label = self.ax.text(pos[0], pos[1], pos[2] + 20,
                                         f"Left: {point_info['name']}",
                                         color='#3498db', fontsize=9)

                    self.calibration_point_items[world_coords] = [item, label]

                elif point_info['right_arm'] is not None:
                    # Right arm point - red color
                    color = '#e74c3c'
                    item = self.ax.scatter(pos[0], pos[1], pos[2],
                                           color=color, s=120, marker='o', alpha=0.8)

                    label = self.ax.text(pos[0], pos[1], pos[2] + 20,
                                         f"Right: {point_info['name']}",
                                         color='#e74c3c', fontsize=9)

                    self.calibration_point_items[world_coords] = [item, label]

        self.canvas.draw()

    def update_hull_visualizations(self):
        """Update workspace hull visualizations"""

        # Clear existing hull collections
        if self.left_hull_collection:
            try:
                self.left_hull_collection.remove()
            except:
                pass
        if self.right_hull_collection:
            try:
                self.right_hull_collection.remove()
            except:
                pass

        self.left_hull_collection = None
        self.right_hull_collection = None

        if not self.show_left_hull_checkbox.isChecked() and not self.show_right_hull_checkbox.isChecked():
            self.canvas.draw()
            return

        alpha = self.hull_transparency_slider.value() / 100.0

        # Left arm workspace hull
        if self.show_left_hull_checkbox.isChecked() and len(self.left_workspace_points) >= 4:
            try:
                # Use stored convex hull if available
                if 'hull' in self.workspace_data['left_arm'] and self.workspace_data['left_arm']['hull'] is not None:
                    hull_vertices = self.left_workspace_points[self.workspace_data['left_arm']['hull']]
                    hull = ConvexHull(hull_vertices)
                    for simplex in hull.simplices:
                        pts = hull_vertices[simplex]
                        xs, ys, zs = pts.T
                        self.ax.plot(xs, ys, zs, color=(0.2, 0.6, 1.0, alpha), linewidth=0.5)
                else:
                    # Compute hull from all points
                    hull = ConvexHull(self.left_workspace_points)
                    for simplex in hull.simplices:
                        pts = self.left_workspace_points[simplex]
                        xs, ys, zs = pts.T
                        self.ax.plot(xs, ys, zs, color=(0.2, 0.6, 1.0, alpha), linewidth=0.5)
            except Exception as e:
                print(f"Could not create left hull: {e}")

        # Right arm workspace hull
        if self.show_right_hull_checkbox.isChecked() and len(self.right_workspace_points) >= 4:
            try:
                # Use stored convex hull if available
                if 'hull' in self.workspace_data['right_arm'] and self.workspace_data['right_arm']['hull'] is not None:
                    hull_vertices = self.right_workspace_points[self.workspace_data['right_arm']['hull']]
                    hull = ConvexHull(hull_vertices)
                    for simplex in hull.simplices:
                        pts = hull_vertices[simplex]
                        xs, ys, zs = pts.T
                        self.ax.plot(xs, ys, zs, color=(1.0, 0.6, 0.2, alpha), linewidth=0.5)
                else:
                    # Compute hull from all points
                    hull = ConvexHull(self.right_workspace_points)
                    for simplex in hull.simplices:
                        pts = self.right_workspace_points[simplex]
                        xs, ys, zs = pts.T
                        self.ax.plot(xs, ys, zs, color=(1.0, 0.6, 0.2, alpha), linewidth=0.5)
            except Exception as e:
                print(f"Could not create right hull: {e}")

        self.canvas.draw()

    def toggle_left_arm(self):
        """Toggle visibility of left arm"""
        self.show_left_arm = not self.show_left_arm
        self.update_arm_visualization()

    def toggle_right_arm(self):
        """Toggle visibility of right arm"""
        self.show_right_arm = not self.show_right_arm
        self.update_arm_visualization()

    def toggle_calibration_points(self):
        """Toggle visibility of calibration points"""
        if self.show_calib_points_checkbox.isChecked():
            self.update_calibration_points()
        else:
            # Remove all calibration point items
            for item in self.calibration_point_items.values():
                if isinstance(item, list):
                    for sub_item in item:
                        try:
                            sub_item.remove()
                        except:
                            pass
                else:
                    try:
                        item.remove()
                    except:
                        pass
            self.calibration_point_items = {}
            self.canvas.draw()

    def toggle_left_hull(self):
        """Toggle visibility of left arm workspace hull"""
        self.update_hull_visualizations()

    def toggle_right_hull(self):
        """Toggle visibility of right arm workspace hull"""
        self.update_hull_visualizations()

    def update_hull_transparency(self):
        """Update transparency of hull meshes"""
        self.update_hull_visualizations()

    def change_visualization_mode(self):
        """Change between different visualization modes"""
        mode = self.mode_combo.currentIndex()
        self.current_mode = mode

        # Reset all visibility settings based on mode
        if mode == 0:  # Default View
            self.show_left_arm = True
            self.show_right_arm = True
            self.show_calib_points_checkbox.setChecked(True)
            self.show_left_hull_checkbox.setChecked(True)
            self.show_right_hull_checkbox.setChecked(True)
            self.show_overlap_checkbox.setChecked(True)

        elif mode == 1:  # Arm Kinematics Only
            self.show_left_arm = True
            self.show_right_arm = True
            self.show_calib_points_checkbox.setChecked(False)
            self.show_left_hull_checkbox.setChecked(False)
            self.show_right_hull_checkbox.setChecked(False)
            self.show_overlap_checkbox.setChecked(False)

        elif mode == 2:  # Calibration Points Only
            self.show_left_arm = False
            self.show_right_arm = False
            self.show_calib_points_checkbox.setChecked(True)
            self.show_left_hull_checkbox.setChecked(False)
            self.show_right_hull_checkbox.setChecked(False)
            self.show_overlap_checkbox.setChecked(True)

        elif mode == 3:  # Workspace Bounds Only
            self.show_left_arm = False
            self.show_right_arm = False
            self.show_calib_points_checkbox.setChecked(False)
            self.show_left_hull_checkbox.setChecked(True)
            self.show_right_hull_checkbox.setChecked(True)
            self.show_overlap_checkbox.setChecked(False)

        elif mode == 4:  # Overlap Analysis
            self.show_left_arm = True
            self.show_right_arm = True
            self.show_calib_points_checkbox.setChecked(True)
            self.show_left_hull_checkbox.setChecked(False)
            self.show_right_hull_checkbox.setChecked(False)
            self.show_overlap_checkbox.setChecked(True)

        elif mode == 5:  # Turntable Focused
            self.show_left_arm = True
            self.show_right_arm = True
            self.show_calib_points_checkbox.setChecked(True)
            self.show_left_hull_checkbox.setChecked(False)
            self.show_right_hull_checkbox.setChecked(False)
            self.show_overlap_checkbox.setChecked(True)

        # Update UI to reflect changes
        self.left_arm_checkbox.setChecked(self.show_left_arm)
        self.right_arm_checkbox.setChecked(self.show_right_arm)

        # Update visualization
        self.update_all_visualizations()

    def update_realtime_positions(self):
        """Update arm positions from serial communication"""
        # In a real implementation, this would read from serial port
        # For now we'll simulate with sine waves to demonstrate functionality

        # Simulate arm movement with sine waves
        t = time.time()

        # Left arm joint angles (simulated movement)
        left_angles = [
            np.sin(t * 0.5) * np.pi / 4,
            np.cos(t * 0.7) * np.pi / 6,
            np.sin(t * 0.9) * np.pi / 3,
            np.cos(t * 1.2) * np.pi / 4,
            np.sin(t * 0.8) * np.pi / 2
        ]

        # Right arm joint angles (simulated movement)
        right_angles = [
            np.sin(t * 0.5 + np.pi) * np.pi / 4,
            np.cos(t * 0.7 + np.pi) * np.pi / 6,
            np.sin(t * 0.9 + np.pi) * np.pi / 3,
            np.cos(t * 1.2 + np.pi) * np.pi / 4,
            np.sin(t * 0.8 + np.pi) * np.pi / 2
        ]

        # Update models with simulated joint angles
        self.left_arm_model.set_joint_angles(left_angles)
        self.right_arm_model.set_joint_angles(right_angles)

        # Update visualization
        self.update_arm_visualization()

    def reset_camera(self):
        """Reset camera view to default position"""
        self.ax.view_init(elev=20, azim=45)
        self.canvas.draw()

    def save_screenshot(self):
        """Save current view as screenshot"""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_visualization_{timestamp}.png"

            self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')

            QMessageBox.information(self, "Screenshot Saved",
                                    f"Screenshot saved as {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error Saving Screenshot",
                                 f"Failed to save screenshot: {str(e)}")

    def move_arm(self, arm_type):
        """Move arm to specified position"""
        try:
            x = self.x_spinbox.value()
            y = self.y_spinbox.value()
            z = self.z_spinbox.value()

            # Base rotation in degrees
            base_rot = np.radians(self.base_spinbox.value())
            wrist_rot = np.radians(self.wrist_spinbox.value())

            # Move arm to position
            if arm_type == "left":
                # Update visualization with new position
                self.left_arm_model.set_joint_angles([
                    base_rot,  # Base angle
                    0.0,  # Shoulder (placeholder)
                    0.0,  # Elbow (placeholder)
                    wrist_rot,  # Wrist1
                    0.0  # Wrist2 (placeholder)
                ])
            else:  # right arm
                # Update visualization with new position
                self.right_arm_model.set_joint_angles([
                    base_rot,  # Base angle
                    0.0,  # Shoulder (placeholder)
                    0.0,  # Elbow (placeholder)
                    wrist_rot,  # Wrist1
                    0.0  # Wrist2 (placeholder)
                ])

            # Update visualization
            self.update_arm_visualization()

        except Exception as e:
            QMessageBox.warning(self, "Movement Error", f"Error moving arm: {str(e)}")

    def toggle_gripper(self, arm_type):
        """Toggle gripper state"""
        if arm_type == "left":
            state = self.left_arm_model.toggle_gripper()
            self.left_gripper_button.setText(f"{'Close' if state else 'Open'} Gripper (Left)")
        else:
            state = self.right_arm_model.toggle_gripper()
            self.right_gripper_button.setText(f"{'Close' if state else 'Open'} Gripper (Right)")

        self.update_arm_visualization()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Escape:
            self.close()

        elif event.key() == Qt.Key_R:
            self.reset_camera()

        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self.save_screenshot()

        super().keyPressEvent(event)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better dark theme support

    # Set application palette for dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    # Create and show the visualization window
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_file = os.path.join(script_dir, "cube_calibration.json")

    visualizer = WorkspaceVisualizer(calibration_file)
    visualizer.show()

    # Start the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
