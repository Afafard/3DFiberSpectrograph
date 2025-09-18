#!/usr/bin/env python3
"""
3D Visualization of calibrated robotic arm workspace using PyQt5 and OpenGL.

Usage:
    python visualize_workspace.py arm1_workspace.pkl
"""

import sys
import numpy as np
from scipy.spatial import ConvexHull
import pickle

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog,
                             QGroupBox, QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# OpenGL imports
try:
    from PyQt5.QtGui import QOpenGLWidget
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("PyOpenGL not installed! Install with: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)


class WorkspaceGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace_data = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.zoom = -500
        self.last_pos = None
        self.show_points = True
        self.show_hull = True
        self.show_wireframe = True
        self.transparency = 0.3
        self.point_size = 2.0

    def load_workspace(self, filename):
        """Load workspace data from file"""
        try:
            with open(filename, 'rb') as f:
                self.workspace_data = pickle.load(f)
            self.update()
            return True
        except Exception as e:
            print(f"Error loading workspace: {e}")
            return False

    def initializeGL(self):
        """Initialize OpenGL context"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height if height > 0 else 1
        gluPerspective(45, aspect, 1, 2000)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.workspace_data:
            return

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)

        # Center the view
        points = self.workspace_data['points']
        center = np.mean(points, axis=0)
        glTranslatef(-center[0], -center[1], -center[2])

        # Draw coordinate axes
        self.draw_axes()

        # Draw points
        if self.show_points:
            self.draw_points()

        # Draw convex hull
        if self.show_hull and self.workspace_data:
            self.draw_convex_hull()

    def draw_axes(self):
        """Draw coordinate axes"""
        glLineWidth(2.0)

        # X axis (red)
        glBegin(GL_LINES)
        glColor4f(1, 0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        glEnd()

        # Y axis (green)
        glBegin(GL_LINES)
        glColor4f(0, 1, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 100, 0)
        glEnd()

        # Z axis (blue)
        glBegin(GL_LINES)
        glColor4f(0, 0, 1, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 100)
        glEnd()

    def draw_points(self):
        """Draw recorded points"""
        points = self.workspace_data['points']
        if len(points) == 0:
            return

        # Create height-based colormap
        z_values = points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)

        glBegin(GL_POINTS)
        glPointSize(self.point_size)

        for point in points:
            # Height-based color (blue to red ramp)
            z_norm = (point[2] - z_min) / (z_max - z_min) if z_max > z_min else 0.5
            r = z_norm
            g = 0.5 * (1 - abs(z_norm - 0.5) * 2)
            b = 1 - z_norm
            glColor4f(r, g, b, 0.7)
            glVertex3f(point[0], point[1], point[2])

        glEnd()

    def draw_convex_hull(self):
        """Draw convex hull"""
        hull = self.workspace_data['hull']
        points = self.workspace_data['points']

        if not hull or len(points) < 4:
            return

        # Set transparency
        glColor4f(0.2, 0.6, 1.0, self.transparency)

        # Draw filled faces
        glBegin(GL_TRIANGLES)
        for simplex in hull.simplices:
            for vertex_idx in simplex:
                vertex = points[vertex_idx]
                glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()

        # Draw wireframe
        if self.show_wireframe:
            glColor4f(1, 1, 1, 0.8)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            for simplex in hull.simplices:
                for i in range(3):
                    start_idx = simplex[i]
                    end_idx = simplex[(i + 1) % 3]
                    start = points[start_idx]
                    end = points[end_idx]
                    glVertex3f(start[0], start[1], start[2])
                    glVertex3f(end[0], end[1], end[2])
            glEnd()

    def mousePressEvent(self, event):
        """Handle mouse press for rotation"""
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for rotation"""
        if self.last_pos is None:
            return

        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        self.rotation_y += dx * 0.5
        self.rotation_x += dy * 0.5

        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        self.zoom += delta * 0.1
        self.update()

    def set_transparency(self, value):
        """Set hull transparency"""
        self.transparency = value / 100.0
        self.update()

    def set_point_size(self, value):
        """Set point size"""
        self.point_size = value
        self.update()


class WorkspaceVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.gl_widget = WorkspaceGLWidget()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Robotic Arm Workspace Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # OpenGL widget
        main_layout.addWidget(self.gl_widget, stretch=4)

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_control_panel(self):
        """Create control panel widget"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)

        load_btn = QPushButton("Load Workspace...")
        load_btn.clicked.connect(self.load_workspace)
        file_layout.addWidget(load_btn)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        # Show/hide options
        self.points_checkbox = QCheckBox("Show Points")
        self.points_checkbox.setChecked(True)
        self.points_checkbox.stateChanged.connect(self.toggle_points)
        display_layout.addWidget(self.points_checkbox)

        self.hull_checkbox = QCheckBox("Show Hull")
        self.hull_checkbox.setChecked(True)
        self.hull_checkbox.stateChanged.connect(self.toggle_hull)
        display_layout.addWidget(self.hull_checkbox)

        self.wireframe_checkbox = QCheckBox("Show Wireframe")
        self.wireframe_checkbox.setChecked(True)
        self.wireframe_checkbox.stateChanged.connect(self.toggle_wireframe)
        display_layout.addWidget(self.wireframe_checkbox)

        # Transparency control
        trans_label = QLabel("Hull Transparency:")
        display_layout.addWidget(trans_label)

        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(30)
        self.transparency_slider.valueChanged.connect(self.gl_widget.set_transparency)
        display_layout.addWidget(self.transparency_slider)

        # Point size control
        size_label = QLabel("Point Size:")
        display_layout.addWidget(size_label)

        self.point_size_spinbox = QDoubleSpinBox()
        self.point_size_spinbox.setRange(1.0, 10.0)
        self.point_size_spinbox.setValue(2.0)
        self.point_size_spinbox.setSingleStep(0.5)
        self.point_size_spinbox.valueChanged.connect(
            lambda v: setattr(self.gl_widget, 'point_size', v) or self.gl_widget.update()
        )
        display_layout.addWidget(self.point_size_spinbox)

        # Workspace info group
        self.info_group = QGroupBox("Workspace Info")
        self.info_layout = QVBoxLayout(self.info_group)
        self.info_layout.addWidget(QLabel("No workspace loaded"))

        # Add groups to layout
        layout.addWidget(file_group)
        layout.addWidget(display_group)
        layout.addWidget(self.info_group)
        layout.addStretch()

        return panel

    def load_workspace(self):
        """Load workspace file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Workspace", "", "Pickle Files (*.pkl);;All Files (*)"
        )

        if filename:
            if self.gl_widget.load_workspace(filename):
                self.statusBar().showMessage(f"Loaded: {filename}")
                self.update_workspace_info(filename)
            else:
                self.statusBar().showMessage("Failed to load workspace")

    def update_workspace_info(self, filename):
        """Update workspace information display"""
        try:
            # Clear existing info
            for i in reversed(range(self.info_layout.count())):
                self.info_layout.itemAt(i).widget().setParent(None)

            # Load and display info
            workspace = self.gl_widget.workspace_data
            if workspace:
                info_text = [
                    f"File: {filename.split('/')[-1]}",
                    f"Points: {workspace['num_points']}",
                    f"Vertices: {workspace['num_vertices']}",
                    f"Volume: {workspace['volume']:.2f} mm³"
                ]

                for text in info_text:
                    label = QLabel(text)
                    label.setFont(QFont("Arial", 9))
                    self.info_layout.addWidget(label)

        except Exception as e:
            self.info_layout.addWidget(QLabel(f"Error: {e}"))

    def toggle_points(self, state):
        """Toggle point visibility"""
        self.gl_widget.show_points = (state == Qt.Checked)
        self.gl_widget.update()

    def toggle_hull(self, state):
        """Toggle hull visibility"""
        self.gl_widget.show_hull = (state == Qt.Checked)
        self.gl_widget.update()

    def toggle_wireframe(self, state):
        """Toggle wireframe visibility"""
        self.gl_widget.show_wireframe = (state == Qt.Checked)
        self.gl_widget.update()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = WorkspaceVisualizer()

    # Load file from command line if provided
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if window.gl_widget.load_workspace(filename):
            window.update_workspace_info(filename)
            window.statusBar().showMessage(f"Loaded: {filename}")

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
