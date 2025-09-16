# viz.py
# !/usr/bin/env python3
"""
3D visualization for RoArm-M3 scanning system using PyQt5/PyQtGraph
"""

import sys
import numpy as np
from typing import List, Tuple
import logging

# Try to import PyQt5 and pyqtgraph
try:
    from PyQt5 import QtWidgets, QtCore
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt5 or pyqtgraph not available. Using matplotlib fallback.")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArmVisualizer:
    """3D visualization for dual arm system using Qt/PyQtGraph"""

    def __init__(self, sample_center=(0, 0, 0), arm_bases=None, workspace_size=600):
        """
        Initialize visualization

        Args:
            sample_center: (x, y, z) center of sample
            arm_bases: Dictionary with arm base positions
            workspace_size: Size of visualization cube in mm
        """
        self.sample_center = np.array(sample_center)
        self.arm_bases = arm_bases or {
            'left': (-400, 0, 200),
            'right': (400, 0, 200)
        }
        self.workspace_size = workspace_size
        self.app = None
        self.window = None
        self.plot_widget = None
        self.scatter_plot = None
        self.arm_lines = {}
        self.path_plots = {}

        # If PyQt is available, use it; otherwise fall back to matplotlib
        if PYQT_AVAILABLE:
            self.use_pyqt = True
            self._setup_pyqt_visualization()
        else:
            self.use_pyqt = False
            self._setup_matplotlib_visualization()

    def _setup_pyqt_visualization(self):
        """Setup PyQt-based visualization"""
        try:
            # Create Qt application
            self.app = QtWidgets.QApplication.instance()
            if self.app is None:
                self.app = QtWidgets.QApplication(sys.argv)

            # Create main window
            self.window = QtWidgets.QMainWindow()
            self.window.setWindowTitle('RoArm-M3 Dual Arm 3D Scanner')
            self.window.setGeometry(100, 100, 1000, 800)

            # Create central widget and layout
            central_widget = QtWidgets.QWidget()
            self.window.setCentralWidget(central_widget)
            layout = QtWidgets.QVBoxLayout(central_widget)

            # Create 3D plot widget
            self.plot_widget = gl.GLViewWidget()
            self.plot_widget.opts['distance'] = 800
            layout.addWidget(self.plot_widget)

            # Add coordinate axes
            self._add_coordinate_axes()

            # Add sample center
            self._add_sample_center()

            # Add arm bases
            self._add_arm_bases()

            # Add workspace boundaries
            self._add_workspace_boundaries()

            logger.info("PyQt visualization initialized successfully")

        except Exception as e:
            logger.error(f"Error setting up PyQt visualization: {e}")
            self.use_pyqt = False
            self._setup_matplotlib_visualization()

    def _setup_matplotlib_visualization(self):
        """Setup matplotlib-based visualization as fallback"""
        logger.info("Using matplotlib fallback visualization")
        try:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._setup_matplotlib_plot()
        except Exception as e:
            logger.error(f"Error setting up matplotlib visualization: {e}")

    def _setup_matplotlib_plot(self):
        """Setup the initial matplotlib plot"""
        if hasattr(self, 'ax'):
            # Set labels
            self.ax.set_xlabel('X (mm)')
            self.ax.set_ylabel('Y (mm)')
            self.ax.set_zlabel('Z (mm)')
            self.ax.set_title('RoArm-M3 Dual Arm 3D Scanner')

            # Set equal aspect ratio
            self.ax.set_xlim([-self.workspace_size / 2, self.workspace_size / 2])
            self.ax.set_ylim([-self.workspace_size / 2, self.workspace_size / 2])
            self.ax.set_zlim([0, self.workspace_size / 2])

            # Plot sample center
            self.ax.scatter([self.sample_center[0]], [self.sample_center[1]], [self.sample_center[2]],
                            c='red', s=100, label='Sample Center')

            # Plot arm bases
            for arm_name, base_pos in self.arm_bases.items():
                color = 'blue' if arm_name == 'left' else 'green'
                self.ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]],
                                c=color, s=50, label=f'{arm_name.capitalize()} Arm Base')

            # Plot workspace boundaries
            self._plot_matplotlib_workspace_boundaries()

            # Add legend
            self.ax.legend()

    def _plot_matplotlib_workspace_boundaries(self):
        """Plot the workspace boundaries in matplotlib"""
        # Draw a wireframe cube representing the workspace
        r = self.workspace_size / 2
        x = [-r, r, r, -r, -r, -r, -r, -r, -r, -r, r, r, r, r, r, r]
        y = [-r, -r, -r, -r, -r, -r, r, r, -r, -r, -r, -r, r, r, r, r]
        z = [0, 0, 0, 0, 0, r, r, 0, 0, r, r, r, r, 0, 0, r]

        # Plot the cube edges
        for i in range(0, len(x), 2):
            if i + 1 < len(x):
                self.ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], 'k--', alpha=0.3)

    def _add_coordinate_axes(self):
        """Add coordinate axes to the 3D plot"""
        # X axis (red)
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [200, 0, 0]]), color=(1, 0, 0, 1), width=3)
        self.plot_widget.addItem(x_axis)

        # Y axis (green)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 200, 0]]), color=(0, 1, 0, 1), width=3)
        self.plot_widget.addItem(y_axis)

        # Z axis (blue)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 200]]), color=(0, 0, 1, 1), width=3)
        self.plot_widget.addItem(z_axis)

    def _add_sample_center(self):
        """Add sample center to the 3D plot"""
        sample_point = gl.GLScatterPlotItem(pos=np.array([self.sample_center]), color=(1, 0, 0, 1), size=10)
        self.plot_widget.addItem(sample_point)

    def _add_arm_bases(self):
        """Add arm base positions to the 3D plot"""
        base_positions = np.array(list(self.arm_bases.values()))
        base_colors = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # Blue for left, green for right
        base_points = gl.GLScatterPlotItem(pos=base_positions, color=base_colors, size=8)
        self.plot_widget.addItem(base_points)

    def _add_workspace_boundaries(self):
        """Add workspace boundaries to the 3D plot"""
        # Create a wireframe cube
        vertices = np.array([
            [-300, -300, 0], [300, -300, 0], [300, 300, 0], [-300, 300, 0],  # Bottom face
            [-300, -300, 300], [300, -300, 300], [300, 300, 300], [-300, 300, 300]  # Top face
        ])

        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges
        ]

        # Create line segments
        for edge in edges:
            pts = vertices[[edge[0], edge[1]]]
            line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 0.5), width=1)
            self.plot_widget.addItem(line)

    def update_arm_positions(self, left_pos: Tuple[float, float, float],
                             right_pos: Tuple[float, float, float]):
        """
        Update the arm positions in the visualization

        Args:
            left_pos: (x, y, z) position of left arm end effector
            right_pos: (x, y, z) position of right arm end effector
        """
        if self.use_pyqt:
            self._update_pyqt_arm_positions(left_pos, right_pos)
        else:
            self._update_matplotlib_arm_positions(left_pos, right_pos)

    def _update_pyqt_arm_positions(self, left_pos: Tuple[float, float, float],
                                   right_pos: Tuple[float, float, float]):
        """Update arm positions in PyQt visualization"""
        try:
            # Remove existing arm lines
            for arm_name in ['left', 'right']:
                if arm_name in self.arm_lines:
                    self.plot_widget.removeItem(self.arm_lines[arm_name])

            # Plot left arm (from base to end effector)
            left_base = self.arm_bases['left']
            left_line_data = np.array([left_base, left_pos])
            left_line = gl.GLLinePlotItem(pos=left_line_data, color=(0, 0, 1, 1), width=3)
            self.plot_widget.addItem(left_line)
            self.arm_lines['left'] = left_line

            # Plot right arm (from base to end effector)
            right_base = self.arm_bases['right']
            right_line_data = np.array([right_base, right_pos])
            right_line = gl.GLLinePlotItem(pos=right_line_data, color=(0, 1, 0, 1), width=3)
            self.plot_widget.addItem(right_line)
            self.arm_lines['right'] = right_line

        except Exception as e:
            logger.error(f"Error updating PyQt arm positions: {e}")

    def _update_matplotlib_arm_positions(self, left_pos: Tuple[float, float, float],
                                         right_pos: Tuple[float, float, float]):
        """Update arm positions in matplotlib visualization"""
        try:
            # Plot left arm (from base to end effector)
            left_base = self.arm_bases['left']
            left_x = [left_base[0], left_pos[0]]
            left_y = [left_base[1], left_pos[1]]
            left_z = [left_base[2], left_pos[2]]
            self.ax.plot(left_x, left_y, left_z, 'b-', linewidth=3, label='Left Arm')

            # Plot right arm (from base to end effector)
            right_base = self.arm_bases['right']
            right_x = [right_base[0], right_pos[0]]
            right_y = [right_base[1], right_pos[1]]
            right_z = [right_base[2], right_pos[2]]
            self.ax.plot(right_x, right_y, right_z, 'g-', linewidth=3, label='Right Arm')

            # Refresh the plot
            plt.draw()

        except Exception as e:
            logger.error(f"Error updating matplotlib arm positions: {e}")

    def plot_scanning_path(self, positions: List[Tuple[float, float, float]],
                           arm_id: str = 'left'):
        """
        Plot a scanning path

        Args:
            positions: List of (x, y, z) positions
            arm_id: 'left' or 'right' to specify which arm
        """
        if self.use_pyqt:
            self._plot_pyqt_scanning_path(positions, arm_id)
        else:
            self._plot_matplotlib_scanning_path(positions, arm_id)

    def _plot_pyqt_scanning_path(self, positions: List[Tuple[float, float, float]],
                                 arm_id: str = 'left'):
        """Plot scanning path in PyQt visualization"""
        try:
            if not positions:
                return

            # Remove existing path for this arm
            if arm_id in self.path_plots:
                self.plot_widget.removeItem(self.path_plots[arm_id])

            # Convert positions to numpy array
            path_data = np.array(positions)

            # Set color based on arm
            color = (0, 0, 1, 0.7) if arm_id == 'left' else (0, 1, 0, 0.7)

            # Create path plot
            path_plot = gl.GLLinePlotItem(pos=path_data, color=color, width=1)
            self.plot_widget.addItem(path_plot)
            self.path_plots[arm_id] = path_plot

        except Exception as e:
            logger.error(f"Error plotting PyQt scanning path: {e}")

    def _plot_matplotlib_scanning_path(self, positions: List[Tuple[float, float, float]],
                                       arm_id: str = 'left'):
        """Plot scanning path in matplotlib visualization"""
        try:
            if not positions:
                return

            x_vals = [pos[0] for pos in positions]
            y_vals = [pos[1] for pos in positions]
            z_vals = [pos[2] for pos in positions]

            color = 'blue' if arm_id == 'left' else 'green'
            label = f'{arm_id.capitalize()} Arm Path'

            self.ax.plot(x_vals, y_vals, z_vals, color=color, alpha=0.7, linewidth=1, label=label)
            self.ax.scatter(x_vals, y_vals, z_vals, color=color, s=20, alpha=0.6)

            # Update legend
            self.ax.legend()

            # Refresh the plot
            plt.draw()

        except Exception as e:
            logger.error(f"Error plotting matplotlib scanning path: {e}")

    def highlight_current_position(self, left_pos: Tuple[float, float, float],
                                   right_pos: Tuple[float, float, float]):
        """
        Highlight the current scanning positions

        Args:
            left_pos: Current left arm position
            right_pos: Current right arm position
        """
        if self.use_pyqt:
            self._highlight_pyqt_current_position(left_pos, right_pos)
        else:
            self._highlight_matplotlib_current_position(left_pos, right_pos)

    def _highlight_pyqt_current_position(self, left_pos: Tuple[float, float, float],
                                         right_pos: Tuple[float, float, float]):
        """Highlight current position in PyQt visualization"""
        try:
            # Create highlight points
            highlight_points = gl.GLScatterPlotItem(
                pos=np.array([left_pos, right_pos]),
                color=np.array([[1, 0, 0, 1], [1, 1, 0, 1]]),  # Red for left, yellow for right
                size=15
            )
            self.plot_widget.addItem(highlight_points)

            # Remove after a short delay (this would need proper Qt timer handling)

        except Exception as e:
            logger.error(f"Error highlighting PyQt current position: {e}")

    def _highlight_matplotlib_current_position(self, left_pos: Tuple[float, float, float],
                                               right_pos: Tuple[float, float, float]):
        """Highlight current position in matplotlib visualization"""
        try:
            # Plot current positions
            self.ax.scatter([left_pos[0]], [left_pos[1]], [left_pos[2]],
                            c='red', s=100, alpha=0.8)
            self.ax.scatter([right_pos[0]], [right_pos[1]], [right_pos[2]],
                            c='yellow', s=100, alpha=0.8)
            plt.draw()

        except Exception as e:
            logger.error(f"Error highlighting matplotlib current position: {e}")

    def clear_paths(self):
        """Clear all scanning paths from visualization"""
        if self.use_pyqt:
            self._clear_pyqt_paths()
        else:
            self._clear_matplotlib_paths()

    def _clear_pyqt_paths(self):
        """Clear paths in PyQt visualization"""
        try:
            for arm_id in list(self.path_plots.keys()):
                self.plot_widget.removeItem(self.path_plots[arm_id])
            self.path_plots.clear()
        except Exception as e:
            logger.error(f"Error clearing PyQt paths: {e}")

    def _clear_matplotlib_paths(self):
        """Clear paths in matplotlib visualization"""
        try:
            # Clear all plotted lines and scatter points except the main elements
            pass  # In matplotlib, this would require more sophisticated tracking
        except Exception as e:
            logger.error(f"Error clearing matplotlib paths: {e}")

    def show(self):
        """Display the visualization"""
        if self.use_pyqt:
            self._show_pyqt()
        else:
            self._show_matplotlib()

    def _show_pyqt(self):
        """Show PyQt visualization"""
        try:
            if self.window:
                self.window.show()
                if self.app:
                    self.app.exec_()
        except Exception as e:
            logger.error(f"Error showing PyQt visualization: {e}")

    def _show_matplotlib(self):
        """Show matplotlib visualization"""
        try:
            plt.show()
        except Exception as e:
            logger.error(f"Error showing matplotlib visualization: {e}")

    def save_figure(self, filename: str):
        """Save the current figure"""
        if self.use_pyqt:
            logger.warning("Save figure not implemented for PyQt visualization")
        else:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            except Exception as e:
                logger.error(f"Error saving matplotlib figure: {e}")


# Example usage
if __name__ == "__main__":
    # Create visualizer
    viz = ArmVisualizer()

    # Example arm positions
    left_pos = (200, -100, 100)
    right_pos = (200, 100, 100)

    # Update arm positions
    viz.update_arm_positions(left_pos, right_pos)

    # Show the visualization
    viz.show()
