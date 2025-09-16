# viz.py
# !/usr/bin/env python3
"""
3D visualization for RoArm-M3 scanning system using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import matplotlib.animation as animation
from typing import List, Tuple


class ArmVisualizer:
    """3D visualization for dual arm system"""

    def __init__(self, sample_center=(0, 0, -200), workspace_size=400):
        """
        Initialize visualization

        Args:
            sample_center: (x, y, z) center of sample
            workspace_size: Size of visualization cube
        """
        self.sample_center = np.array(sample_center)
        self.workspace_size = workspace_size

        # Setup figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize plot elements
        self.left_arm_line = None
        self.right_arm_line = None
        self.sample_point = None
        self.workspace_points = []

        self._setup_plot()

    def _setup_plot(self):
        """Setup the initial plot"""
        # Set labels
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('RoArm-M3 Dual Arm 3D Scanner')

        # Set equal aspect ratio
        self.ax.set_xlim([-self.workspace_size / 2, self.workspace_size / 2])
        self.ax.set_ylim([-self.workspace_size / 2, self.workspace_size / 2])
        self.ax.set_zlim([-self.workspace_size, 0])

        # Plot sample center
        self.sample_point = self.ax.scatter(
            [self.sample_center[0]],
            [self.sample_center[1]],
            [self.sample_center[2]],
            c='red', s=100, label='Sample Center'
        )

        # Plot workspace boundaries
        self._plot_workspace_boundaries()

        # Add legend
        self.ax.legend()

    def _plot_workspace_boundaries(self):
        """Plot the workspace boundaries"""
        # Draw a wireframe cube representing the workspace
        r = self.workspace_size / 2
        x = np.array([[-r, -r, -r, -r, -r, -r, -r, -r],
                      [r, r, r, r, r, r, r, r]])
        y = np.array([[-r, -r, -r, -r, r, r, r, r],
                      [-r, -r, r, r, -r, -r, r, r]])
        z = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [-2 * r, -2 * r, -2 * r, -2 * r, -2 * r, -2 * r, -2 * r, -2 * r]])

        # Plot the cube edges
        for i in range(8):
            self.ax.plot(x[:, i], y[:, i], z[:, i], 'k--', alpha=0.3)

        # Plot the base
        base_x = [-r, r, r, -r, -r]
        base_y = [-r, -r, r, r, -r]
        base_z = [0, 0, 0, 0, 0]
        self.ax.plot(base_x, base_y, base_z, 'k-', alpha=0.5)

        # Plot the top
        top_z = [-2 * r, -2 * r, -2 * r, -2 * r, -2 * r]
        self.ax.plot(base_x, base_y, top_z, 'k-', alpha=0.5)

    def update_arm_positions(self, left_pos: Tuple[float, float, float],
                             right_pos: Tuple[float, float, float]):
        """
        Update the arm positions in the visualization

        Args:
            left_pos: (x, y, z) position of left arm end effector
            right_pos: (x, y, z) position of right arm end effector
        """
        # Clear previous arm lines
        if self.left_arm_line:
            self.left_arm_line.remove()
        if self.right_arm_line:
            self.right_arm_line.remove()

        # Plot left arm (from base to end effector)
        left_base = [0, -150, 0]  # Approximate left arm base position
        left_x = [left_base[0], left_pos[0]]
        left_y = [left_base[1], left_pos[1]]
        left_z = [left_base[2], left_pos[2]]
        self.left_arm_line = self.ax.plot(left_x, left_y, left_z, 'b-', linewidth=3, label='Left Arm')[0]

        # Plot right arm (from base to end effector)
        right_base = [0, 150, 0]  # Approximate right arm base position
        right_x = [right_base[0], right_pos[0]]
        right_y = [right_base[1], right_pos[1]]
        right_z = [right_base[2], right_pos[2]]
        self.right_arm_line = self.ax.plot(right_x, right_y, right_z, 'g-', linewidth=3, label='Right Arm')[0]

        # Update legend
        self.ax.legend()

        # Refresh the plot
        plt.draw()

    def plot_scanning_path(self, positions: List[Tuple[float, float, float]],
                           arm_id: str = 'left'):
        """
        Plot a scanning path

        Args:
            positions: List of (x, y, z) positions
            arm_id: 'left' or 'right' to specify which arm
        """
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

    def show(self):
        """Display the visualization"""
        plt.show()

    def save_figure(self, filename: str):
        """Save the current figure"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')


# Example usage
if __name__ == "__main__":
    # Create visualizer
    viz = ArmVisualizer()

    # Example arm positions
    left_pos = (100, -100, -150)
    right_pos = (100, 100, -150)

    # Update arm positions
    viz.update_arm_positions(left_pos, right_pos)

    # Show the visualization
    viz.show()
