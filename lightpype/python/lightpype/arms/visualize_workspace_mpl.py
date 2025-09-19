#!/usr/bin/env python3
"""
3D Visualization of calibrated robotic arm workspace using matplotlib.

Usage:
    python visualize_workspace_matplotlib.py dual_calibration.pkl
"""

import sys
import numpy as np
from scipy.spatial import ConvexHull
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, CheckButtons, Slider
import argparse


class WorkspaceVisualizer:
    def __init__(self):
        self.workspace_data = None
        self.fig = None
        self.ax = None
        self.point_collection = None
        self.hull_collection = None
        self.hull_wireframe = None
        self.show_points = True
        self.show_hull = True
        self.show_wireframe = True
        self.transparency = 0.3
        self.point_size = 20

    def load_workspace(self, filename):
        """Load workspace data from file"""
        try:
            with open(filename, 'rb') as f:
                self.workspace_data = pickle.load(f)
            print(f"Loaded workspace from {filename}")
            return True
        except Exception as e:
            print(f"Error loading workspace: {e}")
            return False

    def create_plot(self):
        """Create the matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('Robotic Arm Workspace Visualization')

        # Set equal aspect ratio
        if self.workspace_data:
            points = self.workspace_data['points']
            x_range = np.ptp(points[:, 0])
            y_range = np.ptp(points[:, 1])
            z_range = np.ptp(points[:, 2])
            max_range = max(x_range, y_range, z_range)

            x_center = np.mean(points[:, 0])
            y_center = np.mean(points[:, 1])
            z_center = np.mean(points[:, 2])

            self.ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
            self.ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
            self.ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)

        plt.tight_layout()

    def update_plot(self):
        """Update the plot with current settings"""
        if not self.workspace_data:
            return

        # Clear current collections
        if self.point_collection:
            self.point_collection.remove()
        if self.hull_collection:
            self.hull_collection.remove()
        if self.hull_wireframe:
            self.hull_wireframe.remove()

        points = self.workspace_data['points']
        hull = self.workspace_data['hull']

        # Plot points
        if self.show_points and len(points) > 0:
            # Height-based colormap (blue to red)
            z_values = points[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            colors = plt.cm.plasma((z_values - z_min) / (z_max - z_min)) if z_max > z_min else 'blue'

            self.point_collection = self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=self.point_size, alpha=0.7, depthshade=True
            )
        else:
            self.point_collection = None

        # Plot convex hull
        if self.show_hull and hull and len(points) >= 4:
            # Create hull faces
            hull_points = points[hull.vertices]

            # Plot filled hull with transparency
            faces = []
            for simplex in hull.simplices:
                triangle = points[simplex]
                faces.append(triangle)

            # Plot hull surface
            if faces:
                faces = np.array(faces)
                # Plot each triangle face
                for face in faces:
                    # Create a triangular surface
                    X = face[:, 0]
                    Y = face[:, 1]
                    Z = face[:, 2]
                    self.hull_collection = self.ax.plot_trisurf(
                        X, Y, Z, alpha=self.transparency, color='lightblue', edgecolor='none'
                    )

            # Plot wireframe
            if self.show_wireframe:
                # Plot edges
                edges = []
                for simplex in hull.simplices:
                    for i in range(3):
                        start_idx = simplex[i]
                        end_idx = simplex[(i + 1) % 3]
                        start = points[start_idx]
                        end = points[end_idx]
                        edges.append([start, end])

                # Plot all edges
                for edge in edges:
                    edge = np.array(edge)
                    self.hull_wireframe = self.ax.plot(
                        edge[:, 0], edge[:, 1], edge[:, 2],
                        color='white', alpha=0.8, linewidth=1
                    )[0]
            else:
                self.hull_wireframe = None
        else:
            self.hull_collection = None
            self.hull_wireframe = None

        # Add coordinate system indicator
        self.add_coordinate_indicator()

        self.fig.canvas.draw_idle()

    def add_coordinate_indicator(self):
        """Add coordinate system indicator"""
        # Find a corner for the coordinate system
        if self.workspace_data and len(self.workspace_data['points']) > 0:
            points = self.workspace_data['points']
            x_min, y_min, z_min = np.min(points, axis=0)
            x_max, y_max, z_max = np.max(points, axis=0)

            # Position indicator at a corner
            x_pos = x_min - (x_max - x_min) * 0.1
            y_pos = y_min - (y_max - y_min) * 0.1
            z_pos = z_min - (z_max - z_min) * 0.1

            # X axis (red)
            self.ax.quiver(x_pos, y_pos, z_pos, 20, 0, 0, color='red', arrow_length_ratio=0.1)
            # Y axis (green)
            self.ax.quiver(x_pos, y_pos, z_pos, 0, 20, 0, color='green', arrow_length_ratio=0.1)
            # Z axis (blue)
            self.ax.quiver(x_pos, y_pos, z_pos, 0, 0, 20, color='blue', arrow_length_ratio=0.1)

            # Labels
            self.ax.text(x_pos + 25, y_pos, z_pos, 'X', color='red', fontsize=10)
            self.ax.text(x_pos, y_pos + 25, z_pos, 'Y', color='green', fontsize=10)
            self.ax.text(x_pos, y_pos, z_pos + 25, 'Z', color='blue', fontsize=10)

    def toggle_points(self, label):
        """Toggle point visibility"""
        self.show_points = not self.show_points
        self.update_plot()

    def toggle_hull(self, label):
        """Toggle hull visibility"""
        self.show_hull = not self.show_hull
        self.update_plot()

    def toggle_wireframe(self, label):
        """Toggle wireframe visibility"""
        self.show_wireframe = not self.show_wireframe
        self.update_plot()

    def update_transparency(self, val):
        """Update hull transparency"""
        self.transparency = val
        self.update_plot()

    def update_point_size(self, val):
        """Update point size"""
        self.point_size = val
        self.update_plot()

    def add_controls(self):
        """Add interactive controls to the plot"""
        # Control panel area
        control_height = 0.3
        control_ax = plt.axes([0.02, 0.02, 0.2, control_height], facecolor='lightgoldenrodyellow')
        control_ax.set_title('Controls')
        control_ax.axis('off')

        # Buttons
        button_ax1 = plt.axes([0.05, 0.2, 0.08, 0.04])
        button_points = Button(button_ax1, 'Toggle Points')
        button_points.on_clicked(self.toggle_points)

        button_ax2 = plt.axes([0.15, 0.2, 0.08, 0.04])
        button_hull = Button(button_ax2, 'Toggle Hull')
        button_hull.on_clicked(self.toggle_hull)

        button_ax3 = plt.axes([0.05, 0.15, 0.18, 0.04])
        button_wireframe = Button(button_ax3, 'Toggle Wireframe')
        button_wireframe.on_clicked(self.toggle_wireframe)

        # Sliders
        slider_ax1 = plt.axes([0.05, 0.1, 0.18, 0.03])
        slider_transparency = Slider(slider_ax1, 'Transparency', 0, 1, valinit=self.transparency)
        slider_transparency.on_changed(self.update_transparency)

        slider_ax2 = plt.axes([0.05, 0.05, 0.18, 0.03])
        slider_pointsize = Slider(slider_ax2, 'Point Size', 1, 100, valinit=self.point_size)
        slider_pointsize.on_changed(self.update_point_size)

        # Info display
        if self.workspace_data:
            info_text = f"Points: {self.workspace_data['num_points']}\n"
            info_text += f"Vertices: {self.workspace_data['num_vertices']}\n"
            info_text += f"Volume: {self.workspace_data['volume']:.2f} mm³"

            self.fig.text(0.8, 0.02, info_text, fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def show(self, filename=None):
        """Show the visualization"""
        if filename:
            if not self.load_workspace(filename):
                return

        self.create_plot()
        self.update_plot()
        self.add_controls()

        plt.show()


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Visualize robotic arm workspace using matplotlib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_workspace_matplotlib.py dual_calibration.pkl
        """
    )

    parser.add_argument("workspace_file", nargs='?',default='dual_calibration.pkl',
                        help="Workspace file to visualize (e.g., dual_calibration.pkl)")

    args = parser.parse_args()

    # Create visualizer
    visualizer = WorkspaceVisualizer()

    # Show visualization

    visualizer.show(args.workspace_file)


if __name__ == "__main__":
    main()
