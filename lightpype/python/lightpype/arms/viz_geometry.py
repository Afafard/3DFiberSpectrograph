#!/usr/bin/env python3
"""
3D Visualization for Dual Arm Spectroscopy Setup
Visualizes the cube geometry, arm positions, and calibration points
Enhanced with detailed 5-DOF arm kinematics and fiber optics modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from typing import List, Tuple, Dict, Any, Optional

# --- GEOMETRIC CONSTANTS ---
CUBE_SIZE = 800  # mm inner dimension
CUBE_HALF = CUBE_SIZE / 2
TURN_TABLE_HEIGHT = 57.15  # mm above cube bottom
TURN_TABLE_RADIUS = 174.625  # mm
ARM_MOUNT_HEIGHT = 165.1  # mm above cube bottom
CROSS_BEAM_HEIGHT = 342.85  # mm above cube bottom

# Approximate arm mounting positions (will be refined through calibration)
LEFT_ARM_X = -374.6  # mm (exact mounting position)
RIGHT_ARM_X = 374.6  # mm (exact mounting position)

# World coordinate system definition
WORLD_ORIGIN = np.array([0, 0, TURN_TABLE_HEIGHT])  # Center of sample

# Arm dimensions (RoArm-M3 Pro specs)
ARM_DIMENSIONS = {
    'base_height': 126.01,
    'arm1_length': 236.82,
    'pivot_length': 30.0,
    'arm2_length': 316.15
}


class ArmKinematicsModel:
    """Handles the kinematic modeling of 5-DOF robotic arms"""

    def __init__(self, arm_type="left", base_position=(0, 0, 0)):
        self.arm_type = arm_type
        self.base_position = np.array(base_position, dtype=float)

        # Arm dimensions (mm) - updated to match RoArm-M3 Pro specs
        self.arm_dimensions = ARM_DIMENSIONS.copy()

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


def create_cube_vertices() -> np.ndarray:
    """Create vertices of the cube"""
    # Outer dimensions including extrusion thickness
    half_width = 423.55  # 847.1mm / 2

    vertices = np.array([
        [-half_width, -half_width, 0],  # 0: bottom-back-left
        [half_width, -half_width, 0],  # 1: bottom-back-right
        [half_width, half_width, 0],  # 2: bottom-front-right
        [-half_width, half_width, 0],  # 3: bottom-front-left
        [-half_width, -half_width, 825.5],  # 4: top-back-left
        [half_width, -half_width, 825.5],  # 5: top-back-right
        [half_width, half_width, 825.5],  # 6: top-front-right
        [-half_width, half_width, 825.5]  # 7: top-front-left
    ])
    return vertices


def create_cube_faces(vertices: np.ndarray) -> List[List[int]]:
    """Define the faces of the cube"""
    faces = [
        [0, 1, 2, 3],  # bottom face
        [4, 5, 6, 7],  # top face
        [0, 1, 5, 4],  # back face
        [2, 3, 7, 6],  # front face
        [0, 3, 7, 4],  # left face
        [1, 2, 6, 5]  # right face
    ]
    return faces


def create_turntable_points() -> Tuple[np.ndarray, np.ndarray]:
    """Create points on the turntable for visualization"""
    # Center point
    center = np.array([[0, 0, TURN_TABLE_HEIGHT]])

    # Edge points (cardinal directions)
    edge_points = np.array([
        [0, TURN_TABLE_RADIUS, TURN_TABLE_HEIGHT],  # front
        [TURN_TABLE_RADIUS, 0, TURN_TABLE_HEIGHT],  # right
        [0, -TURN_TABLE_RADIUS, TURN_TABLE_HEIGHT],  # back
        [-TURN_TABLE_RADIUS, 0, TURN_TABLE_HEIGHT]  # left
    ])

    # Additional points around the turntable perimeter
    perimeter_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    perimeter_points = np.array([
        [TURN_TABLE_RADIUS * np.cos(angle),
         TURN_TABLE_RADIUS * np.sin(angle),
         TURN_TABLE_HEIGHT]
        for angle in perimeter_angles
    ])

    return center, edge_points, perimeter_points


def create_shared_reference_points() -> List[Tuple[str, np.ndarray]]:
    """Create the 7 shared reference points that both arms can reach"""
    shared_points = [
        ("Sample center on turntable", np.array([0, 0, TURN_TABLE_HEIGHT])),
        ("Turntable front edge (Y+)", np.array([0, TURN_TABLE_RADIUS, TURN_TABLE_HEIGHT])),
        ("Turntable back edge (Y-)", np.array([0, -TURN_TABLE_RADIUS, TURN_TABLE_HEIGHT])),
        ("Turntable left edge (X-)", np.array([-TURN_TABLE_RADIUS, 0, TURN_TABLE_HEIGHT])),
        ("Turntable right edge (X+)", np.array([TURN_TABLE_RADIUS, 0, TURN_TABLE_HEIGHT])),
        ("Cross beam front center", np.array([0, CUBE_HALF - 30, CROSS_BEAM_HEIGHT])),
        ("Cross beam back center", np.array([0, -(CUBE_HALF - 30), CROSS_BEAM_HEIGHT]))
    ]
    return shared_points


def plot_arm_model(ax, arm_model, color='blue', alpha=1.0):
    """Plot a robotic arm model"""
    segments = arm_model.get_segments()

    if len(segments) < 2:
        return

    # Draw each segment with different colors for joints
    segment_colors = [
        color,  # Base to shoulder
        plt.cm.Blues(0.7),  # Shoulder to elbow
        plt.cm.Blues(0.5),  # Elbow to wrist1
        plt.cm.Blues(0.3),  # Wrist1 to wrist2
        color  # Wrist2 to end effector
    ]

    # Draw arm segments
    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]

        # Skip zero-length segments (wrist2 rotation)
        if np.allclose(start, end):
            continue

        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=segment_colors[i],
                linewidth=6,
                alpha=alpha)

    # Draw joints as spheres
    for i, point in enumerate(segments):
        base_color = 'darkblue' if color == 'blue' else 'darkviolet'
        ax.scatter(point[0], point[1], point[2],
                   color=base_color if i == 0 else color,
                   s=50, alpha=alpha)

    # Draw end effector
    end_effector = segments[-1]
    ax.scatter(end_effector[0], end_effector[1], end_effector[2],
               color='red', s=100, alpha=alpha, marker='o')

    # Draw fiber optic attachments
    draw_fiber_attachments(ax, end_effector, arm_model.arm_type, alpha)


def draw_fiber_attachments(ax, end_effector, arm_type, alpha=1.0):
    """Draw fiber optic attachments on the end effector"""
    x, y, z = end_effector

    # Spectrometer fiber (left arm) or illuminator fiber (right arm)
    fiber_offset = np.array([0, 0, 20])  # 20mm above end effector
    fiber_pos = end_effector + fiber_offset

    if arm_type == "left":
        color = 'blue'
        label = 'Spectrometer Fiber'
    else:
        color = 'orange'
        label = 'Illuminator Fiber'

    # Draw fiber
    ax.plot([x, fiber_pos[0]],
            [y, fiber_pos[1]],
            [z, fiber_pos[2]],
            color=color, linewidth=3, alpha=alpha)

    # Draw fiber tip
    ax.scatter(fiber_pos[0], fiber_pos[1], fiber_pos[2],
               color=color, s=80, alpha=alpha, marker='^')

    # Add label
    ax.text(fiber_pos[0], fiber_pos[1], fiber_pos[2] + 10,
            label, color=color, fontsize=9, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))


def plot_3d_setup():
    """Create 3D visualization of the setup"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create cube
    cube_vertices = create_cube_vertices()
    cube_faces = create_cube_faces(cube_vertices)

    # Plot cube faces
    cube_verts = [cube_vertices[face] for face in cube_faces]
    cube_collection = Poly3DCollection(cube_verts, alpha=0.1, facecolor='gray', edgecolor='black')
    ax.add_collection3d(cube_collection)

    # Plot cube edges
    for face in cube_faces:
        face_vertices = cube_vertices[face]
        face_vertices = np.append(face_vertices, [face_vertices[0]], axis=0)  # Close the loop
        ax.plot(face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2], 'k-', linewidth=0.5)

    # Draw horizontal structural supports at arm mounting height (wrapping around exterior)
    support_y = np.array([-450, 450])  # Extended beyond cube for visual effect
    support_z = np.array([ARM_MOUNT_HEIGHT, ARM_MOUNT_HEIGHT])

    # Left support (wrapping around exterior)
    left_support_x = np.array([-450, -450])
    ax.plot(left_support_x, support_y, support_z, 'darkgray', linewidth=10, alpha=0.8)

    # Right support (wrapping around exterior)
    right_support_x = np.array([450, 450])
    ax.plot(right_support_x, support_y, support_z, 'darkgray', linewidth=10, alpha=0.8)

    # Front and back vertical supports for cross beams
    beam_x_positions = [-400, 400]  # At cube edges
    beam_z_positions = [0, 825.5]  # From bottom to top

    for x_pos in beam_x_positions:
        # Vertical supports
        ax.plot([x_pos, x_pos], [CUBE_HALF - 20, CUBE_HALF - 20], beam_z_positions, 'brown', linewidth=6)
        ax.plot([x_pos, x_pos], [-(CUBE_HALF - 20), -(CUBE_HALF - 20)], beam_z_positions, 'brown', linewidth=6)

    # Draw cross beams
    # Front cross beam (connecting vertical supports)
    front_beam_x = np.array([-400, 400])
    front_beam_y = np.array([CUBE_HALF - 20, CUBE_HALF - 20])  # Slightly in front of cube face
    front_beam_z = np.array([CROSS_BEAM_HEIGHT, CROSS_BEAM_HEIGHT])
    ax.plot(front_beam_x, front_beam_y, front_beam_z, 'brown', linewidth=8, label='Cross Beams')

    # Back cross beam (connecting vertical supports)
    back_beam_x = np.array([-400, 400])
    back_beam_y = np.array([-(CUBE_HALF - 20), -(CUBE_HALF - 20)])
    back_beam_z = np.array([CROSS_BEAM_HEIGHT, CROSS_BEAM_HEIGHT])
    ax.plot(back_beam_x, back_beam_y, back_beam_z, 'brown', linewidth=8)

    # Create turntable
    center, edge_points, perimeter_points = create_turntable_points()

    # Plot turntable center
    ax.scatter(center[:, 0], center[:, 1], center[:, 2], c='red', s=100, label='Turntable Center')

    # Plot turntable edge points
    ax.scatter(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2],
               c='orange', s=50, label='Turntable Edge Points')

    # Plot turntable perimeter
    ax.scatter(perimeter_points[:, 0], perimeter_points[:, 1], perimeter_points[:, 2],
               c='orange', s=20, alpha=0.7)

    # Draw filled turntable (dark shaded)
    theta = np.linspace(0, 2 * np.pi, 100)
    turntable_x = TURN_TABLE_RADIUS * np.cos(theta)
    turntable_y = TURN_TABLE_RADIUS * np.sin(theta)
    turntable_z = np.full_like(turntable_x, TURN_TABLE_HEIGHT)

    # Create filled turntable surface
    ax.plot_trisurf(turntable_x, turntable_y, turntable_z, color='darkred', alpha=0.7)

    # Draw turntable edge
    ax.plot(turntable_x, turntable_y, turntable_z, 'darkred', linewidth=2)

    # Create shared reference points
    shared_points = create_shared_reference_points()
    shared_coords = np.array([point[1] for point in shared_points])

    # Plot shared reference points with numbers
    for i, (desc, coord) in enumerate(shared_points):
        ax.scatter(coord[0], coord[1], coord[2], c='blue', s=100)
        ax.text(coord[0], coord[1], coord[2], f' {i + 1}. {desc}', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Create arm models
    left_arm_model = ArmKinematicsModel("left", (LEFT_ARM_X, 0, ARM_MOUNT_HEIGHT))
    right_arm_model = ArmKinematicsModel("right", (RIGHT_ARM_X, 0, ARM_MOUNT_HEIGHT))

    # Set neutral joint angles: both arms folded vertically over sample center
    # Left arm (X=-374.6): must reach to (0,0,57.15)
    # Joint angles: base=0°, shoulder=-90° (down), elbow=+90° (fold back), wrist1=-90° (point down)
    left_arm_model.set_joint_angles([0, -np.pi/2, np.pi/2, -np.pi/2, 0])

    # Right arm (X=+374.6): must reach to (0,0,57.15)
    # Same joint configuration — symmetric inward fold
    right_arm_model.set_joint_angles([0, -np.pi/2, np.pi/2, -np.pi/2, 0])

    # Plot arm models
    plot_arm_model(ax, left_arm_model, 'blue', 0.8)
    plot_arm_model(ax, right_arm_model, 'purple', 0.8)

    # Draw arm reach zones — now tight vertical cylinders over sample
    # Left arm: cone centered at (0, 0, 57.15), pointing downward
    center_x = 0
    center_y = 0
    center_z = TURN_TABLE_HEIGHT

    # Cone centered above sample, pointing down — radius 100mm max
    r_max = 120  # mm (conservative reach radius from center)
    h_max = 150  # mm (max vertical drop from arm base)

    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(np.pi/2, np.pi, 15)  # Only downward hemisphere

    # Left arm cone: centered at sample, origin offset by base position
    left_base_x = LEFT_ARM_X
    # Map cone in local space centered at (0,0,THT)
    X = np.outer(np.cos(theta), np.sin(phi)) * r_max
    Y = np.outer(np.sin(theta), np.sin(phi)) * r_max
    Z = np.outer(np.ones_like(theta), np.cos(phi)) * h_max

    # Shift cone to be centered on sample, then offset base position
    X = center_x + X
    Y = center_y + Y
    Z = center_z + Z - h_max  # Start from base height down

    ax.plot_surface(X, Y, Z, alpha=0.12, color='blue', shade=True)

    # Right arm cone — identical
    ax.plot_surface(X, Y, Z, alpha=0.12, color='purple', shade=True)

    # Connect arm bases to vertical supports
    ax.plot([LEFT_ARM_X, LEFT_ARM_X], [0, 0], [ARM_MOUNT_HEIGHT, ARM_MOUNT_HEIGHT + 50], 'darkgray', linewidth=3)
    ax.plot([RIGHT_ARM_X, RIGHT_ARM_X], [0, 0], [ARM_MOUNT_HEIGHT, ARM_MOUNT_HEIGHT + 50], 'darkgray', linewidth=3)

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Dual Arm Spectroscopy Setup - 3D Visualization')

    # Set equal aspect ratio
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_zlim([0, 900])

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

    # Add text annotations for key dimensions
    ax.text(0, 0, -100, f'Cube: {CUBE_SIZE}mm × {CUBE_SIZE}mm × {CUBE_SIZE}mm',
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.text(0, 0, -150, f'Turntable Height: {TURN_TABLE_HEIGHT}mm',
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Show the plot
    plt.tight_layout()
    plt.show()


def print_geometry_summary():
    """Print a summary of the geometry"""
    print("\n" + "=" * 60)
    print("DUAL ARM SPECTROSCOPY SETUP - GEOMETRY SUMMARY")
    print("=" * 60)

    print(f"Cube Dimensions: {CUBE_SIZE}mm × {CUBE_SIZE}mm × {CUBE_SIZE}mm (inner)")
    print(f"Turntable Height: {TURN_TABLE_HEIGHT}mm above cube bottom")
    print(f"Turntable Radius: {TURN_TABLE_RADIUS}mm")
    print(f"Arm Mounting Height: {ARM_MOUNT_HEIGHT}mm above cube bottom")
    print(f"Cross Beam Height: {CROSS_BEAM_HEIGHT}mm above cube bottom")

    print("\nWorld Coordinate System:")
    print("  Origin (0,0,0): Center of sample on turntable")
    print("  X-axis: Left (-) / Right (+)")
    print("  Y-axis: Back (-) / Front (+)")
    print("  Z-axis: Vertical (up +)")

    print(f"\nArm Mounting Positions:")
    print(f"  Spectrometer Arm: X = {LEFT_ARM_X}mm, Y = 0mm, Z = {ARM_MOUNT_HEIGHT}mm")
    print(f"  Illuminator Arm: X = {RIGHT_ARM_X}mm, Y = 0mm, Z = {ARM_MOUNT_HEIGHT}mm")

    print(f"\nArm Specifications (RoArm-M3 Pro):")
    print(f"  Base Height: {ARM_DIMENSIONS['base_height']}mm")
    print(f"  Arm1 Length: {ARM_DIMENSIONS['arm1_length']}mm")
    print(f"  Pivot Length: {ARM_DIMENSIONS['pivot_length']}mm")
    print(f"  Arm2 Length: {ARM_DIMENSIONS['arm2_length']}mm")
    print(f"  Total Reach: {sum(ARM_DIMENSIONS.values())}mm")

    print(f"\nShared Reference Points (7 points both arms can reach):")
    shared_points = create_shared_reference_points()
    for i, (desc, coord) in enumerate(shared_points):
        print(f"  {i + 1:2d}. {desc}: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Print geometry summary
    print_geometry_summary()

    # Create 3D visualization
    print("\nGenerating 3D visualization...")
    plot_3d_setup()

    print("\n3D visualization complete!")
    print("Blue/purple arms: 5-DOF robotic arms with kinematic modeling")
    print("Red end effectors: Arm end points")
    print("Blue/orange pyramids: Spectrometer/illuminator fiber optics")
    print("Blue numbered points: Shared reference points (both arms can reach)")
    print("Red filled circle: Turntable center (world origin)")
    print("Blue/purple transparent surfaces: Arm reach zones (focused vertically over sample)")
    print("Brown beams: Cross beams connecting vertical supports")
    print("Dark gray bars: Horizontal structural supports at arm mounting height")