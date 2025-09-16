#!/usr/bin/env python3
"""
3D scanning path planner for dual RoArm-M3 system with proper spherical coordinates
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arm import DualArmController
import numpy as np
from typing import List, Tuple
import math


class ScanningPlanner:
    """Path planner for 3D scanning with dual robotic arms using spherical coordinates"""

    def __init__(self, controller: DualArmController, sample_center: Tuple[float, float, float],
                 arm_bases: dict = None):
        """
        Initialize scanning planner

        Args:
            controller: Dual arm controller
            sample_center: (x, y, z) coordinates of sample center (origin)
            arm_bases: Dictionary with arm base positions
        """
        self.controller = controller
        self.sample_center = np.array(sample_center)
        self.arm_bases = arm_bases or {
            'left': (-400, 0, 200),  # Left arm base in mm
            'right': (400, 0, 200)  # Right arm base in mm
        }
        self.arm_reach = 500  # mm (approximate reach)
        self.safety_margin = 30  # mm safety margin
        self.min_safe_distance = 150  # mm minimum distance between arms

    def spherical_to_cartesian(self, r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian (sample-centered)

        Args:
            r: Radial distance from sample center (mm)
            theta: Polar angle (from +z axis) in radians
            phi: Azimuthal angle (from +x axis, toward +y) in radians

        Returns:
            (x, y, z) Cartesian coordinates relative to sample center
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return (x, y, z)

    def cartesian_to_spherical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical (sample-centered)

        Args:
            x, y, z: Cartesian coordinates relative to sample center

        Returns:
            (r, theta, phi) spherical coordinates
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r == 0:
            return (0, 0, 0)
        theta = np.arccos(z / r)  # Polar angle from +z axis
        phi = np.arctan2(y, x)  # Azimuthal angle from +x axis
        return (r, theta, phi)

    def generate_scanning_positions(self,
                                    r_range: Tuple[float, float],
                                    theta_range: Tuple[float, float],
                                    phi_range: Tuple[float, float],
                                    r_steps: int = 10,
                                    theta_steps: int = 10,
                                    phi_steps: int = 10) -> List[Tuple[float, float, float]]:
        """
        Generate scanning positions in spherical coordinates

        Args:
            r_range: (min, max) radial distance in mm
            theta_range: (min, max) polar angle in radians (0 to pi)
            phi_range: (min, max) azimuthal angle in radians (0 to 2pi)
            r_steps, theta_steps, phi_steps: Number of steps in each dimension

        Returns:
            List of (r, theta, phi) positions
        """
        r_min, r_max = r_range
        theta_min, theta_max = theta_range
        phi_min, phi_max = phi_range

        # Generate linear spacing for better coverage
        r_values = np.linspace(r_min, r_max, r_steps)
        theta_values = np.linspace(theta_min, theta_max, theta_steps)
        phi_values = np.linspace(phi_min, phi_max, phi_steps)

        positions = []
        for r in r_values:
            for theta in theta_values:
                for phi in phi_values:
                    positions.append((r, theta, phi))

        return positions

    def check_arm_reachability(self, position: Tuple[float, float, float],
                               arm_base: Tuple[float, float, float]) -> bool:
        """
        Check if a position is reachable by an arm

        Args:
            position: (x, y, z) target position relative to sample
            arm_base: (x, y, z) arm base position

        Returns:
            True if position is reachable
        """
        # Convert sample-relative position to absolute coordinates
        target_x = self.sample_center[0] + position[0]
        target_y = self.sample_center[1] + position[1]
        target_z = self.sample_center[2] + position[2]

        # Calculate distance from arm base to target
        distance = np.sqrt(
            (target_x - arm_base[0]) ** 2 +
            (target_y - arm_base[1]) ** 2 +
            (target_z - arm_base[2]) ** 2
        )

        # Check if within reach (with safety margin)
        return distance <= (self.arm_reach - self.safety_margin)

    def check_collision_risk(self, left_pos: Tuple[float, float, float],
                             right_pos: Tuple[float, float, float]) -> bool:
        """
        Check if two arm positions have collision risk

        Args:
            left_pos: (x, y, z) left arm position relative to sample
            right_pos: (x, y, z) right arm position relative to sample

        Returns:
            True if collision risk detected
        """
        # Convert to absolute coordinates
        left_abs = (
            self.sample_center[0] + left_pos[0],
            self.sample_center[1] + left_pos[1],
            self.sample_center[2] + left_pos[2]
        )

        right_abs = (
            self.sample_center[0] + right_pos[0],
            self.sample_center[1] + right_pos[1],
            self.sample_center[2] + right_pos[2]
        )

        # Calculate distance between arm end effectors
        dist = np.sqrt(
            sum((a - b) ** 2 for a, b in zip(left_abs, right_abs))
        )

        # Check collision risk
        if dist < self.min_safe_distance:
            return True

        # Check if arms are crossing the center line (potential collision)
        if (left_abs[1] > 0 and right_abs[1] < 0) or (left_abs[1] < 0 and right_abs[1] > 0):
            # Arms on opposite sides, check x-coordinate overlap
            if abs(left_abs[0] - right_abs[0]) < 200 and abs(left_abs[2] - right_abs[2]) < 100:
                return True

        return False

    def assign_positions_to_arms(self, spherical_positions: List[Tuple[float, float, float]]) -> List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Assign spherical positions to arms with collision avoidance

        Args:
            spherical_positions: List of (r, theta, phi) positions

        Returns:
            List of ((left_pos), (right_pos)) pairs
        """
        # Convert spherical to Cartesian positions relative to sample
        cartesian_positions = [
            self.spherical_to_cartesian(r, theta, phi)
            for r, theta, phi in spherical_positions
        ]

        # Separate positions based on Y coordinate and reachability
        left_assignable = []
        right_assignable = []

        for i, pos in enumerate(cartesian_positions):
            # Check reachability for both arms
            left_reachable = self.check_arm_reachability(pos, self.arm_bases['left'])
            right_reachable = self.check_arm_reachability(pos, self.arm_bases['right'])

            if left_reachable and right_reachable:
                # Both can reach, assign based on Y coordinate
                if pos[1] <= 0:  # Negative Y -> left arm preferred
                    left_assignable.append((i, pos))
                else:  # Positive Y -> right arm preferred
                    right_assignable.append((i, pos))
            elif left_reachable:
                left_assignable.append((i, pos))
            elif right_reachable:
                right_assignable.append((i, pos))

        # Create pairs avoiding collisions
        pairs = []
        used_indices = set()

        # Pair positions trying to minimize collision risk
        for left_idx, left_pos in left_assignable:
            if left_idx in used_indices:
                continue

            best_right = None
            best_distance = float('inf')

            # Find the best matching right position that doesn't collide
            for right_idx, right_pos in right_assignable:
                if right_idx in used_indices:
                    continue

                if not self.check_collision_risk(left_pos, right_pos):
                    # Calculate some metric for good pairing (e.g., distance to optimize workspace)
                    pairing_metric = abs(left_pos[0] - right_pos[0]) + abs(left_pos[2] - right_pos[2])
                    if pairing_metric < best_distance:
                        best_distance = pairing_metric
                        best_right = (right_idx, right_pos)

            if best_right:
                right_idx, right_pos = best_right
                pairs.append((left_pos, right_pos))
                used_indices.add(left_idx)
                used_indices.add(right_idx)

        return pairs

    def plan_hemispherical_scan(self,
                                r_min: float = 80, r_max: float = 250,
                                theta_min: float = 0, theta_max: float = np.pi / 2,
                                phi_min: float = 0, phi_max: float = 2 * np.pi,
                                r_steps: int = 6,
                                theta_steps: int = 5,
                                phi_steps: int = 10) -> List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Plan a hemispherical scanning pattern for both arms with collision avoidance

        Args:
            r_min, r_max: Radial distance range in mm
            theta_min, theta_max: Polar angle range (0 to pi/2 for hemisphere)
            phi_min, phi_max: Azimuthal angle range (0 to 2pi)
            r_steps, theta_steps, phi_steps: Sampling resolution

        Returns:
            List of ((left_arm_pos), (right_arm_pos)) pairs in Cartesian coordinates
        """
        # Generate positions in spherical coordinates
        spherical_positions = self.generate_scanning_positions(
            r_range=(r_min, r_max),
            theta_range=(theta_min, theta_max),
            phi_range=(phi_min, phi_max),
            r_steps=r_steps,
            theta_steps=theta_steps,
            phi_steps=phi_steps
        )

        # Convert to world coordinates and assign to arms
        world_positions = []
        for r, theta, phi in spherical_positions:
            # Convert to Cartesian relative to sample center
            dx, dy, dz = self.spherical_to_cartesian(r, theta, phi)
            world_x = self.sample_center[0] + dx
            world_y = self.sample_center[1] + dy
            world_z = self.sample_center[2] + dz
            world_positions.append((world_x, world_y, world_z))

        # Assign positions to arms with collision avoidance
        # Convert back to sample-relative coordinates for assignment
        sample_positions = [
            (pos[0] - self.sample_center[0],
             pos[1] - self.sample_center[1],
             pos[2] - self.sample_center[2])
            for pos in world_positions
        ]

        # Assign positions to arms
        arm_pairs = self.assign_positions_to_arms(
            [(r, theta, phi) for r, theta, phi in spherical_positions]
        )

        # Convert back to world coordinates for return
        world_pairs = []
        for left_sph, right_sph in arm_pairs:
            # Convert spherical to Cartesian for left arm
            left_cart = self.spherical_to_cartesian(left_sph[0], left_sph[1], left_sph[2])
            left_world = (
                self.sample_center[0] + left_cart[0],
                self.sample_center[1] + left_cart[1],
                self.sample_center[2] + left_cart[2]
            )

            # Convert spherical to Cartesian for right arm
            right_cart = self.spherical_to_cartesian(right_sph[0], right_sph[1], right_sph[2])
            right_world = (
                self.sample_center[0] + right_cart[0],
                self.sample_center[1] + right_cart[1],
                self.sample_center[2] + right_cart[2]
            )

            world_pairs.append((left_world, right_world))

        return world_pairs

    def execute_scan(self, position_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]):
        """
        Execute the scanning sequence

        Args:
            position_pairs: List of ((arm1_pos), (arm2_pos)) pairs
        """
        print(f"Executing scan with {len(position_pairs)} position pairs")

        for i, (arm1_pos, arm2_pos) in enumerate(position_pairs):
            print(f"Moving to position pair {i + 1}/{len(position_pairs)}")
            print(f"  Arm 1: ({arm1_pos[0]:.1f}, {arm1_pos[1]:.1f}, {arm1_pos[2]:.1f})")
            print(f"  Arm 2: ({arm2_pos[0]:.1f}, {arm2_pos[1]:.1f}, {arm2_pos[2]:.1f})")

            # Move both arms simultaneously
            success = self.controller.move_both_to_world(arm1_pos, arm2_pos)

            if success:
                print("  Movement successful")
                # Here you would trigger your spectrometer/light measurement
                # time.sleep(0.5)  # Allow time for measurement
            else:
                print("  Movement failed")

            # Add a small delay between movements
            # time.sleep(0.1)


def main():
    # Initialize controller
    controller = DualArmController('/dev/ttyUSB0', '/dev/ttyUSB1')

    try:
        # Connect to arms
        if not controller.connect():
            print("Failed to connect to arms")
            return

        # Define sample center (in mm) - origin point
        sample_center = (0, 0, 0)

        # Define arm bases
        arm_bases = {
            'left': (-400, 0, 200),  # Spectrometer arm
            'right': (400, 0, 200)  # Illuminator arm
        }

        # Initialize planner
        planner = ScanningPlanner(controller, sample_center, arm_bases)

        # Plan hemispherical scan
        position_pairs = planner.plan_hemispherical_scan(
            r_min=80, r_max=250,
            theta_min=0, theta_max=np.pi / 2,
            phi_min=0, phi_max=2 * np.pi,
            r_steps=4,
            theta_steps=3,
            phi_steps=6
        )

        print(f"Planned {len(position_pairs)} scanning positions")

        # Execute scan (commented out for safety)
        # Uncomment the next line to actually move the arms
        # planner.execute_scan(position_pairs)

        print("Scan planning complete. Uncomment execute_scan() to run.")

    except KeyboardInterrupt:
        print("\nScanning interrupted by user")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
