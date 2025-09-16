#!/usr/bin/env python3
"""
3D scanning path planner for dual RoArm-M3 system
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arm import DualArmController
import numpy as np
from typing import List, Tuple
import math


class ScanningPlanner:
    """Path planner for 3D scanning with dual robotic arms"""

    def __init__(self, controller: DualArmController, sample_center: Tuple[float, float, float]):
        """
        Initialize scanning planner

        Args:
            controller: Dual arm controller
            sample_center: (x, y, z) coordinates of sample center
        """
        self.controller = controller
        self.sample_center = np.array(sample_center)
        self.arm_reach = 250  # mm (approximate reach)
        self.safety_margin = 20  # mm safety margin

    def spherical_to_cartesian(self, r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian

        Args:
            r: Radial distance
            theta: Polar angle (from z-axis)
            phi: Azimuthal angle (from x-axis)

        Returns:
            (x, y, z) Cartesian coordinates
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return (x, y, z)

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
            r_range: (min, max) radial distance
            theta_range: (min, max) polar angle (radians)
            phi_range: (min, max) azimuthal angle (radians)
            r_steps, theta_steps, phi_steps: Number of steps in each dimension

        Returns:
            List of (r, theta, phi) positions
        """
        r_min, r_max = r_range
        theta_min, theta_max = theta_range
        phi_min, phi_max = phi_range

        positions = []
        for r in np.linspace(r_min, r_max, r_steps):
            for theta in np.linspace(theta_min, theta_max, theta_steps):
                for phi in np.linspace(phi_min, phi_max, phi_steps):
                    positions.append((r, theta, phi))

        return positions

    def plan_hemispherical_scan(self,
                                r_min: float = 100, r_max: float = 200,
                                r_steps: int = 5,
                                theta_steps: int = 8,
                                phi_steps: int = 16) -> List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Plan a hemispherical scanning pattern for both arms

        Args:
            r_min, r_max: Radial distance range
            r_steps, theta_steps, phi_steps: Sampling resolution

        Returns:
            List of ((arm1_pos), (arm2_pos)) pairs
        """
        # Generate positions for upper hemisphere (theta: 0 to pi/2)
        positions = self.generate_scanning_positions(
            r_range=(r_min, r_max),
            theta_range=(0, np.pi / 2),
            phi_range=(0, 2 * np.pi),
            r_steps=r_steps,
            theta_steps=theta_steps,
            phi_steps=phi_steps
        )

        # Convert to world coordinates around sample
        world_positions = []
        for r, theta, phi in positions:
            # Convert to Cartesian relative to sample center
            dx, dy, dz = self.spherical_to_cartesian(r, theta, phi)
            world_x = self.sample_center[0] + dx
            world_y = self.sample_center[1] + dy
            world_z = self.sample_center[2] + dz
            world_positions.append((world_x, world_y, world_z))

        # For dual arm system, we'll pair positions:
        # Arm 1 covers one hemisphere, Arm 2 covers the other
        # This is a simplified approach - in reality, you'd want to optimize
        # for maximum coverage with minimal collision risk

        pairs = []
        half = len(world_positions) // 2

        # Arm 1 gets first half, Arm 2 gets second half
        for i in range(min(half, len(world_positions) - half)):
            arm1_pos = world_positions[i]
            arm2_pos = world_positions[half + i]
            pairs.append((arm1_pos, arm2_pos))

        return pairs

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

        # Define sample center (in mm)
        sample_center = (0, 0, -200)  # Center of workspace

        # Initialize planner
        planner = ScanningPlanner(controller, sample_center)

        # Plan hemispherical scan
        position_pairs = planner.plan_hemispherical_scan(
            r_min=100, r_max=200,
            r_steps=3,
            theta_steps=4,
            phi_steps=8
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