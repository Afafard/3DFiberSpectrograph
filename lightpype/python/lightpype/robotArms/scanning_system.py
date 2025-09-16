# scanning_system.py
# !/usr/bin/env python3
"""
Complete 3D scanning system for dual RoArm-M3 robotic arms
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arm import DualArmController, ScanningPlanner, ArmState
from viz import ArmVisualizer
from ..gpio.turntable_control import StepperMotorManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScanningSystem:
    """Complete 3D scanning system with dual arms and turntable"""

    def __init__(self, left_port: str = '/dev/ttyUSB0', right_port: str = '/dev/ttyUSB1'):
        """
        Initialize the scanning system

        Args:
            left_port: Serial port for left arm
            right_port: Serial port for right arm
        """
        self.controller = DualArmController(left_port, right_port)
        self.motor_manager = StepperMotorManager()
        self.visualizer = None
        self.sample_center = (0, 0, -200)  # Default sample center
        self.planner = None
        self.is_calibrated = False
        self.scan_data = []

    def initialize_system(self) -> bool:
        """
        Initialize all system components

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Connect to arms
            if not self.controller.connect():
                logger.error("Failed to connect to robotic arms")
                return False

            # Initialize planner
            self.planner = ScanningPlanner(self.controller, self.sample_center)

            # Initialize visualizer
            self.visualizer = ArmVisualizer(self.sample_center)

            logger.info("System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False

    def calibrate_system(self, calibration_points: List[Tuple[float, float, float]] = None):
        """
        Calibrate the entire system

        Args:
            calibration_points: List of calibration points, if None uses default
        """
        if calibration_points is None:
            # Default calibration points around sample center
            calibration_points = [
                (0, 0, -150),  # Above sample
                (50, 0, -200),  # Right of sample
                (-50, 0, -200),  # Left of sample
                (0, 50, -200),  # Front of sample
                (0, -50, -200),  # Back of sample
                (30, 30, -180),  # Diagonal
                (-30, -30, -220),  # Opposite diagonal
            ]

        try:
            logger.info("Starting system calibration...")
            self.controller.calibrate_arms(calibration_points)
            self.is_calibrated = True
            logger.info("System calibration completed successfully")

        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            raise

    def load_calibration(self, filename: str = "calibration.pkl"):
        """
        Load calibration from file

        Args:
            filename: Calibration file name
        """
        try:
            self.controller.load_calibration(filename)
            self.is_calibrated = True
            logger.info(f"Calibration loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            raise

    def save_calibration(self, filename: str = "calibration.pkl"):
        """
        Save calibration to file

        Args:
            filename: Calibration file name
        """
        try:
            self.controller.save_calibration(filename)
            logger.info(f"Calibration saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            raise

    def setup_sample(self):
        """Setup the sample for scanning"""
        try:
            # Move arms to safe positions
            self.controller.move_left_to_world(200, -100, -150)
            self.controller.move_right_to_world(200, 100, -150)

            # Open grippers for sample mounting
            self.controller.release_with_left()
            self.controller.release_with_right()

            logger.info("Sample setup complete - mount your sample now")
            input("Press Enter when sample is mounted and ready...")

        except Exception as e:
            logger.error(f"Error setting up sample: {e}")
            raise

    def grasp_instruments(self):
        """Grasp the illuminator and spectrometer"""
        try:
            logger.info("Grasping instruments...")

            # Move to instrument positions (these would be calibrated positions)
            # For now, using approximate positions
            self.controller.move_left_to_world(150, -150, -180)  # Illuminator position
            self.controller.move_right_to_world(150, 150, -180)  # Spectrometer position

            time.sleep(2)  # Wait for positioning

            # Grasp instruments
            self.controller.grasp_with_left()  # Illuminator
            self.controller.grasp_with_right()  # Spectrometer

            time.sleep(1)  # Allow grippers to close

            logger.info("Instruments grasped successfully")

        except Exception as e:
            logger.error(f"Error grasping instruments: {e}")
            raise

    def release_instruments(self):
        """Release the illuminator and spectrometer"""
        try:
            logger.info("Releasing instruments...")

            # Open grippers
            self.controller.release_with_left()  # Illuminator
            self.controller.release_with_right()  # Spectrometer

            time.sleep(1)  # Allow grippers to open

            # Move arms to safe positions
            self.controller.move_left_to_world(200, -100, -150)
            self.controller.move_right_to_world(200, 100, -150)

            logger.info("Instruments released successfully")

        except Exception as e:
            logger.error(f"Error releasing instruments: {e}")
            raise

    def plan_scan(self,
                  r_range: Tuple[float, float] = (100, 200),
                  theta_range: Tuple[float, float] = (0, np.pi / 2),
                  phi_range: Tuple[float, float] = (0, 2 * np.pi),
                  r_steps: int = 5,
                  theta_steps: int = 8,
                  phi_steps: int = 16) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Plan a scanning sequence

        Args:
            r_range: Radial distance range
            theta_range: Polar angle range
            phi_range: Azimuthal angle range
            r_steps, theta_steps, phi_steps: Sampling resolution

        Returns:
            List of position pairs for both arms
        """
        if not self.planner:
            raise RuntimeError("System not initialized - call initialize_system() first")

        return self.planner.plan_hemispherical_scan(
            r_min=r_range[0], r_max=r_range[1],
            r_steps=r_steps,
            theta_steps=theta_steps,
            phi_steps=phi_steps
        )

    def execute_scan(self, position_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
                     use_turntable: bool = True, turntable_steps: int = 8):
        """
        Execute a complete scanning sequence

        Args:
            position_pairs: List of position pairs for arms
            use_turntable: Whether to use turntable for additional coverage
            turntable_steps: Number of turntable positions
        """
        if not self.is_calibrated:
            logger.warning("System not calibrated - results may be inaccurate")

        try:
            logger.info(f"Starting scan with {len(position_pairs)} position pairs")

            # If using turntable, divide positions among turntable steps
            if use_turntable:
                positions_per_step = len(position_pairs) // turntable_steps
                if positions_per_step == 0:
                    positions_per_step = 1
                    turntable_steps = len(position_pairs)

                for step in range(turntable_steps):
                    # Rotate turntable
                    angle = (360.0 / turntable_steps) * step
                    logger.info(f"Rotating turntable to {angle} degrees")
                    self.motor_manager.set_motor_state("moving", "move_to", angle)
                    time.sleep(2)  # Wait for rotation

                    # Execute positions for this turntable step
                    start_idx = step * positions_per_step
                    end_idx = min((step + 1) * positions_per_step, len(position_pairs))

                    for i in range(start_idx, end_idx):
                        arm1_pos, arm2_pos = position_pairs[i]
                        logger.info(f"Executing position pair {i + 1}/{len(position_pairs)} at turntable angle {angle}")

                        # Move arms to positions
                        success = self.controller.move_both_to_world(arm1_pos, arm2_pos)

                        if success:
                            logger.info("  Movement successful")
                            # Here you would trigger your spectrometer/light measurement
                            # time.sleep(0.5)  # Allow time for measurement

                            # Update visualization if available
                            if self.visualizer:
                                self.visualizer.update_arm_positions(arm1_pos, arm2_pos)
                        else:
                            logger.warning("  Movement failed")

                        time.sleep(0.1)  # Small delay between movements

            else:
                # Execute without turntable
                self.planner.execute_scan(position_pairs)

            logger.info("Scan completed successfully")

        except Exception as e:
            logger.error(f"Error during scan execution: {e}")
            raise

    def run_complete_scan(self, save_calibration: bool = True):
        """
        Run a complete scanning workflow

        Args:
            save_calibration: Whether to save calibration after completion
        """
        try:
            # Initialize system
            if not self.initialize_system():
                raise RuntimeError("Failed to initialize system")

            # Load existing calibration or perform new calibration
            try:
                self.load_calibration()
                logger.info("Using existing calibration")
            except:
                logger.info("No existing calibration found, performing new calibration")
                self.calibrate_system()
                if save_calibration:
                    self.save_calibration()

            # Setup sample
            self.setup_sample()

            # Grasp instruments
            self.grasp_instruments()

            # Plan scan
            position_pairs = self.plan_scan(
                r_range=(80, 180),
                theta_range=(0, np.pi / 2),
                phi_range=(0, 2 * np.pi),
                r_steps=4,
                theta_steps=6,
                phi_steps=12
            )

            logger.info(f"Planned {len(position_pairs)} scanning positions")

            # Execute scan
            self.execute_scan(position_pairs, use_turntable=True, turntable_steps=8)

            # Release instruments
            self.release_instruments()

            logger.info("Complete scanning workflow finished successfully")

        except Exception as e:
            logger.error(f"Error in complete scan workflow: {e}")
            # Ensure cleanup even if error occurs
            try:
                self.release_instruments()
            except:
                pass
            raise

        finally:
            # Cleanup
            if hasattr(self, 'controller'):
                self.controller.disconnect()
            if hasattr(self, 'motor_manager'):
                self.motor_manager.cleanup()


# Example usage and test functions
def test_basic_movement():
    """Test basic arm movements"""
    system = ScanningSystem()

    try:
        if system.initialize_system():
            # Test movements
            logger.info("Testing basic movements...")

            # Move to some test positions
            system.controller.move_left_to_world(200, -50, -200)
            system.controller.move_right_to_world(200, 50, -200)

            time.sleep(2)

            # Test grippers
            system.controller.grasp_with_left()
            time.sleep(1)
            system.controller.release_with_left()

            system.controller.grasp_with_right()
            time.sleep(1)
            system.controller.release_with_right()

            logger.info("Basic movement test completed")

    except Exception as e:
        logger.error(f"Error in basic movement test: {e}")
    finally:
        system.controller.disconnect()


def test_calibration():
    """Test calibration process"""
    system = ScanningSystem()

    try:
        if system.initialize_system():
            # Perform calibration
            system.calibrate_system()

            # Save calibration
            system.save_calibration("test_calibration.pkl")

            logger.info("Calibration test completed")

    except Exception as e:
        logger.error(f"Error in calibration test: {e}")
    finally:
        system.controller.disconnect()


def test_full_scan():
    """Test complete scanning workflow"""
    system = ScanningSystem()

    try:
        system.run_complete_scan(save_calibration=True)
        logger.info("Full scan test completed")

    except Exception as e:
        logger.error(f"Error in full scan test: {e}")


if __name__ == "__main__":
    # Run tests based on command line arguments
    import argparse

    parser = argparse.ArgumentParser(description='RoArm-M3 Scanning System')
    parser.add_argument('--test', choices=['movement', 'calibration', 'full'],
                        help='Run specific test')
    parser.add_argument('--scan', action='store_true', help='Run complete scan')

    args = parser.parse_args()

    if args.test == 'movement':
        test_basic_movement()
    elif args.test == 'calibration':
        test_calibration()
    elif args.test == 'full':
        test_full_scan()
    elif args.scan:
        system = ScanningSystem()
        system.run_complete_scan()
    else:
        # Default: show help
        parser.print_help()
