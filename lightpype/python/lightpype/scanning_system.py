# scanning_system.py (updated)
# !/usr/bin/env python3
"""
Complete 3D scanning system for dual RoArm-M3 robotic arms with proper geometry and LED status indicators
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arm import DualArmController, ScanningPlanner
from viz import ArmVisualizer
from turntable_control import StepperMotorManager
from lightpype.python.lightpype.gpio_control.led_manager import LEDManager  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScanningSystem:
    """Complete 3D scanning system with dual arms, turntable, and LED status indicators"""

    def __init__(self, left_port: str = '/dev/ttyUSB0', right_port: str = '/dev/ttyUSB1'):
        """
        Initialize the scanning system with proper geometry

        Args:
            left_port: Serial port for left arm (spectrometer)
            right_port: Serial port for right arm (illuminator)
        """
        self.controller = DualArmController(left_port, right_port)
        self.motor_manager = StepperMotorManager()
        self.led_manager = LEDManager()  # Add LED manager
        self.visualizer = None
        self.sample_center = (0, 0, 0)  # Sample is our origin
        self.arm_bases = {
            'left': (-400, 0, 200),  # Left arm base (spectrometer) in mm
            'right': (400, 0, 200)  # Right arm base (illuminator) in mm
        }
        self.planner = None
        self.is_calibrated = False
        self.scan_data = []
        self.arm_workspace_radius = 500  # mm

    def initialize_system(self) -> bool:
        """
        Initialize all system components

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Set initializing state
            self.led_manager.set_initializing_state()

            # Connect to arms
            if not self.controller.connect():
                logger.error("Failed to connect to robotic arms")
                self.led_manager.set_error_state()
                return False

            # Initialize planner
            self.planner = ScanningPlanner(self.controller, self.sample_center, self.arm_bases)

            # Initialize visualizer
            self.visualizer = ArmVisualizer(self.sample_center, self.arm_bases)

            # Set system ready
            self.led_manager.set_system_ready()
            logger.info("System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            self.led_manager.set_error_state()
            return False

    def calibrate_system(self, calibration_points: List[Tuple[float, float, float]] = None):
        """
        Calibrate the entire system with proper geometry

        Args:
            calibration_points: List of calibration points, if None uses default
        """
        # Set calibration state
        self.led_manager.set_calibration_state()

        if calibration_points is None:
            # Default calibration points around sample center (hemispherical pattern)
            calibration_points = [
                (0, 0, 100),  # Above sample
                (100, 0, 0),  # Right of sample
                (-100, 0, 0),  # Left of sample
                (0, 100, 0),  # Front of sample
                (0, -100, 0),  # Back of sample
                (70, 70, 50),  # Diagonal front-right
                (-70, 70, 50),  # Diagonal front-left
                (70, -70, 50),  # Diagonal back-right
                (-70, -70, 50),  # Diagonal back-left
                (0, 0, 200),  # High above sample
            ]

        try:
            logger.info("Starting system calibration...")
            self.controller.calibrate_arms(calibration_points)
            self.is_calibrated = True
            logger.info("System calibration completed successfully")

        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            self.led_manager.set_error_state()
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
            self.led_manager.set_error_state()
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
            self.led_manager.set_error_state()
            raise

    def setup_sample(self):
        """Setup the sample for scanning"""
        try:
            # Set idle state
            self.led_manager.set_idle_state()

            # Move arms to safe positions away from sample
            self.controller.move_left_to_world(300, -200, 100)  # Left arm safe position
            self.controller.move_right_to_world(300, 200, 100)  # Right arm safe position

            # Open grippers for sample mounting
            self.controller.release_with_left()
            self.controller.release_with_right()

            logger.info("Sample setup complete - mount your sample now")
            input("Press Enter when sample is mounted and ready...")

        except Exception as e:
            logger.error(f"Error setting up sample: {e}")
            self.led_manager.set_error_state()
            raise

    def grasp_instruments(self):
        """Grasp the illuminator and spectrometer"""
        try:
            logger.info("Grasping instruments...")
            # Set moving state
            self.led_manager.set_moving_state()

            # Move to instrument pickup positions
            self.controller.move_left_to_world(200, -300, 50)  # Spectrometer pickup
            self.controller.move_right_to_world(200, 300, 50)  # Illuminator pickup

            time.sleep(2)  # Wait for positioning

            # Grasp instruments
            self.controller.grasp_with_left()  # Spectrometer
            self.controller.grasp_with_right()  # Illuminator

            time.sleep(1)  # Allow grippers to close

            # Move to safe positions above sample
            self.controller.move_left_to_world(200, -100, 150)
            self.controller.move_right_to_world(200, 100, 150)

            logger.info("Instruments grasped successfully")

        except Exception as e:
            logger.error(f"Error grasping instruments: {e}")
            self.led_manager.set_error_state()
            raise

    def release_instruments(self):
        """Release the illuminator and spectrometer"""
        try:
            logger.info("Releasing instruments...")
            # Set moving state
            self.led_manager.set_moving_state()

            # Move to release positions
            self.controller.move_left_to_world(200, -300, 50)  # Spectrometer release
            self.controller.move_right_to_world(200, 300, 50)  # Illuminator release

            time.sleep(1)  # Wait for positioning

            # Open grippers
            self.controller.release_with_left()  # Spectrometer
            self.controller.release_with_right()  # Illuminator

            time.sleep(1)  # Allow grippers to open

            # Move arms to safe positions
            self.controller.move_left_to_world(300, -200, 100)
            self.controller.move_right_to_world(300, 200, 100)

            logger.info("Instruments released successfully")

        except Exception as e:
            logger.error(f"Error releasing instruments: {e}")
            self.led_manager.set_error_state()
            raise

    def plan_scan(self,
                  r_range: Tuple[float, float] = (50, 300),
                  theta_range: Tuple[float, float] = (0, np.pi / 2),  # Hemisphere
                  phi_range: Tuple[float, float] = (0, 2 * np.pi),
                  r_steps: int = 8,
                  theta_steps: int = 6,
                  phi_steps: int = 12) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Plan a scanning sequence with proper hemispherical coverage

        Args:
            r_range: Radial distance range in mm
            theta_range: Polar angle range (0 to pi/2 for hemisphere)
            phi_range: Azimuthal angle range (0 to 2pi for full circle)
            r_steps, theta_steps, phi_steps: Sampling resolution

        Returns:
            List of position pairs for both arms
        """
        if not self.planner:
            raise RuntimeError("System not initialized - call initialize_system() first")

        # Set path planning state
        self.led_manager.set_path_planning_state()

        result = self.planner.plan_hemispherical_scan(
            r_min=r_range[0], r_max=r_range[1],
            theta_min=theta_range[0], theta_max=theta_range[1],
            phi_min=phi_range[0], phi_max=phi_range[1],
            r_steps=r_steps,
            theta_steps=theta_steps,
            phi_steps=phi_steps
        )

        # Set idle state after planning
        self.led_manager.set_idle_state()
        return result

    def visualize_plan(self, position_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]):
        """
        Visualize the planned scanning path in real-time

        Args:
            position_pairs: List of position pairs to visualize
        """
        if not self.visualizer:
            logger.warning("No visualizer available")
            return

        logger.info(f"Visualizing {len(position_pairs)} planned positions...")

        # Plot the scanning path
        left_positions = [pair[0] for pair in position_pairs]
        right_positions = [pair[1] for pair in position_pairs]

        self.visualizer.plot_scanning_path(left_positions, 'left')
        self.visualizer.plot_scanning_path(right_positions, 'right')

        # Show the visualization
        self.visualizer.show()

    def execute_scan(self, position_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
                     use_turntable: bool = True, turntable_steps: int = 12):
        """
        Execute a complete scanning sequence with real-time visualization

        Args:
            position_pairs: List of position pairs for arms
            use_turntable: Whether to use turntable for additional coverage
            turntable_steps: Number of turntable positions
        """
        if not self.is_calibrated:
            logger.warning("System not calibrated - results may be inaccurate")

        try:
            logger.info(f"Starting scan with {len(position_pairs)} position pairs")
            # Set scanning state
            self.led_manager.set_scanning_state()

            # Update visualization with planned path
            if self.visualizer:
                self.visualizer.clear_paths()
                left_positions = [pair[0] for pair in position_pairs]
                right_positions = [pair[1] for pair in position_pairs]
                self.visualizer.plot_scanning_path(left_positions, 'left')
                self.visualizer.plot_scanning_path(right_positions, 'right')

            # If using turntable, divide positions among turntable steps
            if use_turntable:
                positions_per_step = max(1, len(position_pairs) // turntable_steps)
                actual_turntable_steps = min(turntable_steps, len(position_pairs))

                for step in range(actual_turntable_steps):
                    # Rotate turntable
                    angle = (360.0 / actual_turntable_steps) * step
                    logger.info(f"Rotating turntable to {angle:.1f} degrees")

                    if hasattr(self.motor_manager, 'set_motor_state'):
                        self.motor_manager.set_motor_state("moving", "move_to", angle)
                        time.sleep(2)  # Wait for rotation

                    # Execute positions for this turntable step
                    start_idx = step * positions_per_step
                    end_idx = min((step + 1) * positions_per_step, len(position_pairs))

                    if start_idx >= len(position_pairs):
                        break

                    for i in range(start_idx, end_idx):
                        arm1_pos, arm2_pos = position_pairs[i]
                        logger.info(
                            f"Executing position pair {i + 1}/{len(position_pairs)} at turntable angle {angle:.1f}°")

                        # Set moving state
                        self.led_manager.set_moving_state()

                        # Move arms to positions
                        success = self.controller.move_both_to_world(arm1_pos, arm2_pos)

                        if success:
                            logger.info("  Movement successful")
                            # Set measurement state
                            self.led_manager.set_measurement_state()

                            # Here you would trigger your spectrometer/light measurement
                            # time.sleep(0.5)  # Allow time for measurement

                            # Clear measurement state
                            self.led_manager.clear_measurement_state()

                            # Update visualization if available
                            if self.visualizer:
                                self.visualizer.update_arm_positions(arm1_pos, arm2_pos)
                                self.visualizer.highlight_current_position(arm1_pos, arm2_pos)
                        else:
                            logger.warning("  Movement failed")
                            self.led_manager.set_error_state()

                        time.sleep(0.1)  # Small delay between movements

            else:
                # Execute without turntable
                self.planner.execute_scan(position_pairs)

            logger.info("Scan completed successfully")
            # Set idle state
            self.led_manager.set_idle_state()

        except Exception as e:
            logger.error(f"Error during scan execution: {e}")
            self.led_manager.set_error_state()
            raise

    def run_complete_scan(self, save_calibration: bool = True, visualize: bool = True):
        """
        Run a complete scanning workflow

        Args:
            save_calibration: Whether to save calibration after completion
            visualize: Whether to show real-time visualization
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

            # Plan scan - hemispherical coverage
            position_pairs = self.plan_scan(
                r_range=(80, 250),  # Safe range within arm reach
                theta_range=(0, np.pi / 2),  # Hemisphere
                phi_range=(0, 2 * np.pi),  # Full azimuthal coverage
                r_steps=6,
                theta_steps=5,
                phi_steps=10
            )

            logger.info(f"Planned {len(position_pairs)} scanning positions")

            # Visualize plan if requested
            if visualize:
                self.visualize_plan(position_pairs)

            # Execute scan
            self.execute_scan(position_pairs, use_turntable=True, turntable_steps=12)

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
            if hasattr(self, 'led_manager'):
                self.led_manager.cleanup()

    def cleanup(self):
        """Clean up all system components"""
        if hasattr(self, 'controller'):
            self.controller.disconnect()
        if hasattr(self, 'motor_manager'):
            self.motor_manager.cleanup()
        if hasattr(self, 'led_manager'):
            self.led_manager.cleanup()
