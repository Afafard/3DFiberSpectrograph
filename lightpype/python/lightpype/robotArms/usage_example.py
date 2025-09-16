# example_usage.py
# !/usr/bin/env python3
"""
Example usage of the RoArm-M3 scanning system
"""

import time
import logging

from scanning_system import ScanningSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function"""
    # Create scanning system
    system = ScanningSystem('/dev/ttyUSB0', '/dev/ttyUSB1')

    try:
        # Initialize the system
        logger.info("Initializing scanning system...")
        if not system.initialize_system():
            logger.error("Failed to initialize system")
            return

        # Load calibration (or perform new calibration if needed)
        try:
            system.load_calibration()
            logger.info("Calibration loaded successfully")
        except:
            logger.info("No existing calibration found, performing new calibration...")
            system.calibrate_system()
            system.save_calibration()

        # Setup sample
        system.setup_sample()

        # Grasp instruments
        system.grasp_instruments()

        # Plan a simple scan
        logger.info("Planning scan...")
        position_pairs = system.plan_scan(
            r_range=(100, 150),
            theta_range=(0, 0.5),
            phi_range=(0, 3.14),
            r_steps=3,
            theta_steps=3,
            phi_steps=6
        )

        logger.info(f"Planned {len(position_pairs)} scanning positions")

        # Execute scan
        logger.info("Executing scan...")
        system.execute_scan(position_pairs[:5], use_turntable=False)  # Only first 5 positions

        # Release instruments
        system.release_instruments()

        logger.info("Example completed successfully!")

    except Exception as e:
        logger.error(f"Error in example: {e}")
    finally:
        # Cleanup
        system.controller.disconnect()


if __name__ == "__main__":
    main()
