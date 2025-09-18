# test_leds.py
#!/usr/bin/env python3
"""
Test script for LED status indicators
"""

import asyncio
from lightpype.python.lightpype.led_manager import LEDManager


async def test_led_patterns():
    """Test all LED patterns"""
    led_manager = LEDManager(config_file="lightpype/python/lightpype/gpio_control.json")


    try:
        print("Testing LED patterns...")

        # Test error state
        print("1. Error state (red fast blink)")
        led_manager.set_error_state()
        await asyncio.sleep(3)

        # Test path planning state
        print("2. Path planning state (green pulse)")
        led_manager.set_path_planning_state()
        await asyncio.sleep(3)

        # Test moving state
        print("3. Moving state (blue fast blink)")
        led_manager.set_moving_state()
        await asyncio.sleep(3)

        # Test idle state
        print("4. Idle state (green solid)")
        led_manager.set_idle_state()
        await asyncio.sleep(2)

        # Test calibration state
        print("5. Calibration state (blue pulse)")
        led_manager.set_calibration_state()
        await asyncio.sleep(3)

        # Test scanning state
        print("6. Scanning state (blue solid)")
        led_manager.set_scanning_state()
        await asyncio.sleep(2)

        # Test measurement state
        print("7. Measurement state (white on)")
        led_manager.set_measurement_state()
        await asyncio.sleep(2)
        led_manager.clear_measurement_state()

        # Test system ready
        print("8. System ready (green solid)")
        led_manager.set_system_ready()
        await asyncio.sleep(2)

        print("All LED patterns tested successfully!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        led_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_led_patterns())