# led_manager.py
"""
LED status manager for the 3D scanning system
"""

import json
import asyncio
import time
from typing import Dict, Optional
from gpiozero import LED
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LEDState(Enum):
    OFF = "off"
    ON = "on"
    BLINKING = "blinking"
    PULSING = "pulsing"


class LEDPattern(Enum):
    """Predefined LED patterns for different system states"""
    OFF = "off"
    SOLID_ON = "solid_on"
    SLOW_BLINK = "slow_blink"
    FAST_BLINK = "fast_blink"
    PULSE = "pulse"
    QUICK_PULSE = "quick_pulse"


class LEDManager:
    """Manages LED status indicators for the 3D scanning system"""

    def __init__(self, config_file: str = "gpio_control.json"):
        self.config_file = config_file
        self.leds: Dict[str, LED] = {}
        self.led_tasks: Dict[str, asyncio.Task] = {}
        self._running = False  # <-- New flag to control lifecycle
        self._load_config()
        self._initialize_leds()
        logger.info("LED Manager initialized")

    def _load_config(self):
        """Load GPIO configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Load LED mappings
            if 'leds' in config:
                for color, pin in config['leds'].items():
                    self.leds[color] = LED(pin)

            logger.info(f"Loaded LED configuration from {self.config_file}")

        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_file} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {self.config_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading LED configuration: {e}")
            raise

    def _initialize_leds(self):
        """Initialize all LEDs to off state"""
        for led in self.leds.values():
            led.off()

    async def _led_task(self, color: str, state: LEDState, frequency: float = 1.0, duration: Optional[float] = None):
        """Main LED control task"""
        led = self.leds.get(color)
        if not led:
            logger.error(f"LED '{color}' not found")
            return

        start_time = time.time()

        try:
            while True:
                # Check if manager is still running — exit early if not
                if not self._running:
                    logger.debug(f"LED task for {color} exiting due to manager shutdown")
                    break

                # Check if task duration has expired
                if duration and (time.time() - start_time) > duration:
                    break

                # Execute based on state — only if manager is running
                if not self._running:
                    break

                if state == LEDState.OFF:
                    led.off()
                    await asyncio.sleep(0.1)

                elif state == LEDState.ON:
                    led.on()
                    await asyncio.sleep(0.1)

                elif state == LEDState.BLINKING:
                    # Blink with specified frequency
                    led.on()
                    await asyncio.sleep(1.0 / (frequency * 2))
                    if not self._running:
                        break
                    led.off()
                    await asyncio.sleep(1.0 / (frequency * 2))

                elif state == LEDState.PULSING:
                    # Simple pulse effect
                    led.on()
                    await asyncio.sleep(0.2)
                    if not self._running:
                        break
                    led.off()
                    await asyncio.sleep(0.2)

        except asyncio.CancelledError:
            # Only attempt to turn off if still running
            if self._running:
                led.off()
            logger.info(f"LED task for {color} cancelled")
            raise

    def set_led_state(self, color: str, state: LEDState, frequency: float = 1.0, duration: Optional[float] = None):
        """Set LED state with optional parameters"""
        # Cancel existing task for this LED if present
        if color in self.led_tasks:
            self.led_tasks[color].cancel()

        # Create and schedule the async task
        task_coro = self._led_task(color, state, frequency, duration)
        self.led_tasks[color] = asyncio.create_task(task_coro)

    def set_error_state(self):
        """Set red LED for error state (fast blinking)"""
        self.set_led_state("red", LEDState.BLINKING, frequency=5.0)

    def set_path_planning_state(self):
        """Set green LED for path planning (slow pulsing)"""
        self.set_led_state("green", LEDState.PULSING, frequency=0.5)

    def set_moving_state(self):
        """Set blue LED for motors moving (fast blinking)"""
        self.set_led_state("blue", LEDState.BLINKING, frequency=3.0)

    def set_idle_state(self):
        """Set system idle state (green solid)"""
        self.set_led_state("green", LEDState.ON)
        self.set_led_state("red", LEDState.OFF)
        self.set_led_state("blue", LEDState.OFF)

    def set_initializing_state(self):
        """Set system initializing state (all LEDs slow blinking)"""
        self.set_led_state("red", LEDState.BLINKING, frequency=0.5)
        self.set_led_state("green", LEDState.BLINKING, frequency=0.5)
        self.set_led_state("blue", LEDState.BLINKING, frequency=0.5)

    def set_calibration_state(self):
        """Set calibration state (blue pulsing)"""
        self.set_led_state("blue", LEDState.PULSING, frequency=1.0)

    def set_scanning_state(self):
        """Set scanning in progress state (blue solid)"""
        self.set_led_state("blue", LEDState.ON)

    def set_measurement_state(self):
        """Set measurement state (white LED on)"""
        self.set_led_state("white", LEDState.ON)

    def clear_measurement_state(self):
        """Turn off measurement LED"""
        self.set_led_state("white", LEDState.OFF)

    def set_system_ready(self):
        """Set system ready state (green solid)"""
        self.set_led_state("green", LEDState.ON)
        self.set_led_state("red", LEDState.OFF)
        self.set_led_state("blue", LEDState.OFF)
        self.set_led_state("white", LEDState.OFF)

    def set_all_off(self):
        """Turn off all LEDs"""
        for color in self.leds.keys():
            self.set_led_state(color, LEDState.OFF)

    def cleanup(self):
        """Clean up all LED tasks"""
        # Set flag to stop all tasks from accessing GPIO
        self._running = False

        # Cancel all active tasks
        for task in self.led_tasks.values():
            task.cancel()

        # Wait a moment to let tasks exit gracefully
        if self.led_tasks:
            asyncio.run_coroutine_threadsafe(
                asyncio.gather(*self.led_tasks.values(), return_exceptions=True),
                asyncio.get_event_loop()
            )
            time.sleep(0.1)  # Brief pause to allow cleanup

        # Now it's safe to turn off LEDs — they won't be accessed by tasks anymore
        for led in self.leds.values():
            try:
                led.off()
            except Exception as e:
                logger.warning(f"Failed to turn off LED: {e}")

        self.led_tasks.clear()
        logger.info("LED Manager cleaned up")

    def start(self):
        """Start the LED manager (call this after initialization)"""
        self._running = True