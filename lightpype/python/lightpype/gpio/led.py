import json
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
from gpiozero import LED
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LEDState(Enum):
    OFF = "off"
    ON = "on"
    BLINKING = "blinking"
    PULSING = "pulsing"
    FADE_IN_OUT = "fade_in_out"


@dataclass
class LEDTask:
    """Represents a task for an LED"""
    name: str
    state: LEDState
    frequency: float = 1.0  # Hz
    duration: Optional[float] = None  # seconds, None for infinite
    priority: int = 0
    callback: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.frequency <= 0:
            self.frequency = 1.0
        if self.kwargs is None:
            self.kwargs = {}


class LEDPattern(Enum):
    """Predefined LED patterns for different states"""
    OFF = "off"
    SOLID_ON = "solid_on"
    SLOW_BLINK = "slow_blink"
    FAST_BLINK = "fast_blink"
    PULSE = "pulse"
    QUICK_PULSE = "quick_pulse"
    FADE_IN_OUT = "fade_in_out"
    ALTERNATING_BLINK = "alternating_blink"
    STROBE = "strobe"
    RAINBOW_CYCLE = "rainbow_cycle"


class LEDManager:
    """Manages multiple LEDs with asynchronous task control and abstracted patterns"""

    def __init__(self, config_file: str = "gpio.json"):
        self.config_file = config_file
        self.leds: Dict[str, LED] = {}
        self.tasks: Dict[str, LEDTask] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_lock = threading.Lock()
        self._state_stack: Dict[str, LEDTask] = {}

        # Load configuration
        self._load_config()

        # Initialize all LEDs to off state
        self._initialize_leds()

    def _load_config(self):
        """Load GPIO configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Load LED mappings
            if 'leds' in config:
                for color, pin in config['leds'].items():
                    self.leds[color] = LED(pin)

            logger.info(f"Loaded configuration from {self.config_file}")

        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_file} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {self.config_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _initialize_leds(self):
        """Initialize all LEDs to off state"""
        for led in self.leds.values():
            led.off()

    def _get_led(self, color: str) -> LED:
        """Get LED object by color name"""
        if color not in self.leds:
            raise ValueError(f"LED '{color}' not found in configuration")
        return self.leds[color]

    def save_state(self, color: str):
        """Save current LED state for later restoration"""
        with self.task_lock:
            if color in self.tasks:
                self._state_stack[color] = self.tasks[color].copy()

    def restore_state(self, color: str):
        """Restore previously saved LED state"""
        with self.task_lock:
            if color in self._state_stack:
                task = self._state_stack[color]
                self.set_led_state(color, task.state, task.frequency,
                                   task.duration, task.priority, task.callback)

    def set_led_state(self, color: str, state: Union[LEDState, LEDPattern],
                      frequency: float = 1.0, duration: Optional[float] = None,
                      priority: int = 0, callback: Optional[Callable] = None,
                      *args, **kwargs) -> bool:
        """
        Set LED state with optional parameters
        Returns True if task was successfully queued
        """
        try:
            led = self._get_led(color)
        except ValueError as e:
            logger.error(f"Error setting LED state: {e}")
            return False

        # Convert pattern to state if needed
        if isinstance(state, LEDPattern):
            state = self._pattern_to_state(state)

        with self.task_lock:
            # Cancel existing task for this LED if present
            if color in self.active_tasks:
                self.active_tasks[color].cancel()
                del self.active_tasks[color]

            # Create new task
            task = LEDTask(
                name=f"{color}_task",
                state=state,
                frequency=frequency,
                duration=duration,
                priority=priority,
                callback=callback,
                args=args,
                kwargs=kwargs
            )

            self.tasks[color] = task

            # Create and schedule the async task
            task_coro = self._led_task(color, task)
            self.active_tasks[color] = asyncio.create_task(task_coro)

            logger.info(f"Set {color} LED to {state.value} state")
            return True

    def _pattern_to_state(self, pattern: LEDPattern) -> LEDState:
        """Convert pattern to corresponding state"""
        pattern_map = {
            LEDPattern.OFF: LEDState.OFF,
            LEDPattern.SOLID_ON: LEDState.ON,
            LEDPattern.SLOW_BLINK: LEDState.BLINKING,
            LEDPattern.FAST_BLINK: LEDState.BLINKING,
            LEDPattern.PULSE: LEDState.PULSING,
            LEDPattern.QUICK_PULSE: LEDState.PULSING,
            LEDPattern.FADE_IN_OUT: LEDState.FADE_IN_OUT,
            LEDPattern.ALTERNATING_BLINK: LEDState.BLINKING,
            LEDPattern.STROBE: LEDState.BLINKING,
            LEDPattern.RAINBOW_CYCLE: LEDState.BLINKING
        }
        return pattern_map.get(pattern, LEDState.OFF)

    async def _led_task(self, color: str, task: LEDTask):
        """Main LED control task"""
        led = self._get_led(color)
        start_time = time.time()

        try:
            while True:
                # Check if task duration has expired
                if task.duration and (time.time() - start_time) > task.duration:
                    break

                # Execute based on state
                if task.state == LEDState.OFF:
                    led.off()
                    await asyncio.sleep(0.1)

                elif task.state == LEDState.ON:
                    led.on()
                    await asyncio.sleep(0.1)

                elif task.state == LEDState.BLINKING:
                    # Blink with specified frequency
                    led.on()
                    await asyncio.sleep(1.0 / (task.frequency * 2))
                    led.off()
                    await asyncio.sleep(1.0 / (task.frequency * 2))

                elif task.state == LEDState.PULSING:
                    # Simple pulse effect
                    led.on()
                    await asyncio.sleep(0.2)
                    led.off()
                    await asyncio.sleep(0.2)

                elif task.state == LEDState.FADE_IN_OUT:
                    # Fade in and out effect
                    led.on()
                    await asyncio.sleep(0.3)
                    led.off()
                    await asyncio.sleep(0.3)

        except asyncio.CancelledError:
            # Task was cancelled, ensure LED is off
            led.off()
            logger.info(f"Task for {color} cancelled")
            raise

    def cancel_led_task(self, color: str) -> bool:
        """Cancel a specific LED task"""
        with self.task_lock:
            if color in self.active_tasks:
                self.active_tasks[color].cancel()
                del self.active_tasks[color]
                if color in self.tasks:
                    del self.tasks[color]
                logger.info(f"Cancelled task for {color}")
                return True
            return False

    def get_led_state(self, color: str) -> LEDState:
        """Get current state of an LED"""
        if color not in self.leds or color not in self.tasks:
            return LEDState.OFF
        return self.tasks[color].state

    def set_led_blink(self, color: str, frequency: float = 1.0,
                      duration: Optional[float] = None) -> bool:
        """Set LED to blink pattern"""
        return self.set_led_state(color, LEDState.BLINKING, frequency, duration)

    def set_led_on(self, color: str, duration: Optional[float] = None) -> bool:
        """Set LED to solid on"""
        return self.set_led_state(color, LEDState.ON, duration=duration)

    def set_led_off(self, color: str) -> bool:
        """Set LED to off"""
        return self.set_led_state(color, LEDState.OFF)

    def set_led_pulse(self, color: str, frequency: float = 1.0,
                      duration: Optional[float] = None) -> bool:
        """Set LED to pulse pattern"""
        return self.set_led_state(color, LEDState.PULSING, frequency, duration)

    async def coordinated_blink(self, colors: List[str], frequency: float = 1.0,
                                duration: Optional[float] = None):
        """Blink multiple LEDs in coordination"""
        tasks = []

        with self.task_lock:
            # Cancel existing tasks for these LEDs
            for color in colors:
                if color in self.active_tasks:
                    self.active_tasks[color].cancel()
                    del self.active_tasks[color]

            # Create new coordinated blink tasks
            for color in colors:
                if color in self.leds:
                    task = LEDTask(
                        name=f"{color}_coord_blink",
                        state=LEDState.BLINKING,
                        frequency=frequency,
                        duration=duration
                    )
                    self.tasks[color] = task
                    task_coro = self._led_task(color, task)
                    tasks.append(asyncio.create_task(task_coro))

        try:
            if duration:
                await asyncio.sleep(duration)
            else:
                # Wait for all tasks to complete (they'll run indefinitely)
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            # Cancel all tasks
            for task in tasks:
                task.cancel()

    # Predefined patterns for different process states

    def set_system_booting(self, duration: Optional[float] = None):
        """System booting pattern"""
        self.set_led_blink("red", frequency=0.5, duration=duration)

    def set_system_ready(self):
        """System ready pattern"""
        self.set_led_on("green")

    def set_system_running(self):
        """System running pattern"""
        self.set_led_pulse("blue", frequency=0.5)

    def set_system_error(self):
        """System error pattern"""
        self.set_led_blink("red", frequency=5.0)

    def set_system_warning(self):
        """System warning pattern"""
        self.set_led_blink("yellow", frequency=2.0)

    def set_process_started(self):
        """Process started pattern"""
        self.set_led_blink("green", frequency=3.0)

    def set_process_running(self):
        """Process running pattern"""
        self.set_led_pulse("blue", frequency=1.0)

    def set_process_completed(self):
        """Process completed pattern"""
        self.set_led_on("green")

    def set_process_failed(self):
        """Process failed pattern"""
        self.set_led_blink("red", frequency=4.0)

    def set_data_processing(self):
        """Data processing pattern"""
        self.set_led_blink("yellow", frequency=2.0)

    def set_data_transmitting(self):
        """Data transmitting pattern"""
        self.set_led_blink("white", frequency=3.0)

    def set_system_idle(self):
        """System idle pattern"""
        self.set_led_off("red")
        self.set_led_off("green")
        self.set_led_off("blue")
        self.set_led_off("yellow")
        self.set_led_off("white")

    def set_system_alert(self):
        """System alert pattern"""
        self.set_led_blink("red", frequency=8.0)

    def set_system_shutdown(self):
        """System shutdown pattern"""
        self.set_led_blink("red", frequency=0.2)

    def set_custom_pattern(self, pattern: LEDPattern, color: str = "red",
                           frequency: float = 1.0):
        """Set a custom predefined pattern"""
        self.set_led_state(color, pattern, frequency)

    def cleanup(self):
        """Clean up all LED tasks and turn off LEDs"""
        with self.task_lock:
            # Cancel all active tasks
            for task in self.active_tasks.values():
                task.cancel()

            # Clear all tasks and active tasks
            self.active_tasks.clear()
            self.tasks.clear()
            self._state_stack.clear()

            # Turn off all LEDs
            for led in self.leds.values():
                led.off()

        logger.info("LED manager cleaned up")


# Decorator for LED state management
def led_state_manager(func):
    """Decorator to manage LED states during function execution"""

    async def wrapper(*args, **kwargs):
        # Get LED manager from first argument if it's an instance
        led_manager = None
        if args and hasattr(args[0], 'led_manager'):
            led_manager = args[0].led_manager
        elif hasattr(func, 'led_manager'):
            led_manager = func.led_manager

        if led_manager:
            # Save current state
            for color in led_manager.leds.keys():
                led_manager.save_state(color)

            # Set starting state
            led_manager.set_process_started()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Restore previous states
            if led_manager:
                for color in led_manager.leds.keys():
                    led_manager.restore_state(color)

    return wrapper


# Example usage and test functions
async def test_led_patterns():
    """Test various LED patterns"""
    led_manager = LEDManager()

    try:
        print("Testing LED patterns...")

        # Test system states
        print("1. System Booting")
        led_manager.set_system_booting()
        await asyncio.sleep(3)

        print("2. System Ready")
        led_manager.set_system_ready()
        await asyncio.sleep(2)

        print("3. System Running")
        led_manager.set_system_running()
        await asyncio.sleep(3)

        print("4. Process Started")
        led_manager.set_process_started()
        await asyncio.sleep(2)

        print("5. Process Running")
        led_manager.set_process_running()
        await asyncio.sleep(3)

        print("6. Process Completed")
        led_manager.set_process_completed()
        await asyncio.sleep(2)

        print("7. Data Processing")
        led_manager.set_data_processing()
        await asyncio.sleep(3)

        print("8. Data Transmitting")
        led_manager.set_data_transmitting()
        await asyncio.sleep(3)

        print("9. System Error")
        led_manager.set_system_error()
        await asyncio.sleep(2)

        print("10. System Warning")
        led_manager.set_system_warning()
        await asyncio.sleep(2)

        print("11. Coordinated Blink")
        await led_manager.coordinated_blink(["red", "green"], frequency=2.0, duration=3.0)

        print("12. Custom Pattern")
        led_manager.set_custom_pattern(LEDPattern.SLOW_BLINK, "blue", 0.5)
        await asyncio.sleep(3)

        print("13. System Idle")
        led_manager.set_system_idle()
        await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        led_manager.cleanup()


# Example class with LED state management
class ExperimentProcessor:
    def __init__(self, led_manager: LEDManager):
        self.led_manager = led_manager
        self.running = False

    @led_state_manager
    async def run_experiment(self, duration: float = 5.0):
        """Run an experiment with LED state management"""
        print("Starting experiment...")
        self.running = True

        # Simulate experiment steps
        await asyncio.sleep(1)
        print("Step 1: Preparing...")
        self.led_manager.set_data_processing()

        await asyncio.sleep(2)
        print("Step 2: Running...")
        self.led_manager.set_system_running()

        await asyncio.sleep(2)
        print("Step 3: Analyzing...")
        self.led_manager.set_data_processing()

        await asyncio.sleep(1)
        print("Experiment completed!")
        self.led_manager.set_process_completed()

        await asyncio.sleep(1)
        self.running = False
        return "Experiment completed successfully"


# Example usage with decorator
async def test_decorator():
    """Test the LED state management decorator"""
    led_manager = LEDManager()
    processor = ExperimentProcessor(led_manager)

    try:
        print("Testing decorator-based LED management...")
        result = await processor.run_experiment(3.0)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        led_manager.cleanup()


if __name__ == "__main__":
    # Run tests
    print("Running LED pattern tests...")
    asyncio.run(test_led_patterns())

    print("\nRunning decorator test...")
    asyncio.run(test_decorator())
