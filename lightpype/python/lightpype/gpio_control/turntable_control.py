import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import json
import logging
from gpiozero import OutputDevice, Device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorState(Enum):
    STOPPED = "stopped"
    MOVING = "moving"
    CALIBRATED = "calibrated"
    ERROR = "error"


@dataclass
class MotorTask:
    """Represents a task for the motor"""
    name: str
    state: MotorState
    action: str  # 'move_to', 'move_by', 'calibrate', 'stop'
    target: Union[float, int] = 0
    duration: Optional[float] = None  # seconds, None for indefinite
    priority: int = 0
    callback: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class MotorPattern(Enum):
    """Predefined motor patterns for different states"""
    STOPPED = "stopped"
    SLOW_MOVEMENT = "slow_movement"
    FAST_MOVEMENT = "fast_movement"
    SMOOTH_ROTATION = "smooth_rotation"
    QUICK_SEQUENCE = "quick_sequence"
    PIVOT_LEFT = "pivot_left"
    PIVOT_RIGHT = "pivot_right"
    CALIBRATION = "calibration"
    ERROR_SEQUENCE = "error_sequence"


class StepperMotorManager:
    """Manages stepper motor with asynchronous task control and abstracted patterns"""

    def __init__(self, config_file: str = "gpio_control.json"):
        self.config_file = config_file
        self.motor_pins: Dict[str, int] = {}
        self.enable_pins: List[int] = []
        self.steps_per_revolution = 400
        self.current_angle = 0.0
        self.delay = 0.115  # Default delay between steps
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.tasks: Dict[str, MotorTask] = {}
        self.task_lock = threading.Lock()
        self._state_stack: Dict[str, MotorTask] = {}
        self.initialized = False
        self.motor_devices = {}  # Store gpiozero OutputDevice instances

        try:
            # Load configuration
            self._load_config()

            # Setup GPIO pins using gpiozero
            self._setup_gpio()
            self.initialized = True

            # 4-step bipolar full-step sequence
            self.step_seq = [
                [1, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1]
            ]

            # Motor control pins
            self.in1 = self.motor_pins.get('in1', 5)
            self.in2 = self.motor_pins.get('in2', 6)
            self.in3 = self.motor_pins.get('in3', 13)
            self.in4 = self.motor_pins.get('in4', 19)

            logger.info("Stepper motor manager initialized")

        except Exception as e:
            logger.error(f"Error initializing stepper motor manager: {e}")
            self.initialized = False

    def _load_config(self):
        """Load GPIO configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Load motor pin mappings
            if 'motor_pins' in config:
                self.motor_pins = config['motor_pins']

            # Load enable pins (optional)
            if 'enable_pins' in config:
                self.enable_pins = config['enable_pins']

            # Load steps per revolution
            if 'steps_per_revolution' in config:
                self.steps_per_revolution = config['steps_per_revolution']

            # Load default delay
            if 'delay' in config:
                self.delay = config['delay']

            logger.info(f"Loaded motor configuration from {self.config_file}")

        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_file} not found, using defaults")
            # Set default pin configuration
            self.motor_pins = {
                'in1': 5,
                'in2': 6,
                'in3': 13,
                'in4': 19
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {self.config_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _setup_gpio(self):
        """Setup GPIO pins using gpiozero"""
        # Setup motor pins
        for pin_name, pin_number in self.motor_pins.items():
            try:
                self.motor_devices[pin_name] = OutputDevice(pin_number)
                self.motor_devices[pin_name].off()  # Initialize to LOW
            except Exception as e:
                logger.error(f"Error setting up GPIO pin {pin_number}: {e}")
                raise

        # Setup enable pins
        for i, pin_number in enumerate(self.enable_pins):
            try:
                pin_name = f"enable_{i}"
                self.motor_devices[pin_name] = OutputDevice(pin_number)
                self.motor_devices[pin_name].on()  # Set enable pins high (active low in many drivers)
            except Exception as e:
                logger.error(f"Error setting up enable pin {pin_number}: {e}")
                raise

    def set_step(self, a1: int, a2: int, b1: int, b2: int):
        """Set the motor step using gpiozero"""
        try:
            self.motor_devices['in1'].value = a1
            self.motor_devices['in2'].value = a2
            self.motor_devices['in3'].value = b1
            self.motor_devices['in4'].value = b2
        except Exception as e:
            logger.error(f"Error setting motor step: {e}")
            raise

    async def _step_motor(self, steps: int):
        """Execute a specific number of motor steps asynchronously"""
        if steps > 0:
            seq = self.step_seq
        else:
            seq = self.step_seq[::-1]
            steps = -steps

        for _ in range(steps):
            for step in seq:
                self.set_step(*step)
                await asyncio.sleep(self.delay)

    async def move_by_steps(self, steps: int):
        """Move the motor by a specific number of steps"""
        if steps != 0:
            await self._step_motor(steps)
            # Update current angle
            angle_change = (steps / self.steps_per_revolution) * 360.0
            self.current_angle += angle_change

    async def move_to_angle(self, target_angle: float):
        """Move the motor to a specific angle"""
        # Calculate the difference in angle
        angle_diff = target_angle - self.current_angle

        # Normalize angle difference to [-180, 180]
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        # Convert angle difference to steps
        steps = int((angle_diff / 360.0) * self.steps_per_revolution)

        await self.move_by_steps(steps)

    async def move_by_angle(self, angle: float):
        """Move the motor by a specific angle"""
        target_angle = self.current_angle + angle
        await self.move_to_angle(target_angle)

    async def calibrate(self, steps_per_rev: int = 200):
        """Calibrate the motor to set steps per revolution"""
        self.steps_per_revolution = steps_per_rev
        logger.info(f"Motor calibrated to {steps_per_rev} steps per revolution")

    async def get_current_angle(self) -> float:
        """Get the current angular position"""
        return self.current_angle

    async def reset_position(self):
        """Reset the current angle to 0"""
        self.current_angle = 0.0

    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            if self.initialized:
                # Turn off all motor pins
                for pin_name, device in self.motor_devices.items():
                    try:
                        device.off()
                        device.close()
                    except Exception as e:
                        logger.error(f"Error closing device {pin_name}: {e}")

                self.motor_devices.clear()
                self.initialized = False
                logger.info("Motor GPIO cleaned up")
        except Exception as e:
            logger.error(f"Error during motor cleanup: {e}")

    # Task management methods

    def save_state(self, name: str = "default"):
        """Save current motor state for later restoration"""
        with self.task_lock:
            if name in self.tasks:
                self._state_stack[name] = self.tasks[name].copy()

    def restore_state(self, name: str = "default"):
        """Restore previously saved motor state"""
        with self.task_lock:
            if name in self._state_stack:
                task = self._state_stack[name]
                self.set_motor_state(task.state, task.action, task.target,
                                     task.duration, task.priority, task.callback)

    def set_motor_state(self, state: Union[MotorState, MotorPattern],
                        action: str = "move_to", target: Union[float, int] = 0,
                        duration: Optional[float] = None, priority: int = 0,
                        callback: Optional[Callable] = None) -> bool:
        """
        Set motor state with optional parameters
        Returns True if task was successfully queued
        """
        if not self.initialized:
            logger.error("Motor manager not initialized")
            return False

        try:
            # Convert pattern to state if needed
            if isinstance(state, MotorPattern):
                state = self._pattern_to_state(state)

            with self.task_lock:
                # Cancel existing task if present
                if 'motor_task' in self.active_tasks:
                    self.active_tasks['motor_task'].cancel()
                    del self.active_tasks['motor_task']

                # Create new task
                task = MotorTask(
                    name="motor_task",
                    state=state,
                    action=action,
                    target=target,
                    duration=duration,
                    priority=priority,
                    callback=callback
                )

                self.tasks['motor_task'] = task

                # Create and schedule the async task
                task_coro = self._motor_task(task)
                self.active_tasks['motor_task'] = asyncio.create_task(task_coro)

                logger.info(f"Set motor to {state.value} state with action {action}")
                return True
        except Exception as e:
            logger.error(f"Error setting motor state: {e}")
            return False

    def _pattern_to_state(self, pattern: MotorPattern) -> MotorState:
        """Convert pattern to corresponding state"""
        pattern_map = {
            MotorPattern.STOPPED: MotorState.STOPPED,
            MotorPattern.SLOW_MOVEMENT: MotorState.MOVING,
            MotorPattern.FAST_MOVEMENT: MotorState.MOVING,
            MotorPattern.SMOOTH_ROTATION: MotorState.MOVING,
            MotorPattern.QUICK_SEQUENCE: MotorState.MOVING,
            MotorPattern.PIVOT_LEFT: MotorState.MOVING,
            MotorPattern.PIVOT_RIGHT: MotorState.MOVING,
            MotorPattern.CALIBRATION: MotorState.CALIBRATED,
            MotorPattern.ERROR_SEQUENCE: MotorState.ERROR
        }
        return pattern_map.get(pattern, MotorState.STOPPED)

    async def _motor_task(self, task: MotorTask):
        """Main motor control task"""
        start_time = time.time()

        try:
            while True:
                # Check if task duration has expired
                if task.duration and (time.time() - start_time) > task.duration:
                    break

                # Execute based on action
                if task.action == "move_to":
                    await self.move_to_angle(task.target)
                    break  # Move is complete

                elif task.action == "move_by":
                    await self.move_by_angle(task.target)
                    break  # Move is complete

                elif task.action == "calibrate":
                    await self.calibrate(task.target)
                    break  # Calibration is complete

                elif task.action == "stop":
                    # Stop motor (do nothing for now, but could implement halt)
                    await asyncio.sleep(0.1)

                elif task.action == "rotate":
                    # Continuous rotation - this would be implemented differently
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            # Task was cancelled
            logger.info("Motor task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in motor task: {e}")
            # Set error state
            with self.task_lock:
                if 'motor_task' in self.tasks:
                    self.tasks['motor_task'].state = MotorState.ERROR

    def cancel_motor_task(self) -> bool:
        """Cancel the current motor task"""
        with self.task_lock:
            if 'motor_task' in self.active_tasks:
                self.active_tasks['motor_task'].cancel()
                del self.active_tasks['motor_task']
                if 'motor_task' in self.tasks:
                    del self.tasks['motor_task']
                logger.info("Cancelled motor task")
                return True
            return False

    def get_motor_state(self) -> MotorState:
        """Get current motor state"""
        if 'motor_task' not in self.tasks:
            return MotorState.STOPPED
        return self.tasks['motor_task'].state

    # Predefined motor patterns for different process states

    def set_motor_stopped(self):
        """Motor stopped pattern"""
        self.set_motor_state(MotorState.STOPPED, "stop")

    def set_motor_slow_movement(self):
        """Slow movement pattern"""
        self.set_motor_state(MotorPattern.SLOW_MOVEMENT, "move_by", 45.0)

    def set_motor_fast_movement(self):
        """Fast movement pattern"""
        self.set_motor_state(MotorPattern.FAST_MOVEMENT, "move_by", 90.0)

    def set_motor_smooth_rotation(self):
        """Smooth continuous rotation pattern"""
        self.set_motor_state(MotorPattern.SMOOTH_ROTATION, "move_to", 360.0)

    def set_motor_quick_sequence(self):
        """Quick movement sequence pattern"""
        self.set_motor_state(MotorPattern.QUICK_SEQUENCE, "move_by", 180.0)

    def set_motor_pivot_left(self):
        """Pivot left pattern"""
        self.set_motor_state(MotorPattern.PIVOT_LEFT, "move_by", -90.0)

    def set_motor_pivot_right(self):
        """Pivot right pattern"""
        self.set_motor_state(MotorPattern.PIVOT_RIGHT, "move_by", 90.0)

    def set_motor_calibration(self):
        """Calibration pattern"""
        self.set_motor_state(MotorPattern.CALIBRATION, "calibrate", 200)

    def set_motor_error_sequence(self):
        """Error sequence pattern"""
        self.set_motor_state(MotorPattern.ERROR_SEQUENCE, "move_by", 360.0)

    def set_motor_custom_pattern(self, pattern: MotorPattern, target: Union[float, int] = 0):
        """Set a custom predefined pattern"""
        self.set_motor_state(pattern, "move_to", target)

    async def run_calibration_sequence(self):
        """Run a full calibration sequence"""
        print("Starting motor calibration...")
        self.set_motor_calibration()
        await asyncio.sleep(1)

        print("Calibration complete")
        self.set_motor_stopped()

    async def run_error_sequence(self):
        """Run error sequence"""
        print("Motor error sequence")
        self.set_motor_error_sequence()
        await asyncio.sleep(2)

        print("Error sequence complete")
        self.set_motor_stopped()

    async def run_movement_sequence(self):
        """Run a movement sequence"""
        print("Starting movement sequence...")

        self.set_motor_slow_movement()
        await asyncio.sleep(1)

        self.set_motor_fast_movement()
        await asyncio.sleep(1)

        self.set_motor_pivot_left()
        await asyncio.sleep(1)

        self.set_motor_pivot_right()
        await asyncio.sleep(1)

        self.set_motor_stopped()
        print("Movement sequence complete")


# Example usage and test functions
async def test_motor_patterns():
    """Test various motor patterns"""
    motor_manager = StepperMotorManager()

    try:
        print("Testing motor patterns...")

        # Test system states
        print("1. Motor Stopped")
        motor_manager.set_motor_stopped()
        await asyncio.sleep(1)

        print("2. Slow Movement")
        motor_manager.set_motor_slow_movement()
        await asyncio.sleep(2)

        print("3. Fast Movement")
        motor_manager.set_motor_fast_movement()
        await asyncio.sleep(2)

        print("4. Pivot Left")
        motor_manager.set_motor_pivot_left()
        await asyncio.sleep(2)

        print("5. Pivot Right")
        motor_manager.set_motor_pivot_right()
        await asyncio.sleep(2)

        print("6. Quick Sequence")
        motor_manager.set_motor_quick_sequence()
        await asyncio.sleep(2)

        print("7. Calibration")
        motor_manager.set_motor_calibration()
        await asyncio.sleep(2)

        print("8. Error Sequence")
        motor_manager.set_motor_error_sequence()
        await asyncio.sleep(2)

        print("9. Custom Pattern")
        motor_manager.set_motor_custom_pattern(MotorPattern.SLOW_MOVEMENT, 90.0)
        await asyncio.sleep(2)

        print("10. Movement Sequence")
        await motor_manager.run_movement_sequence()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        motor_manager.cleanup()


# Example usage with patterns
async def test_motor_with_patterns():
    """Test motor with predefined patterns"""
    motor_manager = StepperMotorManager()

    try:
        print("Testing motor with patterns...")

        # Run calibration
        await motor_manager.run_calibration_sequence()

        # Run error sequence
        await motor_manager.run_error_sequence()

        # Run movement sequence
        await motor_manager.run_movement_sequence()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        motor_manager.cleanup()


if __name__ == "__main__":
    # Run tests
    print("Running motor pattern tests...")
    asyncio.run(test_motor_patterns())

    print("\nRunning motor with patterns test...")
    asyncio.run(test_motor_with_patterns())
