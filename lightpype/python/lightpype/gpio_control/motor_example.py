import RPi.GPIO as GPIO
import asyncio
import time
from typing import Optional, Tuple

# GPIO pin definitions for coils
IN1 = 5  # Coil A+
IN2 = 6  # Coil A-
IN3 = 13  # Coil B+
IN4 = 19  # Coil B-

EN_PINS = []  # Removed: previously defined enable pins


class StepperMotor:
    def __init__(self):
        self.steps_per_revolution = 200  # Standard for most stepper motors
        self.current_angle = 0.0
        self.delay = 0.005  # Default delay between steps

        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)

        motor_pins = [IN1, IN2, IN3, IN4]
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # 4-step bipolar full-step sequence
        self.step_seq = [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1]
        ]

        # Motor control pins
        self.in1 = IN1
        self.in2 = IN2
        self.in3 = IN3
        self.in4 = IN4

    def set_step(self, a1: int, a2: int, b1: int, b2: int):
        """Set the motor step"""
        GPIO.output(self.in1, a1)
        GPIO.output(self.in2, a2)
        GPIO.output(self.in3, b1)
        GPIO.output(self.in4, b2)

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
        print(f"Motor calibrated to {steps_per_rev} steps per revolution")

    async def get_current_angle(self) -> float:
        """Get the current angular position"""
        return self.current_angle

    async def reset_position(self):
        """Reset the current angle to 0"""
        self.current_angle = 0.0

    def cleanup(self):
        """Clean up GPIO resources"""
        GPIO.cleanup()


# Example usage
async def main():
    motor = StepperMotor()

    try:
        # Calibrate the motor (if needed)
        await motor.calibrate(200)  # Standard 200 steps per revolution

        print(f"Initial angle: {await motor.get_current_angle()}°")

        # Move to specific angles
        await motor.move_to_angle(90.0)
        print(f"Angle after move to 90°: {await motor.get_current_angle()}°")

        await motor.move_by_angle(45.0)
        print(f"Angle after move by +45°: {await motor.get_current_angle()}°")

        await motor.move_to_angle(0.0)
        print(f"Angle after move to 0°: {await motor.get_current_angle()}°")

        # Move by steps
        await motor.move_by_steps(100)
        print(f"Angle after move by 100 steps: {await motor.get_current_angle()}°")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        motor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())