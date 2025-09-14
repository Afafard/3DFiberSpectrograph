from gpiozero import LED
from time import sleep
import random

# Define LED pins (using BCM numbering)
red_led = LED(12)
green_led = LED(20)
blue_led = LED(21)
white_led = LED(16)

# Turn off all LEDs initially
red_led.off()
green_led.off()
blue_led.off()
white_led.off()


def clear_leds():
    """Turn off all LEDs"""
    red_led.off()
    green_led.off()
    blue_led.off()
    white_led.off()


def single_led_dance():
    """Single LED dancing pattern"""
    patterns = [
        (red_led, green_led, blue_led, white_led),
        (green_led, blue_led, white_led, red_led),
        (blue_led, white_led, red_led, green_led),
        (white_led, red_led, green_led, blue_led)
    ]

    for pattern in patterns:
        clear_leds()
        for led in pattern:
            led.on()
            sleep(0.2)
            led.off()


def rainbow_dance():
    """Rainbow color pattern"""
    colors = [red_led, green_led, blue_led, white_led]

    for _ in range(5):
        for i, color in enumerate(colors):
            clear_leds()
            color.on()
            sleep(0.15)

        # Cycle colors
        colors = colors[1:] + [colors[0]]


def random_dance():
    """Random LED pattern"""
    for _ in range(20):
        clear_leds()
        # Turn on a random LED
        random.choice([red_led, green_led, blue_led, white_led]).on()
        sleep(0.3)


def pulse_dance():
    """Pulsing effect"""
    leds = [red_led, green_led, blue_led, white_led]

    for _ in range(10):
        # Turn on all LEDs with different intensities
        for led in leds:
            led.on()
            sleep(0.1)

        # Turn off all LEDs
        clear_leds()
        sleep(0.2)


def chase_pattern():
    """Chasing pattern"""
    leds = [red_led, green_led, blue_led, white_led]

    for _ in range(3):
        for i, led in enumerate(leds):
            clear_leds()
            led.on()
            sleep(0.3)

        # Reverse direction
        for i in range(len(leds) - 1, -1, -1):
            clear_leds()
            leds[i].on()
            sleep(0.3)


def main():
    """Main dance loop"""
    try:
        while True:
            # Different patterns with varying durations
            single_led_dance()
            sleep(0.5)

            rainbow_dance()
            sleep(0.5)

            random_dance()
            sleep(0.5)

            pulse_dance()
            sleep(0.5)

            chase_pattern()
            sleep(0.5)

    except KeyboardInterrupt:
        print("\nDance sequence interrupted")
    finally:
        clear_leds()


if __name__ == "__main__":
    main()
