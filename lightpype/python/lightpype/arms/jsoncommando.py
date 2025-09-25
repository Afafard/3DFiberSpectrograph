import serial
import json
import time


class ArmCommander:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1):
        """
        Initialize the ArmCommander with serial connection settings.

        Args:
            port (str): Serial port path. Default is '/dev/ttyUSB0'.
            baudrate (int): Baud rate for serial communication. Default is 115200.
            timeout (int): Read timeout in seconds. Default is 1.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None

    def connect(self):
        """Establish serial connection."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=1
            )
            time.sleep(2)  # Give device time to reset
            print(f"Connected to {self.port}")
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

    def disconnect(self):
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from serial port.")

    def send_json_command(self, command_dict):
        """
        Send a JSON command over serial with newline termination.

        Args:
            command_dict (dict): Dictionary to be serialized as JSON.

        Returns:
            str: Response received from device, or None if timeout/error.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            raise RuntimeError("Serial connection not established. Call connect() first.")

        try:
            json_str = json.dumps(command_dict)
            command_bytes = (json_str + '\n').encode('utf-8')
            self.serial_conn.write(command_bytes)
            print(f"Sent: {json_str}")

            # Wait for and read response
            response = self.serial_conn.readline()
            if response:
                decoded = response.decode('utf-8').strip()
                print(f"Received: {decoded}")
                return decoded
            else:
                print("No response received (timeout).")
                return None

        except UnicodeDecodeError:
            print("Received non-UTF8 data.")
            return None
        except Exception as e:
            print(f"Error sending command: {e}")
            return None

    def send_command_and_wait(self, command_dict, timeout_seconds=5):
        """
        Send a JSON command and wait for response with optional custom timeout.

        Args:
            command_dict (dict): Command to send.
            timeout_seconds (int): Maximum time to wait for response.

        Returns:
            str: Response string or None if no response within timeout.
        """
        self.serial_conn.timeout = timeout_seconds
        return self.send_json_command(command_dict)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example usage:
if __name__ == "__main__":
    with ArmCommander() as commander:
        response = commander.send_json_command({"T": 401, "cmd": 3})
        if response:
            print("WiFi mode set to AP+STA successfully.")
        else:
            print("Failed to set WiFi mode.")