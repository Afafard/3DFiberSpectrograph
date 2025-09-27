import serial
from arm import RoArmM3, Joint
import time
import json

# Setup serial transport
ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)


# Wrap serial in a transport-compatible interface
class SerialTransport:
    def __init__(self, ser):
        self.ser = ser

    def send(self, data: str):
        self.ser.write(data.encode('utf-8'))

    def recv(self) -> str:
        line = self.ser.readline().decode('utf-8').strip()
        return line


transport = SerialTransport(ser)
arm = RoArmM3(transport)

try:
    print("=== SERIAL TEST ===")

    # 1. Move to initial position
    arm.move_to_initial()
    print("Moved to initial position.")
    time.sleep(0.5)  # Allow time for movement

    # 2. Move joint 1 (base) slightly
    arm.move_joint_rad(Joint.BASE, 0.5)
    time.sleep(0.5)  # Allow time for movement
    arm.move_joint_rad(Joint.BASE, -0.5)
    time.sleep(0.5)  # Allow time for movement
    print("Moved base joint slightly.")
    time.sleep(1)
    arm.move_axis(1,50,10)
    time.sleep(1)
    arm.move_axis(2, 150, 10)
    arm.move_to_xyz(50,200,200)
    arm.move_joints_angle()


    arm.move_to_initial()


    time.sleep(0.5)
    feedback = None
    while not feedback:
        feedback = arm.get_feedback()
        if feedback:
            print("Current feedback:", json.dumps(feedback, indent=2))

finally:
    ser.close()
