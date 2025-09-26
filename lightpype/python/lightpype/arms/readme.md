# Arms Subdirectory

This directory contains abstractions for controlling robotic arms, including:

- **Communication protocols** for interfacing with hardware controllers
- **Calibration routines** to ensure precise joint and end-effector positioning
- **Path planning algorithms** for smooth, collision-aware motion trajectories

All components are designed to be modular and hardware-agnostic.



## Joint Configuration

The RoArm-M3 Pro features 6 controllable joints, each with specific rotation ranges and control parameters:

### 1. Base Joint (B)
- **Control Labels**: "B L" (Base Left) and "B R" (Base Right)
- **Rotation Range**: 360° (3.14 to -3.14 radians)
- **Default Position**: 0 radians (middle position)
- **Function**: Controls the horizontal rotation of the entire arm

### 2. Shoulder Joint (S)
- **Control Labels**: "S D" (Shoulder Down) and "S U" (Shoulder Up)
- **Rotation Range**: 180° (1.57 to -1.57 radians)
- **Default Position**: 0 radians (middle position)
- **Function**: Controls the primary up/down movement of the arm
- **Special Feature**: Uses dual-drive technology for increased torque

### 3. Elbow Joint (E)
- **Control Labels**: "E D" (Elbow Down) and "E U" (Elbow Up)
- **Rotation Range**: 180° (rotation values vary based on position)
- **Default Position**: 1.57 radians (middle position)
- **Function**: Controls the secondary up/down movement of the arm

### 4. Wrist Joint 1 (W)
- **Control Labels**: "W+DG" (Wrist Down/Grip) and "W-UG" (Wrist Up/Grip)
- **Rotation Range**: 180° (1.57 to -1.57 radians)
- **Default Position**: 0 radians (middle position)
- **Function**: Controls the up/down tilt of the wrist

### 5. Wrist Joint 2 (R)
- **Control Labels**: "R+DG" (Rotation Plus) and "R-UG" (Rotation Minus)
- **Rotation Range**: 360° (3.14 to -3.14 radians)
- **Default Position**: 0 radians (middle position)
- **Function**: Controls the rotational movement of the wrist

### 6. End Joint/Gripper (G)
- **Control Labels**: "G+DG" (Grip Close) and "G-UG" (Grip Open)
- **Rotation Range**: 135° (3.14 to 1.08 radians)
- **Default Position**: 3.14 radians (closed position)
- **Function**: Controls the opening and closing of the gripper



The RoArm-M3 Pro robotic arm uses a JSON command system where commands are identified by a "T" value. These "T codes" represent different functions and capabilities of the arm. This document provides a comprehensive reference for all available T codes, organized by category and numerical order.

## Etymology of "T" Codes

The "T" in T codes stands for "Type" or "Task," indicating the type of command or task to be performed. This naming convention is used throughout the RoArm-M3 Pro firmware to categorize different command functions.

## T Codes by Category

### Movement Control (100-199)

| T Code | Name | Description | Example | Parameters |
|--------|------|-------------|---------|------------|
| 101 | CMD_SINGLE_JOINT_CTRL | Controls a single joint/servo | `{"T":101,"joint":0,"rad":1.57,"spd":50,"acc":10}` | joint: Joint ID (0-5)<br>rad: Angle in radians<br>spd: Speed (1-100)<br>acc: Acceleration (1-100) |
| 102 | CMD_MULTI_JOINT_CTRL | Controls multiple joints simultaneously | `{"T":102,"joints":[0,1,2],"rads":[1.57,0.78,1.57],"spd":50,"acc":10}` | joints: Array of joint IDs<br>rads: Array of angles in radians<br>spd: Speed (1-100)<br>acc: Acceleration (1-100) |
| 103 | CMD_ALL_JOINT_CTRL | Controls all joints at once | `{"T":103,"rads":[1.57,0.78,1.57,0,0,1.57],"spd":50,"acc":10}` | rads: Array of 6 angles in radians<br>spd: Speed (1-100)<br>acc: Acceleration (1-100) |
| 110 | CMD_COORDCTRL | Controls end-effector position in Cartesian space | `{"T":110,"x":150,"y":0,"z":100,"roll":0,"pitch":90,"yaw":0,"spd":50}` | x,y,z: Coordinates in mm<br>roll,pitch,yaw: Orientation in degrees<br>spd: Speed (1-100) |


A.F.