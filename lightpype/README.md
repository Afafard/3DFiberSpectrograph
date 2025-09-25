# Dual Robotic Arm 3D Goniometric Scanner

![Mascot](mascot.png)

## Overview

This project implements a 3D goniometric scanning system using two RoArm-M3 robotic arms to characterize the optical properties of samples in a complete hemispherical geometry. The system is designed for scientific imaging applications where precise angular measurements of light interaction with materials are required.

## Problem Context

Scientific characterization of materials often requires understanding how light interacts with surfaces at various angles. Traditional goniometric systems are limited in their angular coverage and flexibility. This system addresses these limitations by using two 5+1 DOF robotic arms with a rotating sample platform to achieve complete hemispherical coverage with precise control.

The challenge is to coordinate two robotic arms and a turntable to systematically sample every combination of illumination and detection angles while avoiding collisions and maintaining precise positioning.

## System Geometry

### Physical Setup
- **Sample Position**: Origin point (0,0,0) - center of the workspace
- **Arm Bases**: 
  - Left Arm (Spectrometer): (-400mm, 0, 200mm) 
  - Right Arm (Illuminator): (400mm, 0, 200mm)
- **Workspace**: 1m³ cube with sample platform at z=0, arms mounted at z=200mm
- **Turntable**: Mounted at z=-50mm, rotates sample in x-y plane
- **Arm Reach**: ~500mm effective reach with 200mm safety margin

### Coordinate System
All coordinates are in millimeters (mm) with the sample at the origin:
- **X-axis**: Left-right across the workspace
- **Y-axis**: Front-back across the workspace  
- **Z-axis**: Vertical (0 at sample surface, positive upward)

### Spherical Coordinate System
Measurements are defined in spherical coordinates relative to the sample:
- **r**: Radial distance from sample center (50-300mm)
- **θ (theta)**: Polar angle from +Z axis (0-π/2 for hemisphere)
- **φ (phi)**: Azimuthal angle from +X axis toward +Y (0-2π)

## System Components

### Hardware
1. **Two RoArm-M3 Robotic Arms** (5+1 DOF each)
   - Left Arm: Spectrometer detection
   - Right Arm: LED illuminator source
2. **Stepper Motor Turntable** for sample rotation
3. **Raspberry Pi 5** for system control
4. **Spectrometer** and **LED Illuminator** end-effectors

### Software Architecture
```
scanning_system.py     # Main system controller
├── arm.py           # Robotic arm control interface
├── pathplanning.py   # Trajectory planning and collision avoidance
├── viz.py            # Real-time 3D visualization (Qt/Matplotlib)
├── turntable_control.py # Stepper motor management
└── calibration.pkl    # Saved calibration data
```

## Operation Workflow

### 1. System Initialization
```bash
python scanning_system.py --scan
```

### 2. Calibration Process
The system performs automatic calibration to map between arm coordinate systems:

1. **Arm Calibration**: Positions both arms at 7+ common reference points
2. **Coordinate Mapping**: Computes transformation matrices between arms
3. **Reachability Analysis**: Determines accessible workspace regions
4. **Collision Avoidance**: Calculates safe operating envelopes

### 3. Sample Setup
1. System moves arms to safe positions
2. User mounts sample at origin (0,0,0)
3. Instruments are automatically grasped by robotic arms

### 4. Scan Planning
User defines scanning parameters:
- **Radial Range**: Distance from sample (50-300mm)
- **Angular Resolution**: Theta (elevation) and Phi (azimuth) steps
- **Coverage**: Complete hemisphere or custom angular ranges

### 5. Execution with Visualization
1. Real-time 3D visualization shows planned paths
2. Arms move in coordination with turntable rotation
3. Spectrometer measurements collected at each position
4. Data stored with precise angular coordinates

## Safety Features

### Collision Avoidance
- **Proximity Detection**: Real-time distance monitoring
- **Path Planning**: Collision-free trajectory optimization
- **Emergency Stop**: Immediate halt on safety violations
- **Workspace Limits**: Hardware and software position constraints

### Error Handling
- **Connection Recovery**: Automatic reconnection attempts
- **Timeout Protection**: Movement completion verification
- **State Monitoring**: Continuous system status tracking
- **Graceful Degradation**: Fallback to safe positions on error

## Performance Specifications

[//]: # (### Angular Resolution)

[//]: # (- **Theoretical**: 0.1° minimum step size)

[//]: # (- **Practical**: 1-10° typical resolution)

[//]: # (- **Repeatability**: ±0.5° positional accuracy)

### Coverage
- **Hemispherical**: 2π steradians complete coverage
- **Sampling Density**: Configurable (typically 500-2000 points)

[//]: # (- **Measurement Time**: 1-10 minutes depending on resolution)

### Data Output
- **Format**: JSON with angular coordinates and spectral data
- **Metadata**: Timestamp, position, instrument settings
- **Calibration**: Traceable to reference positions

## Installation Requirements

### Hardware Setup
1. Connect RoArm-M3 arms to USB ports:
   - Left Arm (Spectrometer): e.g.`/dev/ttyUSB0`
   - Right Arm (Illuminator): e.g. `/dev/ttyUSB1`
2. Connect stepper motor controller for turntable
3. Mount sample at origin (0,0,0)
4. Position arms at specified base coordinates

### Software Dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
pyserial>=3.5
numpy>=1.21.0
scipy>=1.7.0
PyQt5>=5.15.0
pyqtgraph>=0.12.0
matplotlib>=3.4.0
```

### Configuration Files
- `motor_gpio.json`: GPIO pin mappings for stepper motor
- `calibration.pkl`: Saved calibration data
- `system_config.json`: Operational parameters
