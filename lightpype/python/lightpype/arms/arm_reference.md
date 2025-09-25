AI Assistant
# RoArm-M3 Pro Control Reference Guide

This document provides a concise, practical reference for controlling the RoArm-M3 Pro robotic arm using JSON commands via HTTP or Serial. It consolidates all essential commands, parameters, and best practices into a single, easy-to-use format for development and integration.

---

## ✅ Core Control Commands

| Command Type | JSON Format | Parameters | Description | Notes |
|--------------|-------------|------------|-------------|-------|
| **LED Control** | `{"type": "LedCtrl", "enable": true}` | `enable`: boolean (`true`/`false`) | Turns built-in LED lights on or off. | Minimal power draw; useful for status indication. |
| **Torque Control** | `{"type": "TorqueCtrl", "id": 0, "enable": false}` | `id`: joint ID (`0`=all, `1–6`=individual)<br>`enable`: boolean | Enables/disables servo torque (holding force). | When disabled, arm can be manually moved. **Automatically re-enables** when any movement command is sent. |
| **DEFA (Force Adaptive)** | `{"type": "DefaCtrl", "enable": true}` | `enable`: boolean (`true`/`false`) | Enables Dynamic External Force Adaptive Control. | Arm returns to last commanded position after manual disturbance. **Requires torque enabled** — enabling DEFA auto-enables torque. |
| **Joint Angle Control** | `{"type": "AngleCtrl", "id": 1, "angle": 90, "speed": 50}` | `id`: joint ID (`1–6`)<br>`angle`: angle in **degrees** (0–180°)<br>`speed`: 1–100 | Moves a single joint to specified angle. | **Note**: Despite documentation sometimes using radians, the JSON command expects **degrees**. |
| **Coordinate Control (End-Effector)** | `{"type": "CoordCtrl", "x": 150, "y": 0, "z": 100, "speed": 50}` | `x`, `y`, `z`: position in **mm**<br>`roll`, `pitch`, `yaw`: orientation in degrees (optional)<br>`speed`: 1–100 | Moves end-effector to Cartesian coordinates using inverse kinematics. | Workspace limits: ~±200mm (X,Y), 0–300mm (Z). Avoid singularities near workspace edges. |
| **Gripper Control** | `{"type": "AngleCtrl", "id": 6, "angle": 0, "speed": 50}` | `id`: always `6`<br>`angle`: 0–180°<br>`speed`: 1–100 | Opens or closes gripper. | `angle=0` = fully open, `angle=180` = fully closed. |
| **Run Sequence** | `{"type": "RunSeq", "name": "seq1", "loop": false, "speed": 50}` | `name`: sequence name<br>`loop`: boolean (`true`/`false`)<br>`speed`: 1–100 | Executes a previously saved motion sequence. | Sequences must be pre-saved via `SavePos` and `SaveSeq`. |
| **Save Position** | `{"type": "SavePos", "name": "home"}` | `name`: string identifier (e.g., `"home"`, `"pick"`)| Saves current joint positions under a named label. | Used with `RunSeq` to replay sequences. Position is saved as joint angles (not coordinates). |
| **Reset Arm** | `{"type": "Reset"}` | None | Moves all joints to default (home) position. | Equivalent to pressing “INIT” on the web UI. |
| **Get Status** | `{"type": "GetStatus"}` | None | Returns current joint angles, end-effector position, and system state. | Essential for feedback loops, calibration, or debugging. Response includes: `joints`, `position`, `status`. |

---

## 🔌 ESP-NOW Leader-Follower Mode (Multi-Arm Sync)

Use these commands to synchronize multiple RoArm-M3 Pro units via ESP-NOW (direct 2.4GHz peer-to-peer).

| Command | JSON Format | Parameters | Description |
|---------|-------------|------------|-------------|
| **Get MAC Address** | `{"T":302}` | None | Returns device’s unique MAC address (e.g., `"mac":"AA:BB:CC:DD:EE:FF"`). **Required** for pairing. |
| **Set as Leader (Broadcast)** | `{"T":301,"mode":1,"dev":0,"cmd":0,"megs":0}` | `mode=1`: F-LEADER-B (Broadcast)<br>`dev`, `cmd`, `megs`: ignored | Enables the arm as a broadcast Leader. |
| **Enable Broadcasting** | `{"T":300,"mode":1,"mac":"FF:FF:FF:FF:FF:FF"}` | `mode=1`: enable<br>`mac="FF:FF:FF:FF:FF:FF"`: broadcast to all | Must follow `mode:1` setup. Sends joint data to all registered Followers. |
| **Add Follower (to Leader)** | `{"T":303,"mac":"AA:BB:CC:DD:EE:FF"}` | `mac`: MAC address of Follower to add | Registers a specific device as a Follower under this Leader. |
| **Set as Follower** | `{"T":301,"mode":3,"dev":0,"cmd":0,"megs":0}` | `mode=3`: FOLLOWER | Configures the arm to listen for and mirror a Leader. |
| **Control All Followers** | `{"T":305,"b":0,"s":1.57,"e":0,"t":0,"r":0,"h":1.57}` | `b`, `s`, `e`, `t`, `r`, `h`: joint angles in **radians**<br>`dev`, `cmd`, `megs`: ignored | Directly sends joint positions to all registered Followers. Bypasses Leader sync — useful for scripting override. |
| **Control Specific Follower** | `{"T":306,"mac":"AA:BB:CC:DD:EE:FF","b":0,...}` | `mac`: target Follower’s MAC<br>`b`, `s`, `e`, `t`, `r`, `h`: joint angles in **radians** | Sends position data to one specific Follower. |
| **Reset to Normal Mode** | `{"T":301,"mode":0,"dev":0,"cmd":0,"megs":0}` | `mode=0`: disabled | Disables ESP-NOW and returns to standard operation. |

> 💡 **Important**: Follower arms ignore all other input (web, HTTP, Serial) while in Follower mode — only Leader commands are honored.

---

## 📦 System & Configuration Commands

| Command | JSON Format | Parameters | Description |
|---------|-------------|------------|-------------|
| **Save WiFi Config (Persist)** | `{"T":407,"mode":3,"ap_ssid":"RoArm-M3","ap_password":"12345678","sta_ssid":"","sta_password":""}` | `mode`: 0=OFF, 1=AP, 2=STA, 3=AP+STA<br>`ap_ssid`, `ap_password`<br>`sta_ssid`, `sta_password` | Saves WiFi settings to non-volatile storage. **Required** for settings to survive reboot. |
| **Save Current WiFi Config** | `{"T":406}` | None | Saves current WiFi settings (SSID, password, mode) to `wifiConfig.json`. Use after changes via web UI. |
| **Reboot Device** | `{"T":600}` | None | Reboots ESP32 controller. Use after config changes or if unresponsive. |
| **Clear NVS (Factory Reset)** | `{"T":604}` | None | Erases all stored settings: WiFi, sequences, Leader-Follower pairs. **Use with caution** — resets to factory defaults. |

> ✅ **Critical Workflow**:  
> 1. Change WiFi or system settings via web UI or JSON.  
> 2. Send `{"T":406}` to persist changes.  
> 3. Reboot with `{"T":600}` if needed.

---

## 📡 Communication Methods

### HTTP (WiFi)
- **Endpoint**: `http://192.168.4.1/js?json={...}`
- **Method**: `GET`
- **Example**:
  ```python
  import requests
  url = "http://192.168.4.1/js?json=" + requests.utils.quote('{"type":"AngleCtrl","id":1,"angle":90,"speed":50}')
  response = requests.get(url)
  ```

### Serial (USB Type-C)
- **Baudrate**: `115200`
- **End delimiter**: `\n` (newline)
- **Example**:
  ```python
  import serial
  ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
  cmd = '{"type":"LedCtrl","enable":true}\n'
  ser.write(cmd.encode())
  ```

> ⚠️ **Note**: Always end Serial commands with `\n`. HTTP expects URL-encoded JSON; Serial expects raw JSON + newline.

---

## 📍 Joint ID Reference

| Joint ID | Name | Control Labels | Range (Degrees) |
|----------|------|----------------|-----------------|
| 1 | Base | B L / B R | -180° to +180° |
| 2 | Shoulder | S D / S U | -90° to +90° |
| 3 | Elbow | E D / E U | ~ -90° to +180° (varies) |
| 4 | Wrist 1 | W+DG / W-UG | -90° to +90° |
| 5 | Wrist 2 | R+DG / R-UG | -180° to +180° |
| 6 | Gripper | G+DG / G-UG | 0° (closed) to 180° (open) |

> 💡 **Tip**: Use `{"type":"GetStatus"}` to read current joint angles before programming.

---

## ✅ Best Practices & Pro Tips

- **Degrees, Not Radians**: All `AngleCtrl` and `CoordCtrl` JSON commands use **degrees**, even if docs show radians.
- **Command Order**: Always send `{"T":406}` after changing WiFi settings to persist across reboots.
- **Torque + DEFA**: DEFA requires torque enabled. Disable torque only when manually positioning.
- **Avoid Rapid Commands**: Send commands no faster than 10–20 Hz to prevent buffer overflow on ESP32.
- **Use `GetStatus` for Feedback**: Monitor actual joint positions before and after movements for accuracy.
- **Test Slow First**: Use low `speed` values (20–30) during initial testing to prevent mechanical stress.
- **Serial Debugging**: Use a serial monitor (e.g., `screen /dev/ttyUSB0 115200`) to see raw responses.
- **Leader-Follower MAC**: Always verify MAC addresses with `{"T":302}` — don’t guess or copy from other devices.
- **Safety First**: Always disable torque before manually moving the arm. Be aware of pinch points and workspace boundaries.

---


## 📚 Resources

- **Web Interface**: `http://192.168.4.1`
- **WiFi Credentials**: SSID `RoArm-M3`, Password `12345678`
- **Python API SDK**: [GitHub - waveshare/RoArm-M3-Python](https://github.com/waveshare/RoArm-M3-Python)
- **Firmware & Docs**: [Waveshare RoArm-M3 Wiki](https://www.waveshare.com/wiki/RoArm-M3)
- **ESP-NOW Protocol**: [Espressif ESP-NOW Docs](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html)


