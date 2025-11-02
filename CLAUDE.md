# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS2-based autonomous control system for the VRX (Virtual RobotX) maritime challenge. The system uses an ONNX neural network model combined with algorithmic control for autonomous surface vehicle navigation, object detection, tracking, and mission execution.

**Key Technologies:**
- ROS2 (Robot Operating System 2) for communication infrastructure
- ONNX Runtime for neural network inference
- NanoOWL (OWL-ViT) for vision-based object detection
- MiDaS for depth estimation
- IMM-PDAF (Interacting Multiple Model - Probabilistic Data Association Filter) for robust object tracking
- PyTorch for deep learning operations

## System Architecture

### Control Flow Hierarchy

1. **Main Controller** (`main_onnx_v5_final_refactored_copy2.py`)
   - ROS2 node that orchestrates the entire system
   - Receives sensor data (LiDAR, GPS, IMU, camera)
   - Uses ONNX model for navigation control when obstacles are present
   - Switches to direct control (algorithmic) when path is clear
   - Publishes thruster commands at 100Hz (10ms timer)

2. **Hybrid Control Strategy**
   - **ONNX Model**: Used when obstacles are detected in the path
   - **Direct Control**: Used when path is clear (LOS guidance-based)
   - Decision made by `AvoidanceController.check_obstacles_and_get_control()`

3. **Sensor Processing Pipeline**
   - GPS → UTM coordinates → agent position
   - IMU → yaw angle (0-360°) → agent heading
   - LiDAR → 201 distance measurements (-100° to +100°) → obstacle detection
   - Camera → object detection → mission execution

### Key Subsystems

#### 1. Obstacle Avoidance (`utils/avoid_control.py`)

**Components:**
- `LOSGuidance`: Line-of-Sight guidance with adaptive lookahead
- `ObstacleDetector`: LiDAR-based obstacle detection with validation gates
- `DirectController`: Algorithmic control for obstacle-free paths
- `LowPassFilter`: Smooths control commands (α=0.35 default)
- `AvoidanceController`: Integrates all components

**Key Parameters:**
- `boat_width`: 1.32m (physical vessel width)
- `boat_height`: 50.0m (max detection range)
- `los_delta`: 10.0m (LOS perpendicular offset)
- `los_lookahead_min/max`: 30.0-80.0m (adaptive lookahead range)

#### 2. Object Detection System (`utils/detection_system.py`)

**Architecture:**
- Uses NanoOWL (CLIP-based zero-shot detector) for buoy detection
- MiDaS depth estimation for range filtering
- Mission-specific detection queries pre-encoded for speed

**Supported Objects:**
- Red cone buoys (mission 1)
- Green cone buoys (mission 1)
- Blue buoys (mission 2)

**Detection Pipeline:**
1. Depth map estimation (MiDaS)
2. Zero-shot detection (NanoOWL with text prompts)
3. Bounding box area filtering (500-80000 pixels)
4. Depth range filtering (0-50m)
5. Best-per-class selection (highest confidence)

#### 3. IMM-PDAF Tracker (`utils/imm_pdaf_tracker.py`)

**Purpose:** Robust multi-object tracking for maritime environments with challenging conditions (waves, occlusion, detection noise)

**Motion Models (4 models for buoy dynamics):**
- `NearlyConstantPosition`: Stationary buoys
- `ConstantVelocity`: Drifting buoys
- `ConstantAcceleration`: Wave-affected buoys
- `SingerModel`: Maneuvering targets

**Features:**
- Probabilistic data association (handles missed detections and clutter)
- Validation gating (Chi-square threshold = 9.21 for 99% confidence)
- Model probability tracking (which motion model is most likely)
- Coast tracking (up to 10 frames without detection)
- Outputs smoothed positions, velocities, and covariances

**Integration:** Detection system provides measurements, tracker provides filtered estimates

#### 4. Mission Strategies (`utils/mission_strategies_new.py`)

**Mission Types:**

1. **PASS_BETWEEN_BUOYS**: Navigate between red and green cone buoys
   - Calculates midpoint between buoys
   - Uses proportional control (gain=0.003)
   - Depth difference filtering to reject far-field false positives

2. **CIRCLE_BUOY**: Circle around blue buoy (clockwise or counter-clockwise)
   - PID controller (Kp=0.8, Ki=0.001, Kd=0.4)
   - Adaptive target x-position based on buoy depth: `target_x = 1240 - 700 * depth`
   - Tracks cumulative rotation (360° completion detection)
   - Maintains last-known commands on detection loss

3. **WAYPOINT_FOLLOW**: Simple proportional waypoint navigation
   - Heading error-based steering
   - Distance-aware speed control

4. **OBSTACLE_AVOID**: Uses `AvoidanceController` with ONNX/direct control hybrid

**Data Source Hierarchy:**
- Prefers tracked objects (IMM-PDAF filtered) for smooth control
- Falls back to raw detections (NanoOWL direct) if tracking lost
- Uses last-known commands if both fail (CIRCLE_BUOY only)

### ROS2 Communication

**Input Topics:**
- `/wamv/sensors/lidars/lidar_wamv_sensor/scan` (LaserScan)
- `/wamv/sensors/gps/gps/fix` (NavSatFix)
- `/wamv/sensors/imu/imu/data` (Imu)
- `/wamv/sensors/cameras/front_left_camera_sensor/image_raw` (Image)
- `/vrx/waypoint` (Point)

**Output Topics:**
- `/wamv/thrusters/left/thrust` (Float64)
- `/wamv/thrusters/right/thrust` (Float64)
- `/vrx/mission_status` (String)
- `/vrx/detections` (Float64MultiArray)
- Various debug/visualization topics

## Development Commands

### Environment Setup

```bash
# Install Python dependencies (ROS2 packages must be installed via apt)
pip install -r requirements.txt

# ROS2 packages (install via apt, not pip):
# - rclpy
# - sensor_msgs
# - geometry_msgs
# - std_msgs
# - cv_bridge
```

### Running the System

```bash
# Source ROS2 workspace
source /opt/ros/humble/setup.bash  # or your ROS2 distro
source ~/vrx_ws/install/setup.bash

# Run main controller (adjust model path as needed)
python3 main_onnx_v5_final_refactored_copy2.py

# Run with ROS2 launch (if launch files exist)
ros2 launch vrx_bringup vrx_control.launch.py
```

### Testing Individual Components

```bash
# Test LiDAR processing
ros2 topic echo /wamv/sensors/lidars/lidar_wamv_sensor/scan

# Test thruster commands
ros2 topic echo /wamv/thrusters/left/thrust
ros2 topic echo /wamv/thrusters/right/thrust

# Monitor mission status
ros2 topic echo /vrx/mission_status

# Visualize detections
ros2 topic echo /vrx/detections
```

## Important Implementation Details

### Control Coordinate Systems

- **Image Coordinates**: (x, y) where x is horizontal (left to right), y is vertical (top to bottom)
- **World Coordinates**: UTM-based, (x, y) represents (Easting, Northing)
- **LiDAR Angles**: -100° to +100° mapped to array indices 0-200
- **IMU Heading**: 0-360° (0° = North, increases clockwise)

### ONNX Model Integration

**Model Path:** `models/correct_IMU/gpu/Ray.onnx`

**Input Shape:** (1, 426)
- LiDAR distances: 201 values
- Agent heading: 1 value
- Angular velocity Y: 1 value
- Agent position: 2 values (UTM coordinates)
- Current waypoint: 2 values
- Previous waypoint: 2 values
- Next waypoint: 2 values
- Previous moment input: 1 value
- Previous force input: 1 value
- **Stacked twice** for temporal context

**Output:**
- `outputs[4][0][0]`: Angular velocity command (scaled by `w_scale = -1.0`)
- `outputs[4][0][1]`: Linear velocity command (scaled by `v_scale = 1.0`)

**Post-processing:** Ensures differential drive constraints (left/right thrusts within ±2000)

### Thruster Command Calculation

```python
# Differential drive kinematics
forward_thrust = linear_velocity * thrust_scale  # thrust_scale = 2000
turn_thrust = angular_velocity * thrust_scale
left_thrust = forward_thrust + turn_thrust
right_thrust = forward_thrust - turn_thrust
# Clipped to [-2000, 2000]
```

### Critical Tuning Parameters

**Main Controller:**
- `thrust_scale`: 2000 (thrust units)
- `max_lidar_distance`: 100.0m
- `waypoint_reached_threshold`: 15.0m

**Avoidance Controller:**
- `filter_alpha`: 0.35-0.5 (lower = smoother but slower response)
- `los_lookahead`: Adaptive 30-80m based on crosstrack error

**Mission-Specific:**
- Circle PID: Kp=0.8, Ki=0.001, Kd=0.4
- Pass-between gain: 0.003
- Circle target formula: `1240 - 700 * depth` (clamped 800-1200)

## Code Patterns and Conventions

### File Organization

- **Main scripts**: Root directory (e.g., `main_onnx_v5_final_refactored_copy2.py`)
- **Utility modules**: `utils/` directory with specific responsibilities
- **Models**: `models/correct_IMU/gpu/` for ONNX models
- **Each utility file is self-contained with docstrings**

### Design Patterns

1. **Manager Pattern**: `MissionManager`, `WaypointManager`, `ROSCommunicationManager`
2. **Strategy Pattern**: Mission-specific strategies inherit from `BaseMissionStrategy`
3. **Composition**: `AvoidanceController` composes `LOSGuidance`, `ObstacleDetector`, `DirectController`
4. **Callback-based ROS2 Integration**: Sensor callbacks update state, timer callback publishes commands

### Naming Conventions

- Classes: PascalCase (e.g., `IMMPDAFTracker`)
- Functions/methods: snake_case (e.g., `calculate_thruster_commands`)
- Private methods: Leading underscore (e.g., `_calculate_thruster_commands`)
- ROS topics: Lowercase with slashes (e.g., `/vrx/mission_status`)

## Troubleshooting Common Issues

### ONNX Model Loading Failures
- Verify model path exists: `models/correct_IMU/gpu/Ray.onnx`
- Check ONNX Runtime installation: CPU vs GPU version
- Ensure input shape matches (1, 426)

### ROS2 Communication Issues
- Source workspace: `source ~/vrx_ws/install/setup.bash`
- Check topic existence: `ros2 topic list`
- Verify message types match subscriber expectations

### NanoOWL Detection Failures
- Ensure NanoOWL path is correct: `/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/nanoowl`
- Check CUDA availability: `torch.cuda.is_available()`
- Verify text encodings are pre-computed during initialization

### Tracking Instability
- Increase max_coast_frames if detections are intermittent
- Adjust validation gate threshold (default 9.21)
- Check motion model transition probabilities (should be "sticky" ~0.97 diagonal)

### Control Oscillations
- Increase filter_alpha for more smoothing (but slower response)
- Reduce PID gains (especially Kp and Kd)
- Check for sensor noise in IMU or GPS

## Key Dependencies and Versions

See `requirements.txt` for exact versions. Notable dependencies:
- `numpy==1.26.4`: Core numerical operations
- `opencv-python==4.10.0.84`: Image processing
- `torch==2.2.2`: Deep learning framework
- `onnxruntime==1.18.0`: ONNX model inference
- `transformers==4.44.2`: NanoOWL model loading

## Git Branch Information

- Current branch: `simulation`
- Main branch: Not specified (check `.git/config` or ask)
- Recent activity: Refactoring and cleanup of v2 code

## Additional Notes

- The system uses **hybrid control**: algorithmic when safe, neural network when obstacles present
- IMM-PDAF tracker is advanced and handles maritime-specific challenges (waves, occlusion)
- Mission strategies use **data source fallback hierarchy** for robustness (tracked → raw → last-known)
- LiDAR obstacle detection uses **two-stage checking**: path to LOS target + fixed 20m forward cone
- All angles must be normalized to appropriate ranges (-π to π for rad, -180° to 180° or 0° to 360° for degrees)
