# PX4 Offboard Bridge 사용 가이드

## 개요

현재 제어 시스템의 명령을 PX4 Pixhawk가 이해할 수 있는 Offboard 제어 메시지로 변환하는 브리지 노드입니다.

## 아키텍처

```
[현재 제어 시스템]
        ↓
  px4_bridge 토픽
  (/px4_bridge/*)
        ↓
[PX4 Offboard Bridge]  ← 이 모듈
        ↓
   PX4 토픽
   (/fmu/in/*)
        ↓
  [PX4 Pixhawk]
```

## 토픽 매핑

### 입력 (구독)
| 토픽 | 메시지 타입 | 내용 |
|------|------------|------|
| `/px4_bridge/control_flag` | `Bool` | 제어 모드 (True=위치, False=속도) |
| `/px4_bridge/velocity_yaw_cmd` | `Float64MultiArray` | [velocity (m/s), yaw (rad)] |
| `/px4_bridge/position_error` | `Float64MultiArray` | [x_error (m), y_error (m)] |

### 출력 (발행)
| 토픽 | 메시지 타입 | 내용 |
|------|------------|------|
| `/fmu/in/offboard_control_mode` | `OffboardControlMode` | Offboard 제어 모드 설정 |
| `/fmu/in/trajectory_setpoint` | `TrajectorySetpoint` | 궤적 설정점 (위치/속도) |

## 제어 모드

### 1. 속도 제어 모드 (`control_flag = False`)

**입력:**
- `velocity_yaw_cmd`: [전진_속도, yaw_각도]

**PX4 변환:**
```python
vx = velocity * cos(yaw)  # North 방향 속도
vy = velocity * sin(yaw)  # East 방향 속도
TrajectorySetpoint.velocity = [vx, vy, 0.0]
TrajectorySetpoint.yaw = yaw
```

**사용 예시:**
```bash
# 1 m/s로 북쪽(0도) 방향 이동
ros2 topic pub /px4_bridge/control_flag std_msgs/msg/Bool "{data: false}"
ros2 topic pub /px4_bridge/velocity_yaw_cmd std_msgs/msg/Float64MultiArray "{data: [1.0, 0.0]}"
```

### 2. 위치 제어 모드 (`control_flag = True`)

**입력:**
- `position_error`: [x_오차, y_오차]

**PX4 변환:**
```python
# 현재 위치 기준 상대 위치로 목표 설정
TrajectorySetpoint.position = [x_error, y_error, 0.0]
TrajectorySetpoint.yaw = yaw
```

**사용 예시:**
```bash
# 현재 위치에서 North +5m, East +3m 지점으로 이동
ros2 topic pub /px4_bridge/control_flag std_msgs/msg/Bool "{data: true}"
ros2 topic pub /px4_bridge/position_error std_msgs/msg/Float64MultiArray "{data: [5.0, 3.0]}"
```

## 실행 방법

### 1. 단독 실행
```bash
cd /home/yuneyoungjun/vrx_ws/src/vrx/kaboat_backup
python3 px4_offboard_bridge.py
```

### 2. ROS2 launch 파일에 추가
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_package',
            executable='px4_offboard_bridge.py',
            name='px4_offboard_bridge',
            output='screen'
        ),
    ])
```

## 중요 사항

### 1. 발행 주기
- **50Hz 고정**: PX4는 최소 2Hz 이상 필요, 권장 50Hz
- Offboard 모드 유지를 위해 지속적으로 메시지 발행 필요

### 2. 좌표계
- **NED (North-East-Down)** 사용
  - X축: North (북쪽)
  - Y축: East (동쪽)
  - Z축: Down (아래쪽, 수상정은 0 고정)

### 3. Yaw 각도
- 범위: **-π ~ +π** (라디안)
- 0도 = North (북쪽)
- π/2 = East (동쪽)
- π = South (남쪽)
- -π/2 = West (서쪽)

### 4. PX4 Offboard 모드 활성화

브리지 노드는 메시지만 발행합니다. **실제 Offboard 모드는 별도로 활성화해야 합니다:**

```bash
# QGroundControl 또는 MAVSDK를 통해:
# 1. Offboard 모드로 전환
# 2. Arm (모터 활성화)
```

또는 ROS2에서:
```python
from px4_msgs.msg import VehicleCommand

# Offboard 모드 활성화 (MAV_CMD_DO_SET_MODE)
cmd = VehicleCommand()
cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
cmd.param1 = 1.0  # 커스텀 모드
cmd.param2 = 6.0  # Offboard 모드
```

## 디버깅

### 토픽 확인
```bash
# 구독 중인 토픽 확인
ros2 topic list | grep px4_bridge

# 메시지 수신 확인
ros2 topic echo /px4_bridge/velocity_yaw_cmd
ros2 topic echo /px4_bridge/position_error
ros2 topic echo /px4_bridge/control_flag
```

### PX4 토픽 확인
```bash
# PX4로 전송되는 메시지 확인
ros2 topic echo /fmu/in/offboard_control_mode
ros2 topic echo /fmu/in/trajectory_setpoint
```

### 로그 확인
브리지 노드는 1초마다 현재 상태를 로그에 출력합니다:
```
[INFO] [px4_offboard_bridge]: [속도 제어] v=1.50 m/s, yaw=45.0°, vx=1.06, vy=1.06
```

## 문제 해결

### 1. "명령이 수신되지 않음"
- `/px4_bridge/*` 토픽이 제대로 발행되고 있는지 확인
- `ros2 topic list`로 토픽 존재 여부 확인

### 2. "PX4가 반응하지 않음"
- PX4가 Offboard 모드인지 확인
- `/fmu/in/*` 토픽이 PX4에 도달하는지 확인
- PX4 QoS 설정 확인 (RELIABLE + TRANSIENT_LOCAL)

### 3. "좌표가 이상함"
- NED 좌표계 확인
- Yaw 각도 범위 확인 (-π ~ +π)
- 현재 시스템과 PX4의 좌표계 일치 확인

## 참고 문서

- [PX4 Offboard 제어 가이드](https://docs.px4.io/main/en/ros2/offboard_control.html)
- [PX4 ROS2 사용자 가이드](https://docs.px4.io/main/en/ros2/user_guide.html)
- [TrajectorySetpoint 메시지](https://docs.px4.io/main/en/msg_docs/TrajectorySetpoint.html)
- [OffboardControlMode 메시지](https://docs.px4.io/main/en/msg_docs/OffboardControlMode.html)

## 라이선스

PX4 Autopilot 공식 문서를 참고하여 작성되었습니다.
