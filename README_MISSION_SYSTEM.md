# VRX 미션 통합 시스템

VRX 로봇을 위한 모듈화된 미션 통합 시스템입니다.

## 📋 목차

1. [시스템 구조](#시스템-구조)
2. [모듈 설명](#모듈-설명)
3. [사용 방법](#사용-방법)
4. [미션 종류](#미션-종류)
5. [실행 방법](#실행-방법)

---

## 🏗️ 시스템 구조

```
vrx_ws/src/vrx/Scripts_git/
├── main_mission.py                    # 미션 통합 실행 노드
├── main_onnx_v5_final_refactored.py  # 리팩토링된 Avoid 전용 노드
├── trajectory_viz.py                  # 시각화 노드
└── utils/
    ├── avoid_control.py               # 장애물 회피 제어 모듈
    ├── base_mission.py                # 미션 베이스 클래스
    ├── mission_gate.py                # 부표 사이 통과 미션
    ├── mission_circle.py              # 부표 선회 미션
    ├── mission_avoid.py               # 장애물 회피 미션
    └── __init__.py                    # 모듈 export
```

---

## 📦 모듈 설명

### 1. `avoid_control.py` - 장애물 회피 제어 모듈

**주요 클래스:**
- `LOSGuidance`: LOS (Line of Sight) 경로 추종 시스템
- `ObstacleDetector`: 장애물 감지 시스템
- `DirectController`: 직접 제어 시스템 (장애물 없을 때)
- `LowPassFilter`: 1차 저주파 필터
- `AvoidanceController`: 통합 장애물 회피 컨트롤러

**기능:**
- LOS guidance 기반 경로 계산
- 장애물 감지 영역 계산
- ONNX 모델과 직접 제어 자동 전환
- 제어 명령 필터링

### 2. `base_mission.py` - 미션 베이스 클래스

**주요 클래스:**
- `MissionStatus`: 미션 상태 Enum
- `BaseMission`: 모든 미션의 추상 베이스 클래스

**기능:**
- 웨이포인트 관리
- 미션 상태 관리 (IDLE, RUNNING, COMPLETED, FAILED)
- 웨이포인트 도달 판정
- 미션 진행률 추적

### 3. `mission_gate.py` - 부표 사이 통과 미션

**미션 설명:**
두 부표 사이를 정확히 통과하는 미션

**제어 방식:**
- PID 기반 헤딩 제어
- 거리에 따른 속도 조절
- 직선 경로 추종

### 4. `mission_circle.py` - 부표 선회 미션

**미션 설명:**
특정 부표를 중심으로 원을 그리며 선회

**제어 방식:**
- 접선 방향 계산
- 반경 유지 제어
- 시계방향/반시계방향 선택 가능

### 5. `mission_avoid.py` - 장애물 회피 미션

**미션 설명:**
장애물을 회피하며 목표 지점까지 이동

**제어 방식:**
- LOS guidance 경로 추종
- 장애물 감지 시 ONNX 모델 사용
- 장애물 없으면 직접 제어
- 필터링된 부드러운 제어

---

## 🚀 사용 방법

### 1. 웨이포인트 설정

`trajectory_viz.py`를 실행하여 궤적 플롯에서 클릭으로 웨이포인트 설정:

```bash
# 터미널 1: 시각화 노드 실행
ros2 run vrx_scripts trajectory_viz.py

# 터미널 2: 미션 통합 노드 실행
ros2 run vrx_scripts main_mission.py
```

### 2. 웨이포인트 순서

최소 **6개의 웨이포인트** 필요:

1. **웨이포인트 1-2**: Gate Mission (부표 사이 통과)
2. **웨이포인트 3-4**: Circle Mission (부표 선회)
3. **웨이포인트 5+**: Avoid Mission (장애물 회피)

### 3. 미션 자동 전환

- 각 미션의 모든 웨이포인트에 도달하면 자동으로 다음 미션으로 전환
- 미션 상태는 `/vrx/mission_status` 토픽으로 발행
- 현재 제어 모드는 `/vrx/control_mode` 토픽으로 발행

---

## 🎯 미션 종류

### 1. Gate Mission (부표 사이 통과)

**목표:** 두 부표 사이의 중심점을 정확히 통과

**제어 특성:**
- 직선 경로 추종
- 거리에 따른 속도 조절 (원거리: 빠름, 근거리: 느림)
- PID 기반 헤딩 제어

**파라미터:**
- `completion_threshold`: 목표 도달 판정 거리 (기본: 15m)
- `kp_heading`: 헤딩 PID 비례 계수 (기본: 2.0)

### 2. Circle Mission (부표 선회)

**목표:** 특정 부표를 중심으로 원형 경로 선회

**제어 특성:**
- 접선 방향 추종
- 일정 반경 유지
- 시계방향/반시계방향 선회

**파라미터:**
- `circle_radius`: 선회 반경 (기본: 10m)
- `circle_direction`: 선회 방향 ('clockwise' or 'counterclockwise')

### 3. Avoid Mission (장애물 회피)

**목표:** 장애물을 회피하며 목표 지점 도달

**제어 특성:**
- LOS guidance 경로 추종
- 장애물 감지 시 ONNX 모델 사용
- 장애물 없으면 직접 제어 사용
- 필터링된 부드러운 제어

**파라미터:**
- `boat_width`: 배 폭 (기본: 2.2m)
- `los_delta`: LOS 수직 오프셋 (기본: 10m)
- `los_lookahead_min/max`: Look-ahead 거리 범위 (기본: 30-80m)
- `filter_alpha`: 필터 계수 (기본: 0.35)

---

## 🔧 실행 방법

### 방법 1: 미션 통합 시스템 (권장)

```bash
# 시각화 노드
ros2 run vrx_scripts trajectory_viz.py

# 미션 통합 노드
ros2 run vrx_scripts main_mission.py
```

### 방법 2: Avoid 미션만 실행

```bash
# 시각화 노드
ros2 run vrx_scripts trajectory_viz.py

# Avoid 전용 노드 (리팩토링 버전)
ros2 run vrx_scripts main_onnx_v5_final_refactored.py
```

### 방법 3: 기존 버전 실행

```bash
# 시각화 노드
ros2 run vrx_scripts trajectory_viz.py

# 기존 버전
ros2 run vrx_scripts main_onnx_v5_final.py
```

---

## 📊 ROS2 토픽

### 발행 토픽

- `/wamv/thrusters/left/thrust` - 좌측 스러스터 명령
- `/wamv/thrusters/right/thrust` - 우측 스러스터 명령
- `/vrx/mission_status` - 현재 미션 상태
- `/vrx/control_mode` - 현재 제어 모드
- `/vrx/control_output` - 제어 출력 (linear, angular)
- `/vrx/obstacle_check_area` - 장애물 체크 영역
- `/vrx/los_target` - LOS target 위치

### 구독 토픽

- `/wamv/sensors/lidars/lidar_wamv_sensor/scan` - LiDAR 데이터
- `/wamv/sensors/gps/gps/fix` - GPS 데이터
- `/wamv/sensors/imu/imu/data` - IMU 데이터
- `/vrx/waypoint` - 웨이포인트 (trajectory_viz.py에서 발행)

---

## 📝 코드 예제

### 커스텀 미션 생성

```python
from utils.base_mission import BaseMission
import numpy as np

class MyCustomMission(BaseMission):
    """커스텀 미션 예제"""
    
    def __init__(self, waypoints, thrust_scale=800):
        super().__init__("My Custom Mission", waypoints, completion_threshold=15.0)
        self.thrust_scale = thrust_scale
    
    def update(self, current_pos, agent_heading):
        """미션 업데이트 로직"""
        if not self.is_running() or self.target_position is None:
            return 0.0, 0.0
        
        if self.check_waypoint_reached(current_pos):
            return 0.0, 0.0
        
        # 커스텀 제어 로직 구현
        left_thrust = 500.0
        right_thrust = 500.0
        
        return left_thrust, right_thrust
    
    def get_control_mode(self):
        """제어 모드 반환"""
        return "CUSTOM_MODE"
```

### 미션 추가

```python
# main_mission.py에서
custom_mission = MyCustomMission(
    waypoints=custom_waypoints,
    thrust_scale=800
)
self.missions.append(custom_mission)
```

---

## 🎨 시각화

`trajectory_viz.py`는 다음을 시각화합니다:

1. **로봇 궤적** - 로봇의 이동 경로
2. **헤딩 화살표** - 현재 헤딩 방향 (빨간색)
3. **목표 헤딩 화살표** - 목표 헤딩 방향 (초록색)
4. **LiDAR 장애물** - 감지된 장애물 포인트
5. **웨이포인트** - 클릭으로 설정한 웨이포인트
6. **장애물 체크 영역** - 장애물 감지 영역 (주황색)
7. **LOS target** - LOS guidance 목표점 (빨간 다이아몬드)
8. **제어 출력** - Linear/Angular velocity 트랙바
9. **미션 모드** - 현재 미션 및 제어 모드 표시

---

## ⚙️ 파라미터 조정

### Avoid Mission 파라미터

`utils/avoid_control.py`에서 조정:

```python
self.avoidance_controller = AvoidanceController(
    boat_width=2.2,              # 배 폭
    boat_height=50.0,            # 배 높이 (탐색 거리)
    max_lidar_distance=100.0,    # LiDAR 최대 거리
    los_delta=10.0,              # LOS 수직 오프셋
    los_lookahead_min=30.0,      # 최소 look-ahead 거리
    los_lookahead_max=80.0,      # 최대 look-ahead 거리
    filter_alpha=0.35            # 필터 계수 (낮을수록 부드러움)
)
```

### Gate Mission 파라미터

`utils/mission_gate.py`에서 조정:

```python
self.kp_heading = 2.0   # 헤딩 제어 비례 계수
self.kp_distance = 0.5  # 거리 제어 비례 계수
```

### Circle Mission 파라미터

`utils/mission_circle.py`에서 조정:

```python
circle_radius=10.0              # 선회 반경
circle_direction='clockwise'    # 선회 방향
```

---

## 🐛 문제 해결

### 1. 미션이 시작되지 않음

**원인:** 웨이포인트가 충분하지 않음  
**해결:** 최소 6개의 웨이포인트를 클릭하여 설정

### 2. 로봇이 제자리에서 회전만 함

**원인:** ONNX 모델 경로가 잘못됨  
**해결:** `main_mission.py`에서 `self.model_path` 확인

### 3. 장애물 회피가 작동하지 않음

**원인:** LiDAR 데이터가 수신되지 않음  
**해결:** LiDAR 센서 토픽 확인 (`/wamv/sensors/lidars/lidar_wamv_sensor/scan`)

### 4. 제어가 불안정함

**원인:** 필터 계수가 너무 큼  
**해결:** `filter_alpha` 값을 낮춤 (예: 0.35 → 0.2)

---

## 📄 라이선스

이 프로젝트는 VRX 시뮬레이션을 위한 로봇 제어 시스템입니다.

---

## 👥 기여

- 모듈화 및 미션 시스템: Claude 3.5 Sonnet
- 원본 코드 및 아이디어: yuneyoungjun

---

## 📚 참고 자료

- [VRX Documentation](https://github.com/osrf/vrx)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [ONNX Runtime](https://onnxruntime.ai/)

