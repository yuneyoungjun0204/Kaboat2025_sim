# VRX Kaboat 프로젝트 구조

## 📁 메인 실행 파일

### Main_MCP.py ⭐ (주요 시스템)
통합 미션 제어 시스템
- 모든 미션 타입 지원
- NanoOWL 객체 탐지 + IMM-PDAF 트래킹
- 웨이포인트 기반 미션 전환

**사용법:**
```bash
ros2 run vrx Main_MCP.py
```

### main_avoid.py (전문화 버전)
장애물 회피 전용 간소화 시스템
- OBSTACLE_AVOID 미션만 실행
- 탐지/트래킹 시스템 제외 (경량화)
- ONNX + LOS guidance 하이브리드

**사용법:**
```bash
ros2 run vrx main_avoid.py
```

### move.py (개발/테스트용)
키보드 수동 조작 컨트롤러
- 개발 및 디버깅용
- 키보드로 직접 스러스터 제어

**사용법:**
```bash
ros2 run vrx move.py
```

### trajectory_viz.py (시각화)
실시간 궤적 시각화
- matplotlib 기반 실시간 플롯
- 위치, 헤딩, LiDAR 데이터 표시

**사용법:**
```bash
ros2 run vrx trajectory_viz.py
```

---

## 📦 Utils 모듈 구조

### 핵심 시스템
- **config.py**: 모든 시스템 파라미터 중앙 관리
- **system_factory.py**: 컴포넌트 팩토리 패턴
- **ros_communication.py**: ROS2 퍼블리셔/서브스크라이버 관리

### 미션 실행
- **mission_strategies_new.py**: 6가지 미션 전략 구현
  - PassBetweenBuoys
  - CircleBuoy
  - WaypointFollow
  - ObstacleAvoid
  - DockMode
  - Rotation
- **mission_executor.py**: 미션 실행 로직
- **mission_control.py**: 미션 제어 루프
- **waypoint_manager.py**: 웨이포인트 관리

### 제어 시스템
- **onnx_controller.py**: ONNX v1 (213 observations)
- **onnx_controller_v2.py**: ONNX v2 (207 observations, Unity ML-Agent 스타일) ⭐ NEW
- **thruster_allocation.py**: Body forces → Thruster commands ⭐ NEW
- **avoid_control.py**: 장애물 회피 제어

### 센서 및 감지
- **detection_system.py**: NanoOWL 객체 탐지
- **sensor_callbacks.py**: 센서 콜백 관리
- **sensor_preprocessing.py**: 센서 데이터 전처리
- **depth_estimation.py**: MiDaS 깊이 추정
- **imm_pdaf_tracker.py**: IMM-PDAF 다중 객체 추적

### 시각화
- **visualization_system.py**: OpenCV 시각화
- **viz_components.py**: 시각화 컴포넌트

### 유틸리티
- **helpers.py**: 공통 헬퍼 함수 ⭐ NEW
  - normalize_heading()
  - calculate_heading_error()
  - find_buoy_with_fallback()
- **parameter_manager.py**: UI 파라미터 관리
- **parameter_onnx.py**: ONNX 센서 데이터 관리

---

## 🎯 미션 타입

1. **PASS_BETWEEN_BUOYS**: 부표 사이 지나가기
2. **CIRCLE_BUOY**: 부표 한바퀴 돌기 (시계/반시계)
3. **WAYPOINT_FOLLOW**: 단순 웨이포인트 추종
4. **OBSTACLE_AVOID**: 장애물 회피 (ONNX + LOS)
5. **DOCK_MODE**: 도킹 스테이션 미션
6. **ROTATION**: 제자리 선회

---

## 🚀 최근 리팩토링 (2025)

### 새로운 기능
1. **통일된 명령 인터페이스** ⭐
   - 모든 미션: `(desired_speed, desired_yaw, desired_force_y)` 반환
   - ROS 토픽: `/vrx/desired_speed`, `/vrx/desired_moment`, `/vrx/desired_force_y`
   - 다양한 배에 쉽게 적용 가능

2. **ONNX Controller v2** ⭐
   - Unity ML-Agent 스타일 observation
   - 단순화된 구조 (207 obs)
   - `config.py`에서 `ONNX_VERSION = 2`로 전환

3. **Thruster Allocation 모듈** ⭐
   - Vectored thruster 지원
   - Differential drive 지원
   - Body forces → Motor commands

4. **헬퍼 함수 통합** ⭐
   - 중복 코드 제거
   - `utils/helpers.py`로 통합

### 점진적 마이그레이션
- 기존 코드 완전 호환
- `execute_body_forces()` 메서드로 새 인터페이스 사용
- 기존 `execute()` 메서드는 자동으로 thruster allocation 수행

---

## ⚙️ 설정 방법

### ONNX 버전 전환
```python
# utils/config.py
ONNX_VERSION = 1  # v1 (기존)
ONNX_VERSION = 2  # v2 (Unity ML-Agent 스타일)
```

### 웨이포인트 설정
```python
# utils/config.py
PREDEFINED_WAYPOINTS = [
    (x, y, 'MISSION_TYPE', radius, params),
    ...
]
```

### 파라미터 조정
모든 파라미터는 `utils/config.py`의 `Constants` 클래스에서 관리

---

## 📝 개발 가이드

### 새로운 미션 추가
1. `mission_strategies_new.py`에 미션 클래스 생성
2. `BaseMissionStrategy` 상속
3. `execute_body_forces()` 구현
4. `MissionType` enum에 추가
5. `config.py`에 파라미터 추가

### Body Force 명령 사용
```python
# 미션에서
desired_speed, desired_yaw, desired_force_y = mission.execute_body_forces(**kwargs)

# ROS 토픽 발행
ros_comm.publish_desired_control(desired_speed, desired_yaw, desired_force_y)

# Thruster 명령 계산
from utils.thruster_allocation import body_forces_to_thruster_commands
left_thrust, right_thrust, left_pos, right_pos = body_forces_to_thruster_commands(
    desired_speed, desired_yaw, desired_force_y, thrust_scale
)
```

---

## 🔧 의존성

### Python 패키지
- rclpy (ROS2)
- numpy
- opencv-python
- onnxruntime
- pymap3d
- filterpy

### 외부 모델
- NanoOWL (객체 탐지)
- MiDaS (깊이 추정)
- ONNX 강화학습 모델

---

## 📊 시스템 아키텍처

```
Main_MCP.py
    ├── VRXSystemFactory
    │   ├── Detection System (NanoOWL + MiDaS)
    │   ├── Tracking System (IMM-PDAF)
    │   ├── Mission Manager
    │   ├── ONNX Controller (v1 or v2)
    │   ├── Waypoint Manager
    │   └── Visualization System
    │
    ├── Mission Control Loop
    │   ├── Sensor Processing
    │   ├── Mission Execution
    │   │   └── Body Forces → Thruster Allocation
    │   └── ROS Publishing
    │
    └── ROS Communication
        ├── Sensors (LiDAR, GPS, IMU, Camera)
        └── Control (Thrust, Position, Status)
```

---

## 📌 TODO

- [ ] 모든 미션에 execute_body_forces() 완전 통합
- [ ] CircleBuoy/Dock 미션 body force 인터페이스 적용
- [ ] 실제 배에서 테스트
- [ ] 파라미터 자동 튜닝 시스템
- [ ] 더 많은 미션 타입 추가

---

**Last Updated**: 2025-01-15
**Version**: 2.0 (Refactored)
