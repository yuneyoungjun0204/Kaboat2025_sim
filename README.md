# 🚤 Kaboat2025 시뮬레이션 프로젝트

VRX (Virtual RobotX) 해상 로봇 대회를 위한 자율 항법 시스템

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 구조](#시스템-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [미션 종류](#미션-종류)
- [Config 시스템](#config-시스템)

---

## 🎯 프로젝트 개요

ROS2 기반 해상 로봇 자율 항법 시스템으로, 3가지 미션을 순차적으로 수행합니다:

1. **Gate Mission**: 두 부표 사이를 정확히 통과
2. **Circle Mission**: 부표를 중심으로 원형 선회
3. **Avoid Mission**: 장애물을 회피하며 목표 지점 도달

### 주요 기능

- ✅ ONNX 모델 기반 AI 제어 (장애물 회피)
- ✅ LiDAR 기반 장애물 감지
- ✅ GPS/IMU 센서 융합
- ✅ 카메라 기반 부표 탐지 (MiDaS 깊이 추정)
- ✅ Config 기반 설정 관리

---

## 🏗️ 시스템 구조

```
Kaboat2025_sim/
├── main_mission.py                    # 통합 미션 시스템 (메인)
├── main_onnx_v5_final_refactored.py  # Avoid 미션 전용
├── trajectory_viz.py                  # 시각화 노드
│
├── config/                            # 설정 파일
│   ├── topics.yaml                    # ROS2 토픽 설정
│   ├── mission_config.yaml            # 미션 파라미터
│   └── README.md                      # Config 가이드
│
├── utils/                             # 유틸리티 모듈
│   ├── config_manager.py              # 설정 관리자
│   ├── avoid_control.py               # 장애물 회피 제어
│   ├── mission_*.py                   # 미션 모듈들
│   ├── sensor_preprocessing.py        # 센서 데이터 처리
│   └── ...
│
├── models/                            # ONNX 모델
├── venv/                              # Python 가상환경
├── requirements.txt                   # Python 의존성
└── README.md                          # 이 파일
```

---

## 🔧 설치 방법

### 1. 시스템 요구사항

- **OS**: Ubuntu 22.04 (ROS2 Humble)
- **Python**: 3.8 이상
- **ROS2**: Humble Hawksbill
- **VRX**: 설치 완료

### 2. 저장소 클론

```bash
cd ~/vrx_ws/src/vrx/
git clone <repository-url> Scripts_git
cd Scripts_git
```

### 3. 가상환경 설정 (자동)

```bash
# 최초 1회 실행
./setup_venv.sh
```

또는 수동 설정:

```bash
# 가상환경 생성
python3 -m venv venv

# 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 4. 가상환경 활성화

매번 사용 시:

```bash
source activate_venv.sh
```

또는:

```bash
source venv/bin/activate
```

---

## 🚀 사용 방법

### 1. ROS2 환경 설정

```bash
source /opt/ros/humble/setup.bash
source ~/vrx_ws/install/setup.bash
```

### 2. 가상환경 활성화

```bash
source activate_venv.sh
```

### 3. 시뮬레이션 실행

#### 터미널 1: 시각화 노드
```bash
python trajectory_viz.py
```

#### 터미널 2: 미션 시스템
```bash
python main_mission.py
```

### 4. 웨이포인트 설정

시각화 창에서 클릭으로 웨이포인트 추가:
- **처음 2개**: Gate Mission
- **다음 2개**: Circle Mission
- **그 다음**: Avoid Mission (2개 이상)

최소 6개 웨이포인트 필요!

---

## 🎯 미션 종류

### 1. Gate Mission (부표 사이 통과)

**목표**: 두 부표의 중심점을 정확히 통과

**제어 방식**:
- PID 헤딩 제어
- 거리 비례 속도 조절
- 직선 경로 추종

**파라미터** (`config/mission_config.yaml`):
```yaml
missions:
  gate:
    completion_threshold: 15.0
    kp_heading: 2.0
```

### 2. Circle Mission (부표 선회)

**목표**: 특정 부표를 중심으로 원형 선회

**제어 방식**:
- 접선 방향 계산
- 일정 반경 유지 (10m)
- 시계방향/반시계방향 선택

**파라미터**:
```yaml
missions:
  circle:
    radius: 10.0
    direction: 'clockwise'
```

### 3. Avoid Mission (장애물 회피)

**목표**: 장애물 회피하며 웨이포인트 도달

**제어 방식**:
- LOS (Line of Sight) 경로 추종
- LiDAR 장애물 감지
- **장애물 있음** → ONNX 모델 (AI 제어)
- **장애물 없음** → 직접 제어 (PID)
- 저주파 필터로 부드러운 제어

**파라미터**:
```yaml
missions:
  avoid:
    boat_width: 2.2
    los_delta: 10.0
    filter_alpha: 0.35
```

---

## ⚙️ Config 시스템

### Config 파일 구조

```
config/
├── topics.yaml          # 토픽 설정
└── mission_config.yaml  # 파라미터 설정
```

### 토픽 변경 (topics.yaml)

```yaml
sensors:
  lidar: '/wamv/sensors/lidars/lidar_wamv_sensor/scan'
  gps: '/wamv/sensors/gps/gps/fix'

actuators:
  thrusters:
    left: '/wamv/thrusters/left/thrust'
    right: '/wamv/thrusters/right/thrust'
```

**장점**: 토픽명 변경 시 YAML만 수정 → 코드 수정 불필요!

### 파라미터 조정 (mission_config.yaml)

```yaml
control:
  thrust_scale: 800
  v_scale: 1.0

missions:
  gate:
    completion_threshold: 15.0
  avoid:
    filter_alpha: 0.35
```

**장점**: 파라미터 튜닝이 쉬워짐!

### 코드에서 사용

```python
from utils import get_config

config = get_config()

# 토픽 가져오기
lidar_topic = config.get_sensor_topic('lidar')

# 파라미터 가져오기
thrust_scale = config.get_param('control', 'thrust_scale')
```

상세 가이드: [`config/README.md`](config/README.md)

---

## 📊 ROS2 토픽

### 구독 토픽 (Subscriptions)

- `/wamv/sensors/lidars/lidar_wamv_sensor/scan` - LiDAR 데이터
- `/wamv/sensors/gps/gps/fix` - GPS 위치
- `/wamv/sensors/imu/imu/data` - IMU (헤딩, 각속도)
- `/wamv/sensors/cameras/front_left_camera_sensor/image_raw` - 카메라
- `/vrx/waypoint` - 웨이포인트 (시각화에서 발행)

### 발행 토픽 (Publications)

- `/wamv/thrusters/left/thrust` - 좌측 스러스터
- `/wamv/thrusters/right/thrust` - 우측 스러스터
- `/vrx/control_output` - 제어 출력 (v, ω)
- `/vrx/control_mode` - 제어 모드
- `/vrx/mission_status` - 미션 상태
- `/vrx/obstacle_check_area` - 장애물 체크 영역
- `/vrx/los_target` - LOS 타겟 위치

---

## 🐛 문제 해결

### 1. 가상환경 활성화 안 됨

```bash
# setup_venv.sh 재실행
./setup_venv.sh
```

### 2. Config 파일을 찾을 수 없음

```bash
# config 디렉토리 확인
ls -la config/

# ConfigManager 테스트
python utils/config_manager.py
```

### 3. ONNX 모델 경로 오류

`config/mission_config.yaml` 수정:
```yaml
model:
  path: '/실제/모델/경로/model.onnx'
```

### 4. 웨이포인트가 설정 안 됨

- 최소 6개 웨이포인트 필요
- trajectory_viz.py가 실행 중인지 확인
- 시각화 창에서 클릭

### 5. 로봇이 멈춤

- LiDAR 데이터 수신 확인
- 제어 모드 확인 (`/vrx/control_mode`)
- 웨이포인트 도달 거리 확인 (기본: 15m)

---

## 📝 추가 문서

- [`config/README.md`](config/README.md) - Config 시스템 상세 가이드
- [`README_MISSION_SYSTEM.md`](README_MISSION_SYSTEM.md) - 미션 시스템 설명
- [`README_MODULAR.md`](README_MODULAR.md) - 모듈 구조 설명
- [`example_config_usage.py`](example_config_usage.py) - Config 사용 예제

---

## 🔗 관련 링크

- [VRX Documentation](https://github.com/osrf/vrx)
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## 📄 라이선스

VRX 시뮬레이션 프로젝트

---

## 👥 개발자

- 원본 코드: yuneyoungjun
- 모듈화 및 Config 시스템: Claude 3.5 Sonnet

---

## 📚 Quick Start

```bash
# 1. 가상환경 설정 (최초 1회)
./setup_venv.sh

# 2. 환경 활성화
source activate_venv.sh

# 3. ROS2 환경 (필요 시)
source /opt/ros/humble/setup.bash

# 4. 실행
python trajectory_viz.py          # 터미널 1
python main_mission.py            # 터미널 2

# 5. 웨이포인트 클릭 (최소 6개)
# 6. 미션 자동 시작!
```

**Happy Sailing! 🚤**
