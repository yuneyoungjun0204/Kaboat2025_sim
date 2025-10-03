# Configuration System

Config 기반 설정 관리 시스템

## 📁 파일 구조

```
config/
├── topics.yaml           # ROS2 토픽 설정
├── mission_config.yaml   # 미션 파라미터 설정
└── README.md            # 이 파일
```

## 🎯 목적

### 문제점
- 토픽명이 코드에 하드코딩됨
- 파라미터 변경 시 여러 파일 수정 필요
- 환경별 설정 관리 어려움

### 해결책
- **중앙 집중식 설정 관리**
- YAML 파일로 토픽명/파라미터 관리
- 코드 수정 없이 설정 변경 가능

## 📋 설정 파일

### 1. `topics.yaml` - 토픽 설정

ROS2 토픽명을 계층적으로 관리:

```yaml
sensors:
  lidar: '/wamv/sensors/lidars/lidar_wamv_sensor/scan'
  gps: '/wamv/sensors/gps/gps/fix'
  imu: '/wamv/sensors/imu/imu/data'

actuators:
  thrusters:
    left: '/wamv/thrusters/left/thrust'
    right: '/wamv/thrusters/right/thrust'

vrx:
  waypoint: '/vrx/waypoint'
  control_output: '/vrx/control_output'
  # ... 기타 토픽
```

**장점:**
- 토픽명 변경 시 한 곳만 수정
- 시뮬레이션/실제 환경 전환 쉬움
- 네임스페이스 관리 용이

### 2. `mission_config.yaml` - 미션 설정

미션 파라미터 및 제어 설정:

```yaml
model:
  path: '/path/to/model.onnx'
  # 환경변수 지원: '${HOME}/models/model.onnx'

control:
  thrust_scale: 800
  v_scale: 1.0
  w_scale: -1.0

missions:
  gate:
    completion_threshold: 15.0
    kp_heading: 2.0

  circle:
    radius: 10.0
    direction: 'clockwise'

  avoid:
    boat_width: 2.2
    filter_alpha: 0.35
```

**장점:**
- 파라미터 튜닝 쉬움
- 환경변수 지원 (`${VAR}`)
- 미션별 설정 분리

## 🔧 사용 방법

### 1. ConfigManager 임포트

```python
from utils import ConfigManager, get_config
```

### 2. 인스턴스 생성

```python
# 방법 1: 직접 생성
config = ConfigManager()

# 방법 2: 전역 싱글톤 (권장)
config = get_config()
```

### 3. 토픽 가져오기

```python
# 센서 토픽
lidar_topic = config.get_sensor_topic('lidar')
# '/wamv/sensors/lidars/lidar_wamv_sensor/scan'

# 액추에이터 토픽
left_thrust = config.get_actuator_topic('thrusters', 'left')
# '/wamv/thrusters/left/thrust'

# VRX 토픽
waypoint_topic = config.get_vrx_topic('waypoint')
# '/vrx/waypoint'

# 일반 토픽 (계층 구조)
topic = config.get_topic('sensors', 'camera', 'front_left')
```

### 4. 파라미터 가져오기

```python
# 제어 파라미터 전체
control_params = config.get_control_params()
thrust_scale = control_params['thrust_scale']

# 특정 파라미터
thrust_scale = config.get_param('control', 'thrust_scale')

# 기본값 지정
alpha = config.get_param('missions', 'avoid', 'filter_alpha', default=0.35)

# 미션 파라미터
gate_params = config.get_mission_params('gate')
threshold = gate_params['completion_threshold']

# 모델 경로
model_path = config.get_model_path()
```

### 5. 타이머 설정

```python
# Hz를 자동으로 초 단위로 변환
control_period = config.get_timer_period('control_update')
# 100Hz → 0.01s

timer = self.create_timer(control_period, callback)
```

## 🔄 환경변수 사용

YAML에서 환경변수 사용 가능:

```yaml
model:
  path: '${HOME}/vrx_ws/models/model.onnx'
  # 자동으로 /home/user/vrx_ws/models/model.onnx로 치환
```

지원 형식: `${VAR_NAME}`

## 📝 실제 예제

### Before (하드코딩)

```python
class VRXController(Node):
    def __init__(self):
        # 토픽 하드코딩
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/wamv/sensors/lidars/lidar_wamv_sensor/scan',  # ❌
            self.callback, 10
        )

        # 파라미터 하드코딩
        self.thrust_scale = 800  # ❌
        self.model_path = '/home/user/model.onnx'  # ❌
```

### After (Config 사용)

```python
from utils import get_config

class VRXController(Node):
    def __init__(self):
        config = get_config()

        # 토픽 Config에서
        self.lidar_sub = self.create_subscription(
            LaserScan,
            config.get_sensor_topic('lidar'),  # ✅
            self.callback,
            config.get_qos('sensor_data')  # ✅
        )

        # 파라미터 Config에서
        self.thrust_scale = config.get_param('control', 'thrust_scale')  # ✅
        self.model_path = config.get_model_path()  # ✅
```

## 🎨 Config 변경 시나리오

### 시나리오 1: 토픽명 변경

**변경 전:**
```yaml
sensors:
  lidar: '/wamv/sensors/lidars/lidar_wamv_sensor/scan'
```

**변경 후:**
```yaml
sensors:
  lidar: '/robot/lidar/data'  # 새 토픽명
```

**결과:** 코드 수정 없이 모든 노드가 새 토픽 사용 ✅

### 시나리오 2: 파라미터 튜닝

**변경 전:**
```yaml
missions:
  avoid:
    filter_alpha: 0.35
```

**변경 후:**
```yaml
missions:
  avoid:
    filter_alpha: 0.2  # 더 부드러운 제어
```

**결과:** 코드 수정 없이 필터 계수 변경 ✅

### 시나리오 3: 환경 전환 (시뮬 → 실제)

**시뮬레이션 (`topics.yaml`):**
```yaml
sensors:
  gps: '/wamv/sensors/gps/gps/fix'
```

**실제 로봇 (`topics.yaml`):**
```yaml
sensors:
  gps: '/mavros/global_position/global'
```

**결과:** YAML 파일만 교체하면 환경 전환 완료 ✅

## 🛠️ 유틸리티 메서드

### 설정 출력 (디버깅)

```python
config = ConfigManager()
config.print_config()
```

### 설정 다시 로드

```python
config.reload()  # 또는
reload_config()  # 전역 설정 리로드
```

## ⚙️ QoS 관리

```yaml
qos:
  sensor_data: 10
  control_command: 10
  visualization: 10
```

```python
qos = config.get_qos('sensor_data')
```

## 🚀 마이그레이션 가이드

### 단계별 적용

1. **ConfigManager 임포트**
   ```python
   from utils import get_config
   ```

2. **인스턴스 생성**
   ```python
   config = get_config()
   ```

3. **토픽명 교체**
   ```python
   # Before
   '/wamv/sensors/lidars/lidar_wamv_sensor/scan'

   # After
   config.get_sensor_topic('lidar')
   ```

4. **파라미터 교체**
   ```python
   # Before
   self.thrust_scale = 800

   # After
   self.thrust_scale = config.get_param('control', 'thrust_scale')
   ```

5. **검증**
   - 테스트 실행
   - 토픽 연결 확인
   - 파라미터 동작 확인

## 📚 참고 자료

- `example_config_usage.py`: 전체 사용 예제
- `utils/config_manager.py`: ConfigManager 구현
- `config/topics.yaml`: 토픽 설정 템플릿
- `config/mission_config.yaml`: 미션 설정 템플릿

## 🎯 Best Practices

1. **전역 싱글톤 사용**: `get_config()` 권장
2. **기본값 지정**: `get_param(..., default=value)`
3. **환경변수 활용**: 절대 경로 대신 `${HOME}` 등 사용
4. **계층 구조 유지**: 토픽/파라미터를 논리적으로 그룹화
5. **주석 추가**: YAML에 설명 주석 포함

## ❓ FAQ

**Q: 기존 하드코딩된 값은 어떻게?**
A: 점진적으로 마이그레이션. 우선 주요 토픽/파라미터부터 Config로 이동

**Q: Config 파일 위치 변경 가능?**
A: `ConfigManager(config_dir='/custom/path')` 로 지정

**Q: 환경변수가 없으면?**
A: 원래 문자열 그대로 반환 (예: `${UNDEFINED}`)

**Q: 성능 영향은?**
A: 초기화 시 한 번만 로드, 런타임 영향 거의 없음

## 🔗 관련 파일

- `utils/config_manager.py`
- `example_config_usage.py`
- `main_mission.py` (적용 예정)
- `main_onnx_v5_final_refactored.py` (적용 예정)
