# VRX 시스템 리팩토링 가이드

## 개요

VRX 시스템을 모듈화하고 개발자 친화적으로 개선했습니다. 주요 변경사항:

1. ✅ **중앙 집중식 설정 관리** - 모든 ROS topics와 파일 경로를 `config.py`에서 관리
2. ✅ **Factory 패턴 도입** - 시스템 초기화 간소화
3. ✅ **타입 힌트 추가** - 코드 안정성 및 IDE 지원 향상
4. ✅ **코드 라인 수 감소** - Main_MCP.py에서 68라인 감소

---

## 주요 변경 사항

### 1. `utils/config.py` - 중앙 집중식 설정

#### 1.1 파일 경로 관리 (새로 추가)

```python
from utils import Constants

# ONNX 모델 경로 가져오기
model_path = Constants.Paths.get_onnx_model_path()

# NanoOWL 경로
nanoowl_path = Constants.Paths.NANOOWL_DIR

# 프로젝트 루트
project_root = Constants.Paths.PROJECT_ROOT
```

**장점:**
- ✅ 경로 변경 시 한 곳만 수정
- ✅ 자동 fallback (여러 모델 중 존재하는 것 사용)
- ✅ 환경별 설정 쉽게 관리

#### 1.2 ROS2 토픽 관리

```python
# 센서 입력 토픽
camera_topic = Constants.Topics.CAMERA_IMAGE
lidar_topic = Constants.Topics.LIDAR_SCAN
gps_topic = Constants.Topics.GPS_FIX
imu_topic = Constants.Topics.IMU_DATA

# 제어 출력 토픽
left_thrust_topic = Constants.Topics.LEFT_THRUST
right_thrust_topic = Constants.Topics.RIGHT_THRUST

# 상태 토픽
mission_status_topic = Constants.Topics.MISSION_STATUS
detections_topic = Constants.Topics.DETECTIONS
```

**토픽 이름 변경 방법:**
```python
# config.py에서만 수정
class Topics:
    CAMERA_IMAGE = '/wamv/sensors/cameras/my_new_camera/image_raw'  # 변경
    # ... 나머지는 자동으로 반영됨
```

#### 1.3 ROS2 큐 사이즈 관리 (새로 추가)

```python
# 센서 데이터
queue_size = Constants.QueueSizes.SENSOR  # 10

# 제어 명령
queue_size = Constants.QueueSizes.CONTROL  # 10

# 상태 정보
queue_size = Constants.QueueSizes.STATUS  # 10
```

---

### 2. `utils/system_factory.py` - Factory 패턴

#### 2.1 기본 사용법

**이전 방식 (Main_MCP.py, 68라인):**
```python
def _init_components(self):
    self.depth_estimator = MiDaSHybridDepthEstimator()
    self.detection_system = DetectionSystem(self.depth_estimator, device=self.device)
    sensor_manager = SensorDataManager()
    self.sensor_handler = SensorCallbackHandler(self.bridge, sensor_manager, self.get_logger())
    self.avoidance_controller = AvoidanceController(
        boat_width=Constants.BOAT_WIDTH,
        boat_height=Constants.BOAT_HEIGHT,
        max_lidar_distance=Constants.MAX_LIDAR_DISTANCE,
        los_delta=Constants.LOS_DELTA,
        los_lookahead_min=Constants.LOS_LOOKAHEAD_MIN,
        los_lookahead_max=Constants.LOS_LOOKAHEAD_MAX,
        filter_alpha=Constants.FILTER_ALPHA
    )
    # ... 40줄 더 ...
```

**개선된 방식 (9라인!):**
```python
def __init__(self):
    super().__init__('vrx_mission_controller')
    self.bridge = CvBridge()

    # 🚀 Factory로 한 번에 생성
    factory = VRXSystemFactory(self, self.bridge, self.get_logger())
    components = factory.create_all_components()

    # 컴포넌트 할당
    self.depth_estimator = components['depth_estimator']
    self.detection_system = components['detection_system']
    self.sensor_handler = components['sensor_handler']
    # ... 나머지 할당
```

#### 2.2 개별 컴포넌트 생성

필요한 컴포넌트만 생성하고 싶을 때:

```python
factory = VRXSystemFactory(node, bridge, logger)

# 탐지 시스템만 생성
detection_system = factory.create_detection_system()

# 센서 시스템만 생성
sensor_manager, sensor_handler = factory.create_sensor_system()

# 미션 시스템만 생성
mission_manager, waypoint_manager, mission_executor = \
    factory.create_mission_system(avoidance_controller)
```

#### 2.3 QuickStart 헬퍼 사용

가장 간단한 방법:

```python
from utils import QuickStart

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # 한 줄로 모든 컴포넌트 생성!
        components = QuickStart.setup_vrx_system(self)

        self.detection_system = components['detection_system']
        self.mission_manager = components['mission_manager']
        # ...
```

---

### 3. `utils/ros_communication.py` - 타입 힌트 추가

#### 3.1 타입 안전성

**이전:**
```python
def publish_thrust_commands(self, left_thrust, right_thrust):
    """스러스터 명령 발행"""
    # ...
```

**개선:**
```python
def publish_thrust_commands(self, left_thrust: float, right_thrust: float) -> None:
    """
    스러스터 명령 발행

    Args:
        left_thrust: 좌측 스러스터 명령 (-2000 ~ 2000)
        right_thrust: 우측 스러스터 명령 (-2000 ~ 2000)
    """
    # ...
```

**장점:**
- ✅ IDE 자동완성 지원
- ✅ 타입 체크 (mypy, pyright 등)
- ✅ 명확한 API 문서

---

## 마이그레이션 가이드

### 기존 코드 업데이트 방법

#### 1. 하드코딩된 토픽명 제거

**이전:**
```python
self.create_subscription(Image, '/wamv/sensors/cameras/front_left_camera_sensor/image_raw', callback, 10)
```

**개선:**
```python
from utils import Constants

self.create_subscription(
    Image,
    Constants.Topics.CAMERA_IMAGE,
    callback,
    Constants.QueueSizes.SENSOR
)
```

#### 2. 하드코딩된 경로 제거

**이전:**
```python
model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray.onnx'
```

**개선:**
```python
from utils import Constants

model_path = Constants.Paths.get_onnx_model_path()  # 자동으로 존재하는 모델 선택
```

#### 3. Factory 사용으로 초기화 간소화

**이전:**
```python
# 20-30줄의 초기화 코드
```

**개선:**
```python
from utils import VRXSystemFactory

factory = VRXSystemFactory(self, bridge, logger)
components = factory.create_all_components()
```

---

## 새로운 기능 추가 방법

### 1. 새 ROS 토픽 추가

**Step 1:** `utils/config.py`에 토픽 추가
```python
class Topics:
    # 기존 토픽들...

    # 새 토픽 추가
    MY_NEW_TOPIC = '/vrx/my_new_topic'
```

**Step 2:** `utils/ros_communication.py`에서 사용
```python
def setup_publishers(self):
    # ...
    self.publishers['my_new'] = self.node.create_publisher(
        String, Constants.Topics.MY_NEW_TOPIC, Constants.QueueSizes.DEFAULT
    )
```

### 2. 새 컴포넌트 추가

**Step 1:** 컴포넌트 클래스 작성 (`utils/my_component.py`)
```python
class MyComponent:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
```

**Step 2:** Factory에 생성 메서드 추가 (`utils/system_factory.py`)
```python
def create_my_component(self) -> MyComponent:
    """내 컴포넌트 생성"""
    self.logger.info("내 컴포넌트 초기화 중...")
    component = MyComponent(
        param1=Constants.MY_PARAM1,
        param2=Constants.MY_PARAM2
    )
    self.logger.info("✓ 내 컴포넌트 초기화 완료")
    return component
```

**Step 3:** `create_all_components()`에 추가
```python
def create_all_components(self) -> Dict[str, Any]:
    # ... 기존 컴포넌트들

    # 내 컴포넌트 추가
    my_component = self.create_my_component()

    components = {
        # ...
        'my_component': my_component,
    }
    return components
```

---

## 성능 최적화

### 개선된 부분

1. **초기화 시간 단축**
   - Factory가 의존성을 자동 관리
   - 중복 초기화 방지

2. **메모리 효율**
   - 싱글톤 패턴 적용 가능
   - 불필요한 객체 생성 방지

3. **유지보수 비용 감소**
   - 코드 라인 수 68라인 감소
   - 변경 포인트 최소화

---

## 트러블슈팅

### Q1: ONNX 모델을 찾을 수 없다는 에러

```python
FileNotFoundError: ONNX 모델을 찾을 수 없습니다: /path/to/models
```

**해결:**
```python
# config.py에서 경로 확인
class Paths:
    ONNX_MODEL = MODELS_DIR / 'Ray.onnx'  # 파일명 확인

    # 또는 절대 경로 사용
    ONNX_MODEL = Path('/절대/경로/to/Ray.onnx')
```

### Q2: 토픽 이름 변경이 반영되지 않음

**원인:** Python import 캐싱

**해결:**
```bash
# Python 캐시 삭제
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# ROS2 워크스페이스 재빌드
cd ~/vrx_ws
colcon build --packages-select vrx
source install/setup.bash
```

### Q3: Factory에서 컴포넌트 초기화 실패

**원인:** 의존성 누락

**해결:**
```python
# 개별 컴포넌트 생성으로 디버깅
factory = VRXSystemFactory(node, bridge, logger)

try:
    depth_estimator = factory.create_depth_estimator()
    print("✓ 깊이 추정기 성공")
except Exception as e:
    print(f"✗ 깊이 추정기 실패: {e}")

try:
    detection_system = factory.create_detection_system()
    print("✓ 탐지 시스템 성공")
except Exception as e:
    print(f"✗ 탐지 시스템 실패: {e}")
```

---

## 베스트 프랙티스

### DO ✅

1. **항상 Constants 사용**
   ```python
   from utils import Constants
   topic = Constants.Topics.CAMERA_IMAGE  # ✅
   ```

2. **Factory로 컴포넌트 생성**
   ```python
   factory = VRXSystemFactory(node, bridge, logger)
   components = factory.create_all_components()  # ✅
   ```

3. **타입 힌트 사용**
   ```python
   def my_function(value: float) -> bool:  # ✅
       return value > 0.0
   ```

### DON'T ❌

1. **하드코딩된 토픽명**
   ```python
   topic = '/wamv/sensors/cameras/front_left_camera_sensor/image_raw'  # ❌
   ```

2. **하드코딩된 경로**
   ```python
   model_path = '/home/user/vrx_ws/src/vrx/models/Ray.onnx'  # ❌
   ```

3. **수동 초기화 (Factory 있을 때)**
   ```python
   # Factory가 있는데도 수동으로 초기화 ❌
   self.depth_estimator = MiDaSHybridDepthEstimator()
   self.detection_system = DetectionSystem(self.depth_estimator)
   # ...
   ```

---

## 예제 코드

### 완전한 새 노드 작성 예제

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from utils import VRXSystemFactory, Constants

class MyVRXNode(Node):
    def __init__(self):
        super().__init__('my_vrx_node')

        # 🚀 Factory로 시스템 초기화
        from cv_bridge import CvBridge
        bridge = CvBridge()
        factory = VRXSystemFactory(self, bridge, self.get_logger())

        # 필요한 컴포넌트만 생성
        components = factory.create_all_components()

        self.detection_system = components['detection_system']
        self.mission_manager = components['mission_manager']

        # ROS2 통신 설정
        from utils.ros_communication import ROSCommunicationManager
        self.ros_comm = ROSCommunicationManager(self)
        self.ros_comm.setup_publishers()

        # 타이머 설정
        self.timer = self.create_timer(
            Constants.MAIN_LOOP_PERIOD,
            self.control_loop
        )

        self.get_logger().info("✓ 노드 초기화 완료!")

    def control_loop(self):
        """제어 루프"""
        # 여기에 로직 작성
        self.ros_comm.publish_thrust_commands(1000.0, 1000.0)

def main(args=None):
    rclpy.init(args=args)
    node = MyVRXNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 요약

### 개선 전
- ❌ 하드코딩된 토픽명, 경로
- ❌ 68라인의 반복적인 초기화 코드
- ❌ 타입 힌트 부족
- ❌ 변경 포인트가 여러 곳에 분산

### 개선 후
- ✅ 중앙 집중식 설정 관리 (`config.py`)
- ✅ Factory 패턴으로 초기화 간소화 (9라인)
- ✅ 타입 힌트 및 docstring 추가
- ✅ 토픽/경로 변경 시 한 곳만 수정

### 다음 단계
1. 다른 파일들도 점진적으로 리팩토링
2. 유닛 테스트 추가
3. CI/CD 파이프라인 구축
