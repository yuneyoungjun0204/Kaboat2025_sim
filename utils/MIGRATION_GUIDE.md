# Utils 폴더 구조 재구성 가이드

## 새로운 폴더 구조

```
utils/
├── core/              # 핵심 유틸리티
│   ├── config.py
│   ├── helpers.py
│   ├── system_factory.py
│   ├── super_optimizer.py
│   └── jetson_optimizer.py
├── sensors/           # 센서 관련
│   ├── sensor_callbacks.py
│   ├── sensor_preprocessing.py
│   ├── depth_estimation.py
│   ├── depth_estimation_optimized.py
│   ├── depth_estimation_ultra.py
│   └── depth_filter.py
├── detection/         # 탐지 관련
│   ├── detection_system.py
│   ├── detection_system_optimized.py
│   └── imm_pdaf_tracker.py
├── control/           # 제어 관련
│   ├── avoid_control.py
│   ├── thruster_allocation.py
│   └── onnx_controller.py
├── mission/           # 미션 관련
│   ├── mission_strategies_new.py
│   ├── mission_control.py
│   ├── mission_executor.py
│   ├── waypoint_manager.py
│   └── parameter_manager.py
├── visualization/     # 시각화 관련
│   ├── visualization_system.py
│   ├── viz_components.py
│   └── image_preprocessor.py
└── communication/     # 통신 관련
    ├── ros_communication.py
    └── px4_adapter.py
```

## 파일 이동 방법

### 방법 1: Python 스크립트 사용 (권장)

```bash
cd utils
python reorganize_files.py
```

### 방법 2: 수동 이동

각 파일을 해당 폴더로 이동:
- `config.py` → `core/config.py`
- `helpers.py` → `core/helpers.py`
- `system_factory.py` → `core/system_factory.py`
- `super_optimizer.py` → `core/super_optimizer.py`
- `jetson_optimizer.py` → `core/jetson_optimizer.py`
- `sensor_callbacks.py` → `sensors/sensor_callbacks.py`
- `sensor_preprocessing.py` → `sensors/sensor_preprocessing.py`
- `depth_estimation.py` → `sensors/depth_estimation.py`
- `depth_estimation_optimized.py` → `sensors/depth_estimation_optimized.py`
- `depth_estimation_ultra.py` → `sensors/depth_estimation_ultra.py`
- `depth_filter.py` → `sensors/depth_filter.py`
- `detection_system.py` → `detection/detection_system.py`
- `detection_system_optimized.py` → `detection/detection_system_optimized.py`
- `imm_pdaf_tracker.py` → `detection/imm_pdaf_tracker.py`
- `avoid_control.py` → `control/avoid_control.py`
- `thruster_allocation.py` → `control/thruster_allocation.py`
- `onnx_controller.py` → `control/onnx_controller.py`
- `mission_strategies_new.py` → `mission/mission_strategies_new.py`
- `mission_control.py` → `mission/mission_control.py`
- `mission_executor.py` → `mission/mission_executor.py`
- `waypoint_manager.py` → `mission/waypoint_manager.py`
- `parameter_manager.py` → `mission/parameter_manager.py`
- `visualization_system.py` → `visualization/visualization_system.py`
- `viz_components.py` → `visualization/viz_components.py`
- `image_preprocessor.py` → `visualization/image_preprocessor.py`
- `ros_communication.py` → `communication/ros_communication.py`
- `px4_adapter.py` → `communication/px4_adapter.py`

## 임포트 경로 변경

### 변경 전
```python
from utils.config import Constants
from utils.helpers import normalize_heading
from utils.mission_strategies_new import MissionManager
```

### 변경 후
```python
from utils.core.config import Constants
from utils.core.helpers import normalize_heading
from utils.mission.mission_strategies_new import MissionManager
```

또는 `utils/__init__.py`를 통해:
```python
from utils import Constants, normalize_heading, MissionManager
```

## 주요 임포트 매핑

| 이전 경로 | 새 경로 |
|---------|--------|
| `utils.config` | `utils.core.config` |
| `utils.helpers` | `utils.core.helpers` |
| `utils.system_factory` | `utils.core.system_factory` |
| `utils.sensor_callbacks` | `utils.sensors.sensor_callbacks` |
| `utils.sensor_preprocessing` | `utils.sensors.sensor_preprocessing` |
| `utils.depth_estimation` | `utils.sensors.depth_estimation` |
| `utils.detection_system` | `utils.detection.detection_system` |
| `utils.avoid_control` | `utils.control.avoid_control` |
| `utils.thruster_allocation` | `utils.control.thruster_allocation` |
| `utils.mission_strategies_new` | `utils.mission.mission_strategies_new` |
| `utils.waypoint_manager` | `utils.mission.waypoint_manager` |
| `utils.visualization_system` | `utils.visualization.visualization_system` |
| `utils.ros_communication` | `utils.communication.ros_communication` |
| `utils.px4_adapter` | `utils.communication.px4_adapter` |

## 확인 사항

파일 이동 후 다음을 확인하세요:
1. 모든 파일이 올바른 폴더에 있는지
2. 각 폴더의 `__init__.py`가 올바르게 설정되어 있는지
3. `utils/__init__.py`가 새 구조를 반영하고 있는지
4. 외부 파일(`Main_MCP.py`, `trajectory_viz.py`)의 임포트가 수정되었는지

