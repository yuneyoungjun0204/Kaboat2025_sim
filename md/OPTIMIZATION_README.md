# Jetson Orin Nano 최적화 가이드

## 개요

Jetson Orin Nano에서 **NanoOWL + MiDaS + IMM-PDAF** 파이프라인을 **2-5배 빠르게** 실행하기 위한 최적화 버전입니다.

### 주요 개선사항

| 기능 | 기존 시스템 | 최적화 시스템 | 개선 |
|------|------------|--------------|------|
| **멀티스레딩** | ❌ 순차 처리 | ✅ 병렬 처리 | ~40% 향상 |
| **Frame Skip** | ❌ 매 프레임 | ✅ 선택적 처리 | ~2-3배 향상 |
| **TensorRT** | ❌ PyTorch | ✅ FP16 TensorRT | ~2배 향상 |
| **메모리 최적화** | ❌ 기본 | ✅ 최적화 | 메모리 감소 |

### 예상 성능

| 모드 | FPS | 품질 | 용도 |
|------|-----|------|------|
| **Fast** | ~15-20 FPS | 낮음 | 빠른 프로토타입 |
| **Balanced** | ~10-15 FPS | 중간 | 권장 ⭐ |
| **Quality** | ~5-10 FPS | 높음 | 정밀 작업 |

---

## 설치

### 1. 기본 요구사항

```bash
# Python 패키지
pip3 install torch torchvision
pip3 install opencv-python
pip3 install numpy scipy

# NanoOWL (기존 설치 유지)
# MiDaS는 torch.hub로 자동 설치됨
```

### 2. TensorRT (선택, 2배 성능 향상)

```bash
# Jetson에 기본 설치되어 있음
python3 -c "import tensorrt; print(tensorrt.__version__)"

# pycuda 설치
pip3 install pycuda
```

---

## 사용법

### 빠른 시작

```python
from utils.optimized_pipeline import create_optimized_pipeline, MissionType
import cv2

# 1. 파이프라인 생성 (Balanced 모드 - 권장)
pipeline = create_optimized_pipeline(preset='balanced')

# 2. 이미지 처리
image = cv2.imread('test.jpg')
result = pipeline.process_frame(image, MissionType.PASS_BETWEEN_BUOYS)

# 3. 결과 사용
for obj in result['tracked_objects']:
    print(f"{obj['label']}: depth={obj['depth']:.2f}m")

# 4. 시각화
vis_image = pipeline.visualize_results(image, result)
cv2.imshow('Result', vis_image)
```

### 프리셋 선택

```python
# Fast 모드: 최대 속도
pipeline = create_optimized_pipeline(preset='fast')

# Balanced 모드: 균형 (권장)
pipeline = create_optimized_pipeline(preset='balanced')

# Quality 모드: 최고 품질
pipeline = create_optimized_pipeline(preset='quality')
```

### 커스텀 설정

```python
from utils.optimized_pipeline import OptimizedPipeline

pipeline = OptimizedPipeline(
    # Depth 설정
    depth_input_size=256,          # 입력 크기 (192, 256, 384)
    use_tensorrt_depth=False,      # TensorRT 사용

    # Detection 설정
    detection_frame_skip=2,        # 2프레임마다 detection
    depth_frame_skip=3,            # 3프레임마다 depth
    use_async=True,                # 비동기 처리

    # Tracking 설정
    fps=30.0,
    max_coast_frames=10
)
```

---

## TensorRT 최적화 (2배 성능 향상)

### 1. Depth 모델 변환

```python
from utils.optimized_pipeline import create_optimized_pipeline

# 파이프라인 생성
pipeline = create_optimized_pipeline()

# TensorRT 변환 (최초 1회, 5-10분 소요)
pipeline.export_depth_to_tensorrt(
    onnx_path='depth_model.onnx',
    engine_path='depth_model.engine'
)
```

또는 CLI:

```bash
# 1. ONNX 변환
python3 -c "
from utils.depth_estimation_optimized import OptimizedDepthEstimator
estimator = OptimizedDepthEstimator(input_size=256)
estimator.export_to_onnx('depth_model.onnx')
"

# 2. TensorRT 변환
trtexec --onnx=depth_model.onnx \
        --saveEngine=depth_model.engine \
        --fp16 \
        --workspace=2048
```

### 2. TensorRT 엔진 사용

```python
pipeline = create_optimized_pipeline(
    preset='balanced',
    use_tensorrt_depth=True,
    depth_engine_path='depth_model.engine'
)
```

---

## 성능 벤치마크

### 실행

```bash
# 기본 벤치마크
python3 benchmark_optimizations.py --frames 100

# Fast 모드 벤치마크
python3 benchmark_optimizations.py --frames 100 --preset fast

# TensorRT 벤치마크 (변환 후)
python3 benchmark_optimizations.py --frames 100 --tensorrt

# 최적화 시스템만 테스트
python3 benchmark_optimizations.py --frames 100 --skip-old
```

### 예상 결과

```
📊 성능 비교
====================================
Detection 평균:
  기존:   250.0ms
  최적화: 100.0ms
  개선:   📈 +60.0%

Total 평균:
  기존:   350.0ms
  최적화: 120.0ms
  개선:   📈 +65.7%

FPS:
  기존:   2.9
  최적화: 8.3
  개선:   📈 +186.2%
```

---

## 주요 최적화 기법

### 1. 멀티스레딩

```python
# Detection과 Depth를 별도 스레드에서 병렬 처리
# - Detection: 메인 스레드
# - Depth: 백그라운드 스레드
```

**효과**: ~30-40% 성능 향상

### 2. Frame Skip

```python
# 매 프레임마다 처리하지 않고 N프레임마다 처리
detection_frame_skip=2  # Detection은 2프레임마다
depth_frame_skip=3      # Depth는 3프레임마다
```

**효과**: ~2-3배 성능 향상

### 3. TensorRT FP16

```python
# FP32 → FP16 변환으로 2배 속도 향상
use_tensorrt_depth=True
```

**효과**: ~2배 성능 향상

### 4. 입력 해상도 감소

```python
# 384x384 → 256x256 (Depth estimation)
depth_input_size=256
```

**효과**: ~2배 성능 향상 (정확도 약간 감소)

### 5. ROI 처리 (선택)

```python
# 관심 영역만 처리
roi_enabled=True
roi_bounds=(100, 100, 540, 380)  # (x1, y1, x2, y2)
```

**효과**: ROI 크기에 따라 향상

---

## 기존 코드와의 통합

### 방법 1: 교체 (권장)

```python
# 기존
# from utils.detection_system import DetectionSystem
# from utils.depth_estimation import MiDaSHybridDepthEstimator

# 최적화
from utils.optimized_pipeline import create_optimized_pipeline

pipeline = create_optimized_pipeline(preset='balanced')
```

### 방법 2: 별칭 사용

```python
# utils/__init__.py에 추가
from .optimized_pipeline import OptimizedPipeline as DetectionSystem
from .depth_estimation_optimized import OptimizedDepthEstimator as MiDaSHybridDepthEstimator
```

---

## 문제 해결

### Q1. ImportError: No module named 'tensorrt'

**A**: TensorRT는 선택사항입니다. 없어도 PyTorch 모드로 작동합니다.

```python
# TensorRT 없이 사용
pipeline = create_optimized_pipeline(use_tensorrt_depth=False)
```

### Q2. CUDA out of memory

**A**: 입력 크기를 줄이거나 Frame skip을 늘리세요.

```python
pipeline = create_optimized_pipeline(
    preset='fast',  # 또는
    depth_input_size=192,
    detection_frame_skip=3
)
```

### Q3. Detection이 작동하지 않음

**A**: NanoOWL 경로를 확인하세요.

```python
# utils/config.py에서 NANOOWL_DIR 확인
from utils.config import Constants
print(Constants.Paths.NANOOWL_DIR)
```

### Q4. 성능이 개선되지 않음

**A**: 다음을 확인하세요:
1. CUDA 사용 확인: `torch.cuda.is_available()`
2. Jetson 전원 모드: `sudo nvpmodel -q` (최대 성능 모드 사용)
3. Frame skip 설정 확인
4. TensorRT 변환 완료 여부

---

## 고급 사용법

### 실시간 스트리밍

```python
import cv2
from utils.optimized_pipeline import create_optimized_pipeline, MissionType

pipeline = create_optimized_pipeline(preset='balanced')

cap = cv2.VideoCapture(0)  # 웹캠

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 처리
    result = pipeline.process_frame(frame, MissionType.PASS_BETWEEN_BUOYS)

    # 시각화
    vis_frame = pipeline.visualize_results(frame, result)

    # 표시
    cv2.imshow('Optimized Pipeline', vis_frame)

    # 30프레임마다 통계 출력
    if cv2.waitKey(1) & 0xFF == ord('s'):
        pipeline.print_performance_stats()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pipeline.cleanup()
```

### ROS 통합

```python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OptimizedNode:
    def __init__(self):
        self.pipeline = create_optimized_pipeline(preset='balanced')
        self.bridge = CvBridge()

        rospy.Subscriber('/camera/image_raw', Image, self.callback)

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        result = self.pipeline.process_frame(cv_image, MissionType.PASS_BETWEEN_BUOYS)

        # 결과 publish...
```

---

## 파일 구조

```
kaboat_backup/
├── utils/
│   ├── detection_system.py              # 기존 시스템
│   ├── detection_system_optimized.py    # 최적화 Detection
│   ├── depth_estimation.py              # 기존 Depth
│   ├── depth_estimation_optimized.py    # 최적화 Depth (TensorRT)
│   ├── optimized_pipeline.py            # 통합 파이프라인
│   └── imm_pdaf_tracker.py              # IMM-PDAF (변경 없음)
│
├── benchmark_optimizations.py           # 벤치마크 스크립트
├── OPTIMIZATION_README.md               # 이 문서
│
└── models/
    ├── depth_model.onnx                 # ONNX 모델 (생성 후)
    └── depth_model.engine               # TensorRT 엔진 (생성 후)
```

---

## 성능 팁

### Jetson 최대 성능 모드

```bash
# 최대 성능 모드 활성화
sudo nvpmodel -m 0
sudo jetson_clocks

# 확인
sudo nvpmodel -q
```

### 메모리 정리

```bash
# Swap 정리
sudo swapoff -a && sudo swapon -a

# 캐시 정리
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### CUDA 최적화

```python
# Python 코드에서
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

---

## 라이선스

기존 프로젝트의 라이선스를 따릅니다.

---

## 지원

문제가 발생하면 이슈를 등록하거나 벤치마크 결과를 공유해주세요!

**Happy Optimizing! 🚀**
