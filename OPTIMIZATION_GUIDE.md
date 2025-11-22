# 🚀 VRX 최적화 설정 가이드

## 📋 목차
1. [탐지 성능 저하 원인](#탐지-성능-저하-원인)
2. [해상도 및 매핑 정보](#해상도-및-매핑-정보)
3. [최적화 설정 방법](#최적화-설정-방법)
4. [프로파일 사용법](#프로파일-사용법)
5. [개별 설정 조정](#개별-설정-조정)

---

## 🔍 탐지 성능 저하 원인

기존 코드에서 **탐지력이 약해진 주요 원인**:

### 1. Detection Threshold가 너무 높음 ⚠️
```python
# config.py (기존): 0.0000004
# 실제 사용 (detection_system_optimized.py): 0.0065 (16,250배 높음!)
```
→ **대부분의 탐지가 필터링됨**

### 2. Box 크기 필터가 너무 엄격함 ⚠️
```python
# config.py (기존): min_box_area=2, max_box_area=800000
# 실제 사용: min_box_area=500, max_box_area=80000
```
→ **작은 객체들이 전부 걸러짐**

### 3. Frame Skip ⏭️
- Detection: 2프레임마다 실행 (50% 감소)
- Depth: 1~3프레임마다 실행

---

## 📐 해상도 및 매핑 정보

### 현재 이미지 처리 해상도
```
원본 이미지:           1280x720
  ↓
Detection 입력:        1280x720 (원본) 또는 축소
  ↓
Depth 전처리 입력:     416x320 (4배 축소)
  ↓
Depth 모델 내부 처리:  256x256
  ↓
Depth 최종 출력:       1280x720 (bilinear upscale)
```

### 좌표 매핑 순서
1. Detection bbox (원본 좌표) → 중심점 계산
2. 원본 좌표 → Depth map 좌표 변환 (스케일 계산)
3. Depth map에서 깊이 추출 (spatial smoothing 적용)
4. **최종 결과는 모두 원본 해상도 좌표로 반환**

→ **해상도가 다르더라도 자동으로 매핑됨!**

---

## ⚙️ 최적화 설정 방법

### 1. config.py 열기
```bash
nano /home/ansl/Real_ka/Kaboat2025_sim/utils/config.py
```

### 2. `OptimizationConfig` 클래스 찾기
파일 하단의 `class OptimizationConfig:` 섹션

---

## 🎯 프로파일 사용법

### 빠른 설정: 3가지 프리셋

#### 1️⃣ 최고 정확도 (느림, 탐지력 최대)
```python
# config.py 또는 실행 코드에서
Constants.OptimizationConfig.apply_profile('max_accuracy')
```

**설정 내용:**
- Detection threshold: 0.001 (낮춤)
- Min box area: 100 (낮춤)
- Frame skip: 1 (모든 프레임)
- Detection 입력: 원본 해상도
- Depth 모델 입력: 384x384

**사용 시기:**
- 탐지력이 중요한 미션 (Dock, Circle)
- 작은 객체를 찾아야 할 때
- 속도가 느려도 괜찮을 때

---

#### 2️⃣ 균형 (기본 권장)
```python
Constants.OptimizationConfig.apply_profile('balanced')
```

**설정 내용:**
- Detection threshold: 0.003
- Min box area: 300
- Frame skip: 1
- Detection 입력: 원본 해상도
- Depth 모델 입력: 256x256
- torch.compile: ON
- Mixed Precision: ON

**사용 시기:**
- 대부분의 상황 (기본값)
- 탐지력과 속도 균형

---

#### 3️⃣ 최고 속도 (빠름, 부정확)
```python
Constants.OptimizationConfig.apply_profile('max_speed')
```

**설정 내용:**
- Detection threshold: 0.0065 (높음)
- Min box area: 500 (높음)
- Detection frame skip: 3 (33%만 처리)
- Depth frame skip: 2
- Detection 입력: 416x320 (축소)

**사용 시기:**
- 속도가 중요한 미션 (Obstacle Avoid)
- 탐지가 필요 없는 구간
- 프레임률이 낮을 때

---

## 🔧 개별 설정 조정

### 탐지력을 높이려면 (정확도 ↑, 속도 ↓)

#### 1. Detection Threshold 낮추기
```python
# utils/config.py
class OptimizationConfig:
    DETECTION_THRESHOLD = 0.001  # 기본값: 0.0065
    # 0.001-0.003 권장 (낮을수록 더 많이 탐지)
```

#### 2. Box 크기 필터 완화
```python
MIN_BOX_AREA = 100  # 기본값: 500
MAX_BOX_AREA = 80000  # 기본값: 80000
```

#### 3. Frame Skip 끄기 (모든 프레임 처리)
```python
DETECTION_FRAME_SKIP = 1  # 기본값: 2
DEPTH_FRAME_SKIP = 1      # 기본값: 1
```

#### 4. Detection 해상도 원본 사용
```python
DETECTION_RESOLUTION = None  # 원본 1280x720
```

---

### 속도를 높이려면 (정확도 ↓, 속도 ↑)

#### 1. Frame Skip 증가
```python
DETECTION_FRAME_SKIP = 3  # 3-4 권장
DEPTH_FRAME_SKIP = 2
```

#### 2. Detection 해상도 낮추기
```python
DETECTION_RESOLUTION = 'max_speed'  # 416x320
```

#### 3. Depth 모델 입력 크기 감소
```python
DEPTH_MODEL_INPUT_SIZE = 256  # 384에서 256으로
```

#### 4. torch.compile 및 Mixed Precision 활성화
```python
USE_TORCH_COMPILE = True
USE_MIXED_PRECISION = True
```

---

## 🎛️ 전체 최적화 설정 예시

### 예시 1: 도킹 미션용 (탐지력 최대)
```python
class OptimizationConfig:
    # 프로파일
    # apply_profile('max_accuracy')  # 또는 수동 설정:

    # 탐지 파라미터
    DETECTION_THRESHOLD = 0.001      # 낮춤
    MIN_BOX_AREA = 100               # 낮춤
    MAX_BOX_AREA = 80000

    # Frame skip
    DETECTION_FRAME_SKIP = 1         # 모든 프레임
    DEPTH_FRAME_SKIP = 1

    # 해상도
    DETECTION_RESOLUTION = None      # 원본
    DEPTH_MODEL_INPUT_SIZE = 384     # 고품질

    # 최적화
    USE_TORCH_COMPILE = False        # 정확도 우선
    USE_MIXED_PRECISION = False
```

---

### 예시 2: 장애물 회피용 (속도 최대)
```python
class OptimizationConfig:
    # 프로파일
    # apply_profile('max_speed')  # 또는 수동 설정:

    # 탐지 파라미터
    DETECTION_THRESHOLD = 0.0065     # 높음
    MIN_BOX_AREA = 500
    MAX_BOX_AREA = 80000

    # Frame skip
    DETECTION_FRAME_SKIP = 3         # 33%만 처리
    DEPTH_FRAME_SKIP = 2

    # 해상도
    DETECTION_RESOLUTION = 'max_speed'  # 축소
    DEPTH_MODEL_INPUT_SIZE = 256

    # 최적화
    USE_TORCH_COMPILE = True
    USE_MIXED_PRECISION = True
```

---

## 🔌 모듈 선택

### Depth Estimator 모듈
```python
DEPTH_MODULE = 'ultra'  # 'ultra', 'optimized', 'standard'
```

- **`'ultra'`**: UltraDepthEstimator (TensorRT 지원, 5-10배 빠름) ✅ 권장
- **`'optimized'`**: OptimizedDepthEstimator (기본 최적화, 2-3배 빠름)
- **`'standard'`**: MiDaSHybridDepthEstimator (느림, 가장 정확)

### Detection System 모듈
```python
DETECTION_MODULE = 'optimized'  # 'optimized', 'standard'
```

- **`'optimized'`**: OptimizedDetectionSystem (멀티스레딩, 빠름) ✅ 권장
- **`'standard'`**: DetectionSystem (기본, 느림)

---

## 🔥 Jetson 하드웨어 최적화

```python
# Jetson 전력 모드 (MAXN)
JETSON_POWER_OPTIMIZATION = True  # nvpmodel -m 0, jetson_clocks

# CUDA 최적화
CUDA_OPTIMIZATION = True  # cuDNN benchmark, TF32

# PyTorch 최적화
PYTORCH_OPTIMIZATION = True  # JIT, gradient 비활성화
```

**효과:**
- Jetson 전력 최적화: 1.2-1.5배 속도 향상
- CUDA + PyTorch: 1.3-2배 속도 향상
- **총 예상: 1.5-2.5배 속도 향상**

---

## 🧪 설정 확인 방법

### Python에서 해상도 정보 출력
```python
from utils.config import Constants

# 해상도 매핑 정보 출력
Constants.OptimizationConfig.print_resolution_info()
```

### 실행 로그 확인
시스템 초기화 시 자동으로 다음 정보가 출력됩니다:
```
✓ 객체 탐지 시스템 초기화 완료
  - Detection threshold: 0.003
  - Box area: 300 - 80000
  - Frame skip: Detection=1, Depth=1
  - Spatial smoothing: True (kernel=31)
✓ Detection 입력: 원본 해상도 (1280x720)
✓ Depth 전처리: max_speed
```

---

## 🆘 문제 해결

### 1. 탐지가 전혀 안 됨
```python
# Detection threshold를 최대한 낮추기
DETECTION_THRESHOLD = 0.0001  # 매우 낮음

# Box 크기 필터 완화
MIN_BOX_AREA = 10
```

### 2. 너무 많이 탐지됨 (False Positive)
```python
# Detection threshold 높이기
DETECTION_THRESHOLD = 0.01

# Box 크기 필터 강화
MIN_BOX_AREA = 800
```

### 3. 속도가 너무 느림
```python
# 프로파일 적용
Constants.OptimizationConfig.apply_profile('max_speed')

# 또는 Frame skip 증가
DETECTION_FRAME_SKIP = 4
```

### 4. Depth가 정확하지 않음
```python
# Spatial smoothing 커널 크기 증가
SPATIAL_KERNEL_SIZE = 51  # 기본값: 31

# Temporal filtering 강화
TEMPORAL_FILTER_ALPHA = 0.5  # 기본값: 0.3
```

---

## 📊 성능 비교표

| 설정 | Detection FPS | 탐지 정확도 | 권장 사용 |
|------|--------------|------------|----------|
| max_accuracy | 5-8 FPS | ★★★★★ | Dock, Circle |
| balanced | 10-15 FPS | ★★★★☆ | 대부분 |
| max_speed | 20-30 FPS | ★★★☆☆ | Obstacle Avoid |

---

## 🚀 빠른 시작

### 1단계: 프로파일 선택
```bash
nano /home/ansl/Real_ka/Kaboat2025_sim/utils/config.py
```

### 2단계: 프로파일 적용 (선택)
```python
# config.py 맨 끝에 추가 (선택사항)
# Constants.OptimizationConfig.apply_profile('balanced')
```

### 3단계: 시스템 실행
```bash
cd /home/ansl/Real_ka/Kaboat2025_sim
python3 main_avoid.py  # 또는 다른 메인 파일
```

### 4단계: 로그 확인
초기화 로그에서 설정이 제대로 적용됐는지 확인

---

## 💡 팁

1. **처음에는 `balanced` 프로파일로 시작**
2. **탐지가 약하면 `DETECTION_THRESHOLD` 낮추기 (0.001-0.003)**
3. **속도가 느리면 `FRAME_SKIP` 증가 (2-4)**
4. **TensorRT 엔진 사용 시 5-10배 속도 향상** (depth_fp16.engine)
5. **실시간으로 설정을 바꾸려면 코드에서 `update_parameters()` 호출**

---

## 📝 참고

- 모든 설정은 `/home/ansl/Real_ka/Kaboat2025_sim/utils/config.py`에 있음
- 설정 변경 후 재실행 필요 (재컴파일 불필요)
- 자세한 파라미터 설명은 config.py 주석 참고
