# 🚀 최적화 설정 빠른 시작 (5분 가이드)

## ⚠️ 탐지력이 약해진 이유

### 주요 원인 3가지
1. **Detection Threshold가 16,250배 높았음** (0.0000004 → 0.0065)
2. **Box 크기 필터가 너무 엄격함** (min: 2 → 500)
3. **Frame Skip** (2-3프레임마다만 처리)

→ **이제 config.py에서 모두 조정 가능!**

---

## 🎯 1분 해결법

### 탐지력 높이기 (가장 간단한 방법)

#### 방법 1: config.py 직접 수정
```bash
nano /home/ansl/Real_ka/Kaboat2025_sim/utils/config.py
```

다음 값들을 찾아서 수정:
```python
class OptimizationConfig:
    # 탐지 임계값 낮추기 (현재: 0.0065)
    DETECTION_THRESHOLD = 0.003  # 0.001-0.003 권장

    # 작은 박스도 탐지되도록
    MIN_BOX_AREA = 200  # 현재: 500

    # 모든 프레임 처리
    DETECTION_FRAME_SKIP = 1  # 현재: 2
```

저장 후 프로그램 재실행!

---

#### 방법 2: 프로파일 사용 (더 간단!)
```bash
nano /home/ansl/Real_ka/Kaboat2025_sim/utils/config.py
```

파일 맨 끝으로 가서 주석 해제:
```python
# 이 줄의 주석을 제거 (# 삭제)
Constants.OptimizationConfig.apply_profile('max_accuracy')
```

저장 후 재실행 → **끝!**

---

## 📐 해상도 정보

### 현재 이미지 처리 흐름
```
카메라 원본 (1280x720)
  ↓
Detection: 원본 1280x720 (탐지용)
  ↓
Depth: 416x320 (4배 축소, 속도 향상)
  ↓
Depth 모델: 256x256 (내부 처리)
  ↓
최종 Depth: 1280x720 (upscale)
```

### 좌표 매핑은 자동!
- Detection bbox (1280x720 좌표)
- → Depth map (스케일 자동 계산)
- → 최종 결과 (1280x720 좌표)

**걱정 마세요! 해상도가 달라도 자동으로 매핑됩니다.**

---

## ⚡ 3가지 프로파일

### 1️⃣ max_accuracy - 탐지력 최대 (느림)
```python
Constants.OptimizationConfig.apply_profile('max_accuracy')
```
**언제?** Dock, Circle 미션, 작은 객체 찾을 때

---

### 2️⃣ balanced - 균형 (권장)
```python
Constants.OptimizationConfig.apply_profile('balanced')
```
**언제?** 대부분의 상황 (기본값)

---

### 3️⃣ max_speed - 속도 최대 (부정확)
```python
Constants.OptimizationConfig.apply_profile('max_speed')
```
**언제?** Obstacle Avoid, 탐지 필요 없을 때

---

## 🔧 개별 설정 예시

### 예시 1: 탐지가 전혀 안 될 때
```python
# config.py
DETECTION_THRESHOLD = 0.0001  # 매우 낮춤
MIN_BOX_AREA = 10             # 매우 작은 박스도 탐지
DETECTION_FRAME_SKIP = 1      # 모든 프레임
```

### 예시 2: 속도가 너무 느릴 때
```python
DETECTION_FRAME_SKIP = 3      # 33%만 처리
DEPTH_FRAME_SKIP = 2
DETECTION_RESOLUTION = 'max_speed'  # 해상도 낮춤
```

### 예시 3: 탐지는 잘 되는데 False Positive 많을 때
```python
DETECTION_THRESHOLD = 0.01    # 높임
MIN_BOX_AREA = 800            # 큰 박스만
```

---

## 🎛️ 모든 최적화 켜기/끄기

### Jetson 하드웨어 최적화
```python
class OptimizationConfig:
    JETSON_POWER_OPTIMIZATION = True   # MAXN 모드
    CUDA_OPTIMIZATION = True           # cuDNN, TF32
    PYTORCH_OPTIMIZATION = True        # JIT, gradient 비활성화
```

### Detection 최적화
```python
    USE_ASYNC_DEPTH = True            # 백그라운드 Depth 처리
    USE_NANOOWL = True                # NanoOWL 사용
    USE_TORCH_COMPILE = True          # torch.compile (30-100% 빠름)
    USE_MIXED_PRECISION = True        # FP16 (1.5-2배 빠름)
```

### TensorRT
```python
    USE_TENSORRT = True               # 5-10배 속도 향상!
    TENSORRT_ENGINE_PATH = "/home/ansl/Real_ka/Kaboat2025_sim/depth_fp16.engine"
```

---

## 🏃 빠른 테스트

### 1단계: 프로파일 적용
```bash
nano /home/ansl/Real_ka/Kaboat2025_sim/utils/config.py
```

맨 끝에 추가:
```python
Constants.OptimizationConfig.apply_profile('balanced')
```

### 2단계: 실행
```bash
cd /home/ansl/Real_ka/Kaboat2025_sim
python3 main_avoid.py  # 또는 다른 메인 파일
```

### 3단계: 로그 확인
```
✓ 객체 탐지 시스템 초기화 완료
  - Detection threshold: 0.003        ← 이 값 확인!
  - Box area: 300 - 80000             ← 이 값 확인!
  - Frame skip: Detection=1, Depth=1  ← 이 값 확인!
```

---

## 📊 설정별 성능 비교

| 항목 | max_accuracy | balanced | max_speed |
|------|--------------|----------|-----------|
| Detection FPS | 5-8 | 10-15 | 20-30 |
| 탐지 정확도 | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Detection Threshold | 0.001 | 0.003 | 0.0065 |
| Min Box Area | 100 | 300 | 500 |
| Frame Skip | 1 | 1 | 3 |

---

## 🆘 문제 해결

### Q: 설정을 바꿨는데 반영이 안 돼요
**A:** 프로그램을 재실행하세요. config.py는 시작 시에만 로드됩니다.

### Q: 탐지가 여전히 약해요
**A:** `DETECTION_THRESHOLD`를 0.0001까지 낮춰보세요.

### Q: 속도가 너무 느려요
**A:** TensorRT 엔진 파일이 있나요? (`depth_fp16.engine`)
```bash
ls /home/ansl/Real_ka/Kaboat2025_sim/depth_fp16.engine
```
없으면 생성:
```bash
python3 convert_to_tensorrt.py
```

### Q: 어떤 파일을 사용하는지 모르겠어요
**A:** config.py에서 모듈 선택:
```python
DEPTH_MODULE = 'ultra'        # depth_estimation_ultra.py
# DEPTH_MODULE = 'optimized'  # depth_estimation_optimized.py
# DEPTH_MODULE = 'standard'   # depth_estimation.py
```

---

## 📝 전체 문서

자세한 내용은 `OPTIMIZATION_GUIDE.md` 참고

---

## ✅ 체크리스트

- [ ] config.py 열기
- [ ] `OptimizationConfig` 섹션 찾기
- [ ] 탐지력 높이기 위해 `DETECTION_THRESHOLD = 0.003` 설정
- [ ] `MIN_BOX_AREA = 200` 설정
- [ ] 프로그램 재실행
- [ ] 로그에서 설정 확인
- [ ] 탐지 성능 테스트

---

**🎉 이제 탐지력이 개선됐을 겁니다!**
