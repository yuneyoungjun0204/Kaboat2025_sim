# 🚀 Jetson Orin Nano 최적화 가이드

Main_MCP.py 실행 시 **자동으로 적용되는 최적화**와 **수동 설정**을 안내합니다.

## 📊 예상 성능 향상

| 최적화 단계 | FPS | 속도 향상 | 자동 적용 |
|------------|-----|---------|----------|
| 기본 설정 | 5-8 FPS | 1x | - |
| + Jetson 전력 최대화 | 8-12 FPS | 1.5x | ✅ 자동 |
| + CUDA/PyTorch 최적화 | 10-15 FPS | 2x | ✅ 자동 |
| + TensorRT FP16 | 20-30 FPS | 5x | ⚠️ 수동 |
| + TensorRT INT8 | 30-40 FPS | 8x | ⚠️ 수동 |

---

## ✅ 자동 적용되는 최적화

`Main_MCP.py` 실행 시 자동으로 적용됩니다:

### 1. **Jetson 전력 모드 최적화** (1.2-1.5배)
- MAXN 모드 활성화 (최대 성능)
- CPU/GPU 클럭 최대화
- **주의**: sudo 권한 필요 (아래 설정 참고)

### 2. **CUDA 최적화** (1.1-1.2배)
- cuDNN 벤치마크 활성화
- TF32 활성화 (Ampere)
- CUDA 캐시 최적화

### 3. **PyTorch 최적화** (1.1-1.2배)
- Gradient 비활성화 (추론 모드)
- JIT OneDNN fusion
- 메모리 할당 최적화

### 4. **메모리 최적화**
- CUDA 캐시 정리
- Python garbage collection

---

## 🔧 수동 설정 (최초 1회)

### Step 1: sudo 권한 설정 (권장)

프로그램 실행 시 자동으로 Jetson 성능이 최적화되도록 설정:

```bash
cd /home/ansl/Real_ka/Kaboat2025_sim
./setup_jetson_sudo.sh
```

**또는 수동으로:**

```bash
# 매번 실행 시 수동으로 (sudo 권한 설정 안 한 경우)
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Step 2: TensorRT 엔진 생성 (최초 1회)

#### 옵션 A: FP16 (권장, 5-10분)
```bash
python3 convert_to_tensorrt.py --precision fp16
```

**성능**: 20-30 FPS (5배 향상)

#### 옵션 B: INT8 (최고 속도, 10-15분)
```bash
python3 convert_to_tensorrt.py --precision int8 --engine-path depth_int8.engine
```

**성능**: 30-40 FPS (8배 향상)
**주의**: 정확도가 약간 떨어질 수 있음 (VRX에서는 충분)

---

## 🎯 실행 방법

### 1단계: 최초 설정 (한 번만)

```bash
# 1. sudo 권한 설정
./setup_jetson_sudo.sh

# 2. TensorRT 엔진 생성 (FP16 권장)
python3 convert_to_tensorrt.py --precision fp16
```

### 2단계: 프로그램 실행

```bash
python3 Main_MCP.py
```

실행 시 자동으로 다음이 적용됩니다:
- ✅ Jetson 전력 최대화 (MAXN 모드)
- ✅ CUDA/PyTorch 최적화
- ✅ TensorRT 엔진 자동 로드 (있는 경우)

---

## 📈 최적화 확인

프로그램 시작 시 다음과 같은 메시지가 출력됩니다:

```
======================================================================
🚀 Jetson Orin Nano 최적화 시작
======================================================================

⚡ Jetson 전력 모드 최적화 중...
  ✅ MAXN 모드 활성화 (최대 성능)
  ✅ Jetson Clocks 활성화 (CPU/GPU 최대 클럭)
  📊 현재 전력 모드: NV Power Mode: MAXN
  ✅ 전력 모드 최적화 완료 (1.2-1.5배 성능 향상)

🚀 CUDA 최적화 설정 중...
  ✅ cuDNN 벤치마크 활성화
  ✅ TF32 활성화 (Ampere)
  ✅ CUDA 최적화 완료

...

🔥 TensorRT FP16 엔진 발견! (5-10배 속도 향상)

======================================================================
✅ Jetson 최적화 완료!
   예상 성능 향상: 1.5-2.5배 (전력 모드 + CUDA + PyTorch)
======================================================================
```

---

## 🔍 성능 벤치마크

### 벤치마크 실행:

```bash
# 기본 설정
python3 benchmark_ultra_optimization.py

# TensorRT 포함
python3 benchmark_ultra_optimization.py --tensorrt
```

---

## ⚠️ 문제 해결

### 1. sudo 권한 에러

**증상:**
```
⚠️ nvpmodel 실행 실패 (sudo 권한 필요)
```

**해결:**
```bash
./setup_jetson_sudo.sh
```

### 2. TensorRT 변환 실패

**증상:**
```
❌ 엔진 빌드 실패!
```

**해결:**
- Jetson 메모리 부족: 다른 프로그램 종료
- 디스크 공간 확인: `df -h`
- TensorRT 설치 확인: `python3 -c "import tensorrt; print(tensorrt.__version__)"`

### 3. TensorRT 엔진 로드 실패

**증상:**
```
⚠️ TensorRT 엔진 없음 - PyTorch 모드 사용
```

**해결:**
```bash
python3 convert_to_tensorrt.py --precision fp16
```

---

## 📚 추가 최적화 팁

### 1. 입력 해상도 줄이기 (추가 1.5배)

`utils/system_factory.py` 수정:
```python
estimator = create_ultra_depth_estimator(
    preset='balanced',
    input_size=192,  # 256 → 192
    ...
)
```

### 2. 더 작은 모델 사용 (추가 2배)

```bash
python3 convert_to_tensorrt.py --model MiDaS_small --input-size 192
```

`utils/depth_estimation_ultra.py`에서 `preset='fast'` 사용

### 3. 배치 처리 (throughput 향상, latency 증가)

실시간 시스템에는 비추천

---

## 🎯 최종 권장 설정

**최고 성능 (정확도 약간 희생):**
```bash
# 1. INT8 TensorRT
python3 convert_to_tensorrt.py --model MiDaS_small --input-size 192 --precision int8

# 2. 실행
python3 Main_MCP.py
```

**예상 성능**: 40-50 FPS

**균형 (권장):**
```bash
# 1. FP16 TensorRT
python3 convert_to_tensorrt.py --precision fp16

# 2. 실행
python3 Main_MCP.py
```

**예상 성능**: 20-30 FPS

---

## 📞 도움말

문제가 발생하면:
1. Jetson 전력 모드 확인: `nvpmodel -q`
2. GPU 사용률 확인: `tegrastats`
3. 로그 확인: 프로그램 출력 메시지

---

**작성일**: 2025-01-22
**Jetson Orin Nano 최적화 가이드**
