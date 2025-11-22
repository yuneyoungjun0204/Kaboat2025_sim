# 🚀 빠른 시작 가이드 - Jetson 최적화

## 📝 요약

`Main_MCP.py` 실행 시 **자동으로 Jetson 최적화**가 적용됩니다!

---

## ⚡ 즉시 실행 (3단계)

### 1️⃣ sudo 권한 설정 (최초 1회, 5초)

```bash
cd /home/ansl/Real_ka/Kaboat2025_sim
./setup_jetson_sudo.sh
```

### 2️⃣ TensorRT 엔진 생성 (최초 1회, 5-10분)

```bash
# FP16 (권장)
python3 convert_to_tensorrt.py --precision fp16
```

### 3️⃣ 프로그램 실행

```bash
python3 Main_MCP.py
```

**끝!** 이제 자동으로 최적화됩니다! 🎉

---

## 📊 예상 성능

| 설정 | FPS | 비고 |
|-----|-----|------|
| 기본 | 5-8 FPS | - |
| + 자동 최적화 | 10-15 FPS | Main_MCP.py 실행 시 자동 |
| + TensorRT FP16 | **20-30 FPS** | ⭐ 권장 |
| + TensorRT INT8 | **30-40 FPS** | 최고 속도 |

---

## 🔧 자동 적용되는 최적화

`Main_MCP.py` 실행 시:

1. ✅ **Jetson 전력 최대화** (MAXN 모드)
2. ✅ **CUDA 최적화** (cuDNN, TF32)
3. ✅ **PyTorch 최적화** (JIT, 메모리)
4. ✅ **TensorRT 자동 로드** (엔진이 있으면)

---

## 💡 INT8으로 더 빠르게 (선택사항)

더 빠른 성능이 필요하면:

```bash
python3 convert_to_tensorrt.py --precision int8 --engine-path depth_int8.engine
```

그리고 `utils/system_factory.py`에서 경로 변경:
```python
tensorrt_engine_path = "/home/ansl/Real_ka/Kaboat2025_sim/depth_int8.engine"
```

---

## 📖 자세한 가이드

- **전체 최적화 가이드**: `JETSON_OPTIMIZATION_GUIDE.md`
- **TensorRT 변환**: `convert_to_tensorrt.py --help`
- **성능 벤치마크**: `python3 benchmark_ultra_optimization.py`

---

## ✅ 체크리스트

- [ ] sudo 권한 설정 (`./setup_jetson_sudo.sh`)
- [ ] TensorRT 엔진 생성 (`convert_to_tensorrt.py`)
- [ ] 프로그램 실행 (`python3 Main_MCP.py`)
- [ ] 성능 확인 (로그 메시지)

---

**즐거운 개발 되세요! 🚀**
