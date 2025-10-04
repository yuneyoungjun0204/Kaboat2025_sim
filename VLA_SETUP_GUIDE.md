# SmolVLM 가상환경 설정 가이드

## 1. 가상환경 생성 및 활성화

```bash
# Python 가상환경 생성 (Python 3.8 이상 필요)
python3 -m venv vla_env

# 가상환경 활성화
source vla_env/bin/activate
```

## 2. CUDA 버전 확인 (GPU 사용 시)

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch와 호환되는 CUDA 버전:
# - CUDA 11.8 또는 12.1 권장
```

## 3. 필수 라이브러리 설치

### 옵션 1: requirements.txt 사용 (권장)

```bash
pip install -r requirements.txt
```

### 옵션 2: 개별 설치

```bash
# 기본 라이브러리
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Vision-Language 모델
pip install transformers>=4.45.0
pip install accelerate>=0.20.0
pip install pillow>=9.0.0
pip install sentencepiece>=0.1.99

# 양자화 (속도 2-4배 향상)
pip install bitsandbytes>=0.41.0

# Claude API (선택사항)
pip install anthropic>=0.18.0

# 기타 의존성
pip install opencv-python numpy scipy matplotlib
```

## 4. GPU 메모리별 모델 선택

### GPU 메모리: 1-2GB
```python
# SmolVLM-500M + 4-bit 양자화
python test_smolvlm_500m.py  # 기본값: 양자화 활성화
```
**예상 메모리**: ~300-500MB

### GPU 메모리: 2-4GB
```python
# SmolVLM-2.2B + 4-bit 양자화
python test_smolvlm.py  # 기본값: 양자화 활성화
```
**예상 메모리**: ~1.2-1.5GB

### GPU 메모리: 5GB 이상
```python
# SmolVLM-2.2B (양자화 없음)
# test_smolvlm.py 수정:
analyzer = SmolVLMAnalyzer(use_quantization=False)
```
**예상 메모리**: ~4.9GB

### CPU만 사용 (GPU 없음)
```python
# 양자화 자동 비활성화됨
python test_smolvlm_500m.py  # 500M 모델 권장 (더 빠름)
```
**예상 처리 시간**: 이미지당 10-30초

## 5. 실행 예시

### SmolVLM-500M (가장 빠름, 권장)
```bash
python test_smolvlm_500m.py
```

### SmolVLM-2.2B (더 정확함)
```bash
python test_smolvlm.py
```

### Claude API (최고 정확도)
```bash
export ANTHROPIC_API_KEY="your-api-key"
python test_claude.py
```

## 6. 성능 비교

| 모델 | GPU 메모리 | 속도 (GPU) | 정확도 | 비용 |
|------|-----------|-----------|--------|------|
| SmolVLM-500M + 양자화 | ~300MB | 0.5-2초 | 60-70% | 무료 |
| SmolVLM-2.2B + 양자화 | ~1.5GB | 1-3초 | 70-80% | 무료 |
| SmolVLM-2.2B | ~5GB | 2-5초 | 70-80% | 무료 |
| Claude-3.5-Sonnet | 0MB | 1.5-3초 | 90%+ | 유료 |

## 7. 문제 해결

### bitsandbytes 설치 오류 (Windows)
```bash
# Windows는 bitsandbytes 미지원
# 양자화 비활성화:
analyzer = SmolVLMAnalyzer(use_quantization=False)
```

### CUDA out of memory
```bash
# 더 작은 모델 사용
python test_smolvlm_500m.py

# 또는 양자화 활성화 확인
# 코드에서: use_quantization=True (기본값)
```

### 모델 다운로드 느림
```bash
# Hugging Face 토큰 설정 (선택사항, 빠른 다운로드)
export HF_TOKEN="your-huggingface-token"
```

## 8. 한국어 vs 영어 프롬프트

### 권장사항
- **SmolVLM-500M**: 영어 프롬프트 사용 (기본 설정)
- **SmolVLM-2.2B**: 영어 프롬프트 사용 (한국어 불안정)
- **Claude**: 한국어 완벽 지원

### 프롬프트 변경 방법
```python
# 영어 (권장)
prompt = "Analyze this image in detail. Describe the main objects, colors, positions."

# 한국어 (불안정할 수 있음)
prompt = "이 이미지를 상세히 분석해주세요."
```

## 9. 배치 처리 팁

대량 이미지 처리 시:
1. **SmolVLM-500M + 양자화** 사용 (가장 빠름)
2. GPU 여러 개 있으면 병렬 처리
3. 이미지 해상도 조정 (640x480 권장)

## 10. 가상환경 비활성화

```bash
deactivate
```
