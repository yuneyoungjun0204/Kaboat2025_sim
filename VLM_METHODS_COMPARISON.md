# 선박 자동 주차 VLM 방법 비교

## 3가지 접근 방법

### 방법 1: Vision-to-Action 직접 학습 ⭐⭐⭐⭐⭐

**파일**: `train_vision_action.py`

```
이미지 → [전진, 좌우, 선회] 직접 출력
```

**장점**:
- ✅ **매우 빠름** (10-50ms)
- ✅ **가벼움** (100MB)
- ✅ **GPU 1-2GB**
- ✅ **간단한 학습**
- ✅ **End-to-End**

**단점**:
- ❌ 데이터 수집 필요 (100-1000개)
- ❌ 새로운 환경 적응 어려움

**실행**:
```bash
python train_vision_action.py
python3 vision_action_ros_node.py
```

---

### 방법 2: VLM + 제어 로직 ⭐⭐⭐⭐

**파일**: `vlm_qwen_node.py` + `train_control_policy.py`

```
이미지 → [Qwen2.5-VL] → "Move RIGHT" → [제어 로직] → [0.5, -0.5, 0.3]
```

**장점**:
- ✅ **범용성** (다양한 명령 이해)
- ✅ **사전학습 활용** (적은 데이터)
- ✅ **Fine-tuning 가능**

**단점**:
- ❌ **느림** (500ms-2s)
- ❌ **무거움** (4-7GB)
- ❌ **GPU 4GB+**
- ❌ 2단계 처리

**실행**:
```bash
# VLM 노드
python3 vlm_qwen_node.py

# 또는 제어 로직 학습 후
python train_control_policy.py
```

---

### 방법 3: VLM Fine-tuning ⭐⭐⭐

**파일**: `finetune_qwen_vl.py`

```
이미지 → [Fine-tuned Qwen2.5-VL] → 더 정확한 "Move RIGHT"
```

**장점**:
- ✅ **높은 정확도** (도메인 특화)
- ✅ **복잡한 상황 이해**

**단점**:
- ❌ **데이터 많이 필요** (500-1000개)
- ❌ **학습 시간 오래** (30분-3시간)
- ❌ **GPU 4GB+**
- ❌ 여전히 2단계 처리

**실행**:
```bash
python finetune_qwen_vl.py
```

---

## 📊 성능 비교표

| 항목 | Vision-to-Action | VLM + 제어 로직 | VLM Fine-tuning |
|------|-----------------|---------------|----------------|
| **추론 속도** | ⚡ 10-50ms | 🐢 500ms-2s | 🐢 500ms-2s |
| **모델 크기** | 📦 100MB | 📦📦 4-7GB | 📦📦 7GB |
| **GPU 메모리** | 💾 1-2GB | 💾💾 4-8GB | 💾💾 4-8GB |
| **학습 데이터** | 100-1000장 | 10-20 매핑 | 500-1000장 |
| **학습 시간** | ⏱️ 10분-1시간 | ⏱️ 10분 | ⏱️⏱️ 30분-3시간 |
| **정확도** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **범용성** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **구현 난이도** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 상황별 추천

### GPU 5.78GB (현재 환경)

**추천 순위**:
1. 🥇 **Vision-to-Action** - 가볍고 빠름
2. 🥈 **VLM + 제어 로직** (양자화) - 균형잡힌
3. 🥉 **VLM Fine-tuning** (양자화) - 가능하지만 빡빡함

### GPU 없음 (CPU만)

**추천**:
- **VLM (Claude API)** - 외부 API 사용
- 또는 **Vision-to-Action (CPU)** - 느리지만 가능

### 실시간 성능 중요

**추천**:
- 🥇 **Vision-to-Action** - 10-50ms
- 🥉 VLM은 너무 느림

### 정확도 최우선

**추천**:
- 🥇 **VLM Fine-tuning** - 최고 정확도
- 🥈 **VLM + 제어 로직** - 범용성 좋음

---

## 📋 단계별 로드맵

### Phase 1: 프로토타입 (1주일)

```bash
# 1. Vision-to-Action 학습
python train_vision_action.py

# 2. 50-100개 샘플 데이터로 테스트
# 3. 시뮬레이터에서 검증
```

**목표**: 기본 동작 확인

---

### Phase 2: 개선 (2-4주)

**옵션 A: Vision-to-Action 데이터 확장**
```bash
# 500-1000개 데이터 수집
# 재학습
python train_vision_action.py
```

**옵션 B: VLM 도입**
```bash
# Qwen2.5-VL 사용
python3 vlm_qwen_node.py

# 제어 로직 학습
python train_control_policy.py
```

---

### Phase 3: 최적화 (1-2개월)

**옵션 A: Vision-to-Action 고도화**
- Data Augmentation
- 더 큰 모델 (ResNet101)
- Ensemble

**옵션 B: VLM Fine-tuning**
```bash
# 1000개 데이터 수집
python finetune_qwen_vl.py
```

---

## 🚀 빠른 시작 가이드

### 방법 1 선택 (Vision-to-Action)

```bash
# 1. 실행
python train_vision_action.py

# 2. ROS 노드
python3 vision_action_ros_node.py

# 3. 토픽 확인
ros2 topic echo /cmd_vel
```

### 방법 2 선택 (VLM + 제어 로직)

```bash
# 1. VLM 노드 실행
python3 vlm_qwen_node.py

# 2. (선택) 제어 로직 학습
python train_control_policy.py

# 3. 명령 확인
ros2 topic echo /vlm/navigation_command
```

### 방법 3 선택 (VLM Fine-tuning)

```bash
# 1. 데이터 준비 (train_data.json)
# 2. Fine-tuning
python finetune_qwen_vl.py

# 3. Fine-tuned 모델 로드
# (vlm_qwen_node.py 수정 필요)
```

---

## 💡 최종 추천

### 현재 프로젝트 (Kaboat2025_sim)

**단기 (1주일)**:
```bash
# Vision-to-Action 사용
python train_vision_action.py
```
- ✅ 빠른 프로토타입
- ✅ 가벼움
- ✅ 즉시 적용 가능

**장기 (1-2개월)**:
```bash
# VLM Fine-tuning
python finetune_qwen_vl.py
```
- ✅ 최고 정확도
- ✅ 복잡한 상황 대응
- ✅ 프로덕션 레벨

---

## 📚 관련 파일

### Vision-to-Action
- `train_vision_action.py` - 학습 스크립트
- `vision_action_ros_node.py` - ROS 노드 (자동 생성)
- `VISION_ACTION_GUIDE.md` - 상세 가이드

### VLM + 제어 로직
- `vlm_qwen_node.py` - Qwen2.5-VL ROS 노드
- `vlm_phi3_node.py` - Phi-3-Vision ROS 노드
- `train_control_policy.py` - 제어 로직 학습
- `VLM_ROS2_SETUP.md` - ROS 설정 가이드

### VLM Fine-tuning
- `finetune_qwen_vl.py` - Fine-tuning 스크립트
- `FINETUNING_GUIDE.md` - Fine-tuning 가이드

---

## ❓ FAQ

**Q: 어떤 방법이 가장 좋나요?**

A: 상황에 따라 다릅니다:
- 빠른 프로토타입: **Vision-to-Action**
- 높은 정확도: **VLM Fine-tuning**
- 균형: **VLM + 제어 로직**

**Q: GPU가 없으면?**

A: **Claude API** 사용 (유료) 또는 **Vision-to-Action (CPU)**

**Q: 데이터가 없으면?**

A: **VLM + 제어 로직** (10-20개 명령 매핑만 필요)

**Q: 실시간 성능이 중요하면?**

A: **Vision-to-Action** 필수 (10-50ms)

---

**시작 추천**: `train_vision_action.py` 먼저 시도! 🚀
