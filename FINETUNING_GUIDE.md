# 선박 자동 주차 VLM 미세조정 가이드

## 두 가지 미세조정 방법

### 방법 1: Qwen2.5-VL Fine-tuning ⭐ 추천
**목적**: 더 정확한 텍스트 명령 생성

**장점**:
- ✅ VLM이 선박 도메인 이해 향상
- ✅ 복잡한 상황 판단 개선
- ✅ LoRA로 메모리 효율적

**단점**:
- ⏱️ 데이터 수집 시간 필요 (100-1000장)
- 💻 GPU 메모리 4GB+ 필요

---

### 방법 2: 제어 로직 학습
**목적**: 텍스트 명령 → 모터 값 직접 변환

**장점**:
- ✅ 매우 가벼움 (10MB 미만)
- ✅ 빠른 학습 (10분 이내)
- ✅ CPU만으로 가능

**단점**:
- ⚠️ 단순 매핑만 가능
- ⚠️ VLM 명령 품질에 의존

---

## 📋 방법 1: Qwen2.5-VL Fine-tuning

### 1단계: 데이터 수집

```bash
# 시뮬레이터/실제 환경에서 이미지 + 정답 명령 수집
mkdir -p training_data/images
```

#### 데이터셋 형식 (train_data.json):
```json
[
  {
    "image": "training_data/images/dock_001.jpg",
    "instruction": "빨간 도형이 어디 있니? 어떻게 움직여야 해?",
    "output": "Move RIGHT"
  },
  {
    "image": "training_data/images/dock_002.jpg",
    "instruction": "빨간 도형이 어디 있니? 어떻게 움직여야 해?",
    "output": "Move LEFT"
  },
  {
    "image": "training_data/images/dock_003.jpg",
    "instruction": "빨간 도형이 어디 있니? 어떻게 움직여야 해?",
    "output": "STOP - RED SHAPE CENTERED"
  }
]
```

#### 필요 데이터 양:
- **최소**: 100개 (기본 테스트)
- **권장**: 500-1000개 (실전 사용)
- **분포**: 각 명령(LEFT/RIGHT/FORWARD/STOP)당 균등하게

### 2단계: 라이브러리 설치

```bash
pip install peft
pip install datasets
pip install accelerate
```

### 3단계: Fine-tuning 실행

```bash
python finetune_qwen_vl.py
```

**예상 시간**:
- 100개 데이터: ~30분 (GPU)
- 1000개 데이터: ~3시간 (GPU)

**GPU 메모리**:
- LoRA: ~4-6GB
- Full fine-tuning: ~27GB

### 4단계: Fine-tuned 모델 사용

```python
# ROS 노드에서 사용
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Fine-tuned 모델 로드
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)
model = PeftModel.from_pretrained(base_model, "./qwen_vl_boat_parking")
processor = AutoProcessor.from_pretrained("./qwen_vl_boat_parking")

# 추론
command = model.predict(image, "빨간 도형이 어디 있니?")
```

---

## 🎯 방법 2: 제어 로직 학습

### 1단계: 학습 데이터 정의

```python
# control_policy_data.json (이미 자동 생성됨)
[
  {"command": "Move LEFT", "motor_values": [-0.5, 0.5, -0.3]},
  {"command": "Move RIGHT", "motor_values": [0.5, -0.5, 0.3]},
  {"command": "STOP", "motor_values": [0.0, 0.0, 0.0]}
]
```

**motor_values**: [thruster_left, thruster_right, rudder_angle]
- 값 범위: -1.0 ~ 1.0 (정규화)

### 2단계: 학습 실행

```bash
python train_control_policy.py
```

**출력**:
```
Epoch 10/100, Loss: 0.0234
Epoch 20/100, Loss: 0.0089
...
모델 저장 완료: control_policy.pt

=== 추론 테스트 ===
Move LEFT            → [-0.51, 0.48, -0.29]
Move RIGHT           → [0.49, -0.52, 0.31]
STOP                 → [0.01, -0.02, 0.00]
```

### 3단계: ROS 노드 통합

```python
import torch
from train_control_policy import ControlPolicyNetwork
from transformers import AutoTokenizer

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
policy = ControlPolicyNetwork(vocab_size=tokenizer.vocab_size)
policy.load_state_dict(torch.load("control_policy.pt"))
policy.eval()

# VLM 명령 → 모터 값
def command_to_motor(text_command):
    tokens = tokenizer(text_command, return_tensors="pt", padding="max_length", max_length=16)
    with torch.no_grad():
        motor_values = policy(tokens["input_ids"], tokens["attention_mask"])
    return motor_values.squeeze().tolist()

# 사용 예시
vlm_command = "Move RIGHT"  # Qwen2.5-VL 출력
motor_cmd = command_to_motor(vlm_command)
# [0.49, -0.52, 0.31] → 모터 제어
```

---

## 🔄 통합 파이프라인

### 최종 시스템 구조:

```
카메라 이미지
    ↓
[Qwen2.5-VL] (Fine-tuned)
    ↓
텍스트 명령 ("Move RIGHT")
    ↓
[제어 정책 네트워크] (학습됨)
    ↓
모터 값 [0.5, -0.5, 0.3]
    ↓
추진기 제어
```

### 통합 ROS 노드:

```python
class IntegratedVLMNavigationNode(Node):
    def __init__(self):
        # 1. Fine-tuned Qwen2.5-VL
        self.vlm_model = load_finetuned_qwen()

        # 2. 제어 정책 네트워크
        self.control_policy = load_control_policy()

    def process_image(self, image):
        # VLM: 이미지 → 텍스트 명령
        text_cmd = self.vlm_model.predict(image)

        # 제어 정책: 텍스트 → 모터 값
        motor_values = self.control_policy(text_cmd)

        # 모터 명령 발행
        self.publish_motor_command(motor_values)
```

---

## 📊 방법 비교

| 항목 | Qwen2.5-VL Fine-tuning | 제어 로직 학습 |
|------|----------------------|---------------|
| **데이터 수집** | 100-1000 이미지 필요 | 10-20 명령 매핑 |
| **학습 시간** | 30분-3시간 | 10분 |
| **GPU 요구** | 4GB+ | 불필요 |
| **모델 크기** | 7GB | 10MB |
| **정확도 향상** | ⭐⭐⭐⭐⭐ | ⭐⭐☆☆☆ |
| **난이도** | 중 | 쉬움 |

---

## 🎯 추천 전략

### 단계별 적용:

**1단계 (즉시)**:
```bash
# 기본 Qwen2.5-VL + 규칙 기반 제어
python vlm_qwen_node.py
```

**2단계 (1주일 내)**:
```bash
# 제어 로직 학습
python train_control_policy.py
# → 10-20개 명령 매핑 정의
```

**3단계 (1개월 내)**:
```bash
# 데이터 100-1000개 수집
# Qwen2.5-VL fine-tuning
python finetune_qwen_vl.py
```

**4단계 (최종)**:
```bash
# 통합 시스템 배포
python integrated_vlm_node.py
```

---

## ⚠️ 주의사항

### Fine-tuning 시:
1. **과적합 방지**: 학습/검증 데이터 분리 (8:2)
2. **데이터 다양성**: 다양한 조명/각도/거리
3. **균형 잡힌 분포**: 각 명령당 균등한 샘플 수

### 제어 로직 학습 시:
1. **안전 범위**: motor_values [-1, 1] 제한
2. **Fallback**: 알 수 없는 명령 → STOP
3. **검증**: 시뮬레이터에서 먼저 테스트

---

## 📚 추가 자료

**필요 라이브러리**:
```bash
pip install peft datasets accelerate transformers>=4.45.0
```

**생성된 파일**:
- `finetune_qwen_vl.py` - Qwen2.5-VL fine-tuning
- `train_control_policy.py` - 제어 로직 학습
- `control_policy_data.json` - 학습 데이터 (자동 생성)

**참고**:
- Qwen2.5-VL 공식 문서: https://github.com/QwenLM/Qwen2-VL
- LoRA fine-tuning 가이드: https://github.com/huggingface/peft
