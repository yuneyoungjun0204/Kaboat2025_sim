# Vision-to-Action 직접 학습 가이드

## 개요

**이미지 입력 → 직접 모터 값 출력**

VLM의 텍스트 단계를 건너뛰고 End-to-End로 학습합니다.

```
카메라 이미지 → [전진값, 좌우값, 선회값]
               ↑
             범위: -1 ~ 1
```

## 장점 vs VLM 방식

| 항목 | Vision-to-Action | VLM (Qwen/Phi-3) |
|------|-----------------|------------------|
| **처리 단계** | 1단계 (직접) | 2단계 (텍스트→변환) |
| **모델 크기** | ~100MB | 4-7GB |
| **GPU 메모리** | 1-2GB | 4-8GB |
| **추론 속도** | **매우 빠름** (10-50ms) | 느림 (500ms-2s) |
| **정확도** | 데이터 품질에 의존 | 높음 (사전학습) |
| **학습 난이도** | **쉬움** | 어려움 |

## 🚀 빠른 시작

### 1단계: 실행

```bash
python train_vision_action.py
```

**출력**:
```
Vision-to-Action 직접 학습
샘플 데이터 생성: vision_action_data.json
학습 시작 (100 epochs)...
Epoch 10/100, Loss: 0.023456
...
모델 저장 완료: vision_action_model.pt

=== 추론 테스트 ===
Screenshot from 2025-10-03 10-15-26.png
  → 액션: [+0.300, +0.000, +0.500]
     전진: +0.300 | 좌우: +0.000 | 선회: +0.500
```

### 2단계: ROS 노드 실행

```bash
python3 vision_action_ros_node.py
```

## 📊 데이터 수집 방법

### 데이터셋 형식 (vision_action_data.json):

```json
[
  {
    "image": "data/img_001.jpg",
    "action": [0.5, 0.0, 0.3]
  },
  {
    "image": "data/img_002.jpg",
    "action": [0.0, 0.0, 0.0]
  }
]
```

**action 값**:
- `[0]`: 전진 (-1=후진, 0=정지, 1=전진)
- `[1]`: 좌우 (-1=왼쪽, 0=중립, 1=오른쪽)
- `[2]`: 선회 (-1=좌회전, 0=직진, 1=우회전)

### 데이터 수집 전략

#### 방법 1: 수동 라벨링 (초기)

```python
# 시뮬레이터에서 이미지 캡처 후 수동으로 라벨링
data = []

# 예시
data.append({
    "image": "captures/dock_left.jpg",
    "action": [0.3, 0.0, 0.5]  # 전진하며 우회전
})

data.append({
    "image": "captures/dock_right.jpg",
    "action": [0.3, 0.0, -0.5]  # 전진하며 좌회전
})

data.append({
    "image": "captures/dock_center_far.jpg",
    "action": [0.7, 0.0, 0.0]  # 전진
})

data.append({
    "image": "captures/dock_center_near.jpg",
    "action": [0.0, 0.0, 0.0]  # 정지
})
```

#### 방법 2: 시뮬레이터 자동 수집

```python
# ROS2에서 자동으로 데이터 수집
class DataCollector(Node):
    def __init__(self):
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.save_image, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.save_action, 10)

        self.data = []

    def save_image(self, msg):
        # 이미지 저장
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        cv2.imwrite(f'data/img_{self.count}.jpg', cv_image)

    def save_action(self, msg):
        # 액션 저장
        self.data.append({
            "image": f'data/img_{self.count}.jpg',
            "action": [
                msg.linear.x,
                msg.linear.y,
                msg.angular.z
            ]
        })
```

#### 방법 3: 전문가 시연 (Imitation Learning)

1. 사람이 조종하며 기록
2. 이미지 + 조이스틱 입력 저장
3. 자동으로 데이터셋 생성

### 필요 데이터 양

- **최소**: 50-100개 (테스트용)
- **권장**: 500-1000개 (실전용)
- **이상적**: 5000개 이상 (프로덕션)

**분포**:
- 다양한 거리 (가까움/중간/멀리)
- 다양한 각도 (왼쪽/중앙/오른쪽)
- 다양한 조명 (낮/밤/흐림)

## 🔧 학습 파라미터 조정

### 기본 설정

```python
train(
    data_path="vision_action_data.json",
    epochs=100,        # 학습 반복 횟수
    batch_size=4,      # 배치 크기
)
```

### 고급 설정

```python
# 모델 크기 조정
class VisionActionNetwork(nn.Module):
    def __init__(self, backbone='resnet50'):
        # resnet18: 가벼움, 빠름
        # resnet50: 균형
        # resnet101: 정확, 느림

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            feature_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            feature_dim = 2048
```

### 학습률 조정

```python
# 빠른 학습 (불안정 가능)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 안정적 학습 (추천)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 느리지만 정확
optimizer = optim.Adam(model.parameters(), lr=1e-5)
```

## 📈 성능 향상 팁

### 1. Data Augmentation

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.ColorJitter(brightness=0.2),  # 밝기 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### 2. 검증 데이터 분리

```python
# 80% 학습, 20% 검증
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
```

### 3. Early Stopping

```python
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

## 🎯 ROS2 통합

### 생성된 ROS 노드 사용

```bash
# 1. 노드 실행
python3 vision_action_ros_node.py

# 2. 다른 터미널에서 카메라 토픽 확인
ros2 topic echo /cmd_vel
```

### 토픽 정보

**구독**:
- `/camera/image_raw` (sensor_msgs/Image)

**발행**:
- `/cmd_vel` (geometry_msgs/Twist)
  - `linear.x`: 전진 (-1~1)
  - `linear.y`: 좌우 (-1~1)
  - `angular.z`: 선회 (-1~1)

### 안전 제한 추가

```python
def limit_action(action, max_speed=0.5):
    """액션 값 제한"""
    action = np.clip(action, -1, 1)  # -1~1 범위
    action = action * max_speed       # 최대 속도 제한
    return action

# 사용
action = model(image)
action = limit_action(action, max_speed=0.5)
```

## 🔍 디버깅

### 과적합 확인

```python
# 학습 데이터와 검증 데이터 Loss 비교
if train_loss < 0.01 and val_loss > 0.1:
    print("과적합 발생! 데이터 증강 또는 Dropout 증가")
```

### 학습 안 됨

```python
# 학습률이 너무 낮거나 높음
# 학습률 스케줄러 사용
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### 예측 값이 극단적

```python
# Tanh 대신 다른 활성화 함수
nn.Sigmoid()  # 0~1 출력 후 스케일 조정
# 또는
# 출력 레이어에 클리핑 추가
```

## 📊 성능 평가

### 평가 메트릭

```python
def evaluate(model, dataloader):
    model.eval()
    total_mae = 0  # Mean Absolute Error

    with torch.no_grad():
        for images, actions in dataloader:
            pred = model(images)
            mae = torch.abs(pred - actions).mean()
            total_mae += mae.item()

    return total_mae / len(dataloader)
```

### 실제 환경 테스트

1. **시뮬레이터 테스트** (먼저)
2. **실제 환경 테스트** (나중)
3. **안전 거리 유지** (항상)

## 🚀 다음 단계

### 단계별 적용

**1주차**: 기본 학습
```bash
# 50-100개 샘플 데이터
python train_vision_action.py
```

**2주차**: 데이터 확장
```bash
# 500-1000개 데이터 수집
# Fine-tuning
```

**3주차**: 실전 배포
```bash
# ROS 노드 통합
python3 vision_action_ros_node.py
```

## 📚 참고

**생성된 파일**:
- `train_vision_action.py` - 학습 스크립트
- `vision_action_model.pt` - 학습된 모델
- `vision_action_ros_node.py` - ROS2 노드
- `vision_action_data.json` - 학습 데이터

**의존성**:
```bash
pip install torch torchvision pillow
```

**모델 비교**:
- Vision-to-Action: 빠름, 가벼움, 데이터 의존적
- VLM (Qwen/Phi-3): 느림, 무거움, 범용적
- OpenVLA: 매우 무거움, 로봇 조작 특화

**추천**: 소규모 프로젝트에는 **Vision-to-Action**이 최적!
