#!/usr/bin/env python3
"""
Vision-to-Action 직접 학습
이미지 → [전진값, 좌우값, 선회값] (범위: -1~1)

데이터셋 형식:
{
    "image": "path/to/image.jpg",
    "action": [0.5, -0.3, 0.2]  # [forward, lateral, rotation]
}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
from tqdm import tqdm


class VisionActionDataset(Dataset):
    """이미지 → 액션 데이터셋"""

    def __init__(self, data_path, transform=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 이미지 로드
        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)

        # 액션 값
        action = torch.tensor(item["action"], dtype=torch.float32)

        return image, action


class VisionActionNetwork(nn.Module):
    """이미지 → 액션 예측 네트워크"""

    def __init__(self, pretrained=True):
        super().__init__()

        # Vision Encoder (ResNet50 사용)
        resnet = models.resnet50(pretrained=pretrained)
        # 마지막 FC layer 제거
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Action Head
        self.action_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),  # [forward, lateral, rotation]
            nn.Tanh()  # -1 ~ 1 범위로 출력
        )

    def forward(self, x):
        # Vision encoding
        features = self.encoder(x)

        # Action prediction
        action = self.action_head(features)

        return action


def create_sample_data():
    """샘플 학습 데이터 생성"""

    # 실제로는 시뮬레이터/실제 환경에서 수집한 데이터 사용
    sample_data = [
        # 빨간 도형이 왼쪽에 있음 → 오른쪽으로 회전
        {"image": "test_img/Screenshot from 2025-10-03 10-15-26.png",
         "action": [0.3, 0.0, 0.5]},  # [전진약간, 좌우0, 우회전]

        # 빨간 도형이 오른쪽에 있음 → 왼쪽으로 회전
        {"image": "test_img/Screenshot from 2025-10-03 10-19-43.png",
         "action": [0.3, 0.0, -0.5]},  # [전진약간, 좌우0, 좌회전]

        # 빨간 도형이 중앙, 멀리 있음 → 전진
        {"image": "test_img/Screenshot from 2025-10-03 10-24-31.png",
         "action": [0.7, 0.0, 0.0]},  # [전진, 좌우0, 선회0]

        # 빨간 도형이 중앙, 가까움 → 정지
        {"image": "test_img/Screenshot from 2025-10-03 10-13-15.png",
         "action": [0.0, 0.0, 0.0]},  # [정지]
    ]

    with open("vision_action_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)

    print("샘플 데이터 생성: vision_action_data.json")
    return sample_data


def train(data_path="vision_action_data.json", epochs=100, batch_size=4):
    """학습 실행"""

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    # 데이터셋
    dataset = VisionActionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델
    model = VisionActionNetwork(pretrained=True).to(device)

    # 옵티마이저 & 손실함수
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 학습
    print(f"\n학습 시작 ({epochs} epochs)...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, actions in dataloader:
            images = images.to(device)
            actions = actions.to(device)

            # Forward
            pred_actions = model(images)
            loss = criterion(pred_actions, actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # 모델 저장
    torch.save(model.state_dict(), "vision_action_model.pt")
    print("\n모델 저장 완료: vision_action_model.pt")

    return model


def test_inference(model_path="vision_action_model.pt"):
    """추론 테스트"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = VisionActionNetwork(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 테스트 이미지
    test_images = [
        "test_img/Screenshot from 2025-10-03 10-15-26.png",
        "test_img/Screenshot from 2025-10-03 10-19-43.png",
        "test_img/Screenshot from 2025-10-03 10-24-31.png",
        "test_img/Screenshot from 2025-10-03 10-13-15.png",
    ]

    print("\n=== 추론 테스트 ===")
    print("액션 형식: [전진(-1~1), 좌우(-1~1), 선회(-1~1)]")
    print("-" * 60)

    with torch.no_grad():
        for img_path in test_images:
            if not os.path.exists(img_path):
                print(f"이미지 없음: {img_path}")
                continue

            # 이미지 로드 & 변환
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # 예측
            action = model(image_tensor).squeeze().cpu().numpy()

            print(f"{os.path.basename(img_path):40s}")
            print(f"  → 액션: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]")
            print(f"     전진: {action[0]:+.3f} | 좌우: {action[1]:+.3f} | 선회: {action[2]:+.3f}")
            print()


def export_to_ros_node(model_path="vision_action_model.pt"):
    """ROS 노드용 추론 코드 생성"""

    code = '''#!/usr/bin/env python3
"""
Vision-to-Action ROS2 노드
이미지 → 직접 모터 값 출력
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage
import numpy as np


class VisionActionNetwork(nn.Module):
    """이미지 → 액션 예측 네트워크"""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.action_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        action = self.action_head(features)
        return action


class VisionActionNode(Node):
    def __init__(self):
        super().__init__('vision_action_node')

        # 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionActionNetwork().to(self.device)
        self.model.load_state_dict(torch.load("vision_action_model.pt", map_location=self.device))
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # CV Bridge
        self.bridge = CvBridge()

        # ROS2 토픽
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Vision-Action Node 준비 완료!')

    def image_callback(self, msg):
        try:
            # ROS Image → PIL Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            pil_image = PILImage.fromarray(cv_image)

            # Transform
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # 예측
            with torch.no_grad():
                action = self.model(image_tensor).squeeze().cpu().numpy()

            # Twist 메시지 생성
            twist = Twist()
            twist.linear.x = float(action[0])   # 전진
            twist.linear.y = float(action[1])   # 좌우 (일부 로봇만 지원)
            twist.angular.z = float(action[2])  # 선회

            # 발행
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f'액션: 전진={action[0]:.2f}, 좌우={action[1]:.2f}, 선회={action[2]:.2f}'
            )

        except Exception as e:
            self.get_logger().error(f'오류: {str(e)}')


def main():
    rclpy.init()
    node = VisionActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''

    with open("vision_action_ros_node.py", 'w') as f:
        f.write(code)

    print("ROS 노드 생성: vision_action_ros_node.py")


def main():
    print("=" * 60)
    print("Vision-to-Action 직접 학습")
    print("이미지 → [전진, 좌우, 선회] 액션 예측")
    print("=" * 60)

    # 1. 샘플 데이터 생성
    create_sample_data()

    # 2. 학습
    train(epochs=100, batch_size=2)

    # 3. 테스트
    test_inference()

    # 4. ROS 노드 생성
    export_to_ros_node()

    print("\n완료! 다음 파일들이 생성되었습니다:")
    print("  - vision_action_model.pt (학습된 모델)")
    print("  - vision_action_ros_node.py (ROS2 노드)")
    print("\nROS 노드 실행:")
    print("  python3 vision_action_ros_node.py")


if __name__ == "__main__":
    main()
