#!/usr/bin/env python3
"""
텍스트 명령 → 모터 제어 변환 학습
간단한 신경망으로 텍스트 명령을 모터 값으로 변환

입력: VLM 텍스트 명령 ("Move RIGHT", "STOP", etc)
출력: [thruster_left, thruster_right, rudder_angle]
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json


class ControlPolicyDataset(Dataset):
    """제어 정책 데이터셋"""

    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 텍스트 명령 토큰화
        text = item["command"]  # "Move RIGHT"
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=16,
            truncation=True,
            return_tensors="pt"
        )

        # 모터 명령 (정규화: -1 ~ 1)
        motor_cmd = torch.tensor(item["motor_values"], dtype=torch.float32)

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "motor_values": motor_cmd
        }


class ControlPolicyNetwork(nn.Module):
    """텍스트 → 모터 명령 변환 네트워크"""

    def __init__(self, vocab_size, hidden_dim=256, output_dim=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256),
            num_layers=2
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # -1 ~ 1 출력
        )

    def forward(self, input_ids, attention_mask):
        # 임베딩
        x = self.embedding(input_ids)

        # Transformer 인코딩
        x = x.permute(1, 0, 2)  # (seq, batch, dim)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        # Pooling (평균)
        x = x.mean(dim=1)

        # 모터 명령 예측
        motor_cmd = self.fc(x)

        return motor_cmd


def create_training_data():
    """예제 학습 데이터 생성"""
    train_data = [
        # 기본 명령
        {"command": "Move LEFT", "motor_values": [-0.5, 0.5, -0.3]},
        {"command": "Move RIGHT", "motor_values": [0.5, -0.5, 0.3]},
        {"command": "Move FORWARD", "motor_values": [0.7, 0.7, 0.0]},
        {"command": "Move BACK", "motor_values": [-0.7, -0.7, 0.0]},
        {"command": "STOP - RED SHAPE CENTERED", "motor_values": [0.0, 0.0, 0.0]},

        # 변형
        {"command": "LEFT", "motor_values": [-0.5, 0.5, -0.3]},
        {"command": "RIGHT", "motor_values": [0.5, -0.5, 0.3]},
        {"command": "FORWARD", "motor_values": [0.7, 0.7, 0.0]},
        {"command": "STOP", "motor_values": [0.0, 0.0, 0.0]},

        # 조합
        {"command": "Move LEFT and FORWARD", "motor_values": [0.3, 0.7, -0.2]},
        {"command": "Move RIGHT and FORWARD", "motor_values": [0.7, 0.3, 0.2]},
    ]

    with open("control_policy_data.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    return train_data


def train():
    # 1. 데이터 준비
    print("학습 데이터 생성 중...")
    create_training_data()

    # 2. 토크나이저
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3. 데이터셋
    dataset = ControlPolicyDataset("control_policy_data.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 모델
    model = ControlPolicyNetwork(vocab_size=tokenizer.vocab_size, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 5. 학습
    print("학습 시작...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(100):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            motor_values = batch["motor_values"].to(device)

            # Forward
            pred_motor = model(input_ids, attention_mask)
            loss = criterion(pred_motor, motor_values)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(dataloader):.4f}")

    # 6. 저장
    torch.save(model.state_dict(), "control_policy.pt")
    print("모델 저장 완료: control_policy.pt")


def test_inference():
    """추론 테스트"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = ControlPolicyNetwork(vocab_size=tokenizer.vocab_size, output_dim=3)
    model.load_state_dict(torch.load("control_policy.pt"))
    model.eval()

    test_commands = [
        "Move LEFT",
        "Move RIGHT",
        "STOP",
        "Move FORWARD"
    ]

    print("\n=== 추론 테스트 ===")
    with torch.no_grad():
        for cmd in test_commands:
            tokens = tokenizer(cmd, return_tensors="pt", padding="max_length", max_length=16)
            motor_values = model(tokens["input_ids"], tokens["attention_mask"])
            print(f"{cmd:20s} → {motor_values.squeeze().tolist()}")


if __name__ == "__main__":
    train()
    test_inference()
