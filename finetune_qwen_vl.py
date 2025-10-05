#!/usr/bin/env python3
"""
Qwen2.5-VL Fine-tuning for 선박 자동 주차 내비게이션
LoRA를 사용한 효율적인 미세조정

데이터셋 형식:
{
    "image": "path/to/image.jpg",
    "instruction": "빨간 도형이 어디 있니? 어떻게 움직여야 해?",
    "output": "Move RIGHT"
}
"""

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from PIL import Image
import json
from qwen_vl_utils import process_vision_info


def prepare_dataset(data_path):
    """데이터셋 준비"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    return dataset


def format_data(example, processor):
    """Qwen2.5-VL 형식으로 변환"""
    # 메시지 구성
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["instruction"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example["output"]},
            ],
        }
    ]

    # 텍스트 준비
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # 이미지 준비
    image_inputs, _ = process_vision_info(messages)

    # 토큰화
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # 라벨 준비 (동일한 input_ids)
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs


def main():
    # 1. 기본 모델 로드
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    print("기본 모델 로딩 중...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. LoRA 설정 (메모리 효율적)
    print("LoRA 설정 중...")
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    print("데이터셋 로딩 중...")
    dataset = prepare_dataset("train_data.json")

    # 데이터셋 변환
    def preprocess(examples):
        return format_data(examples, processor)

    train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./qwen_vl_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # GPU 메모리에 맞게 조정
        gradient_accumulation_steps=8,  # 효과적 배치 크기 16
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,  # 메모리 절약
        gradient_checkpointing=True,  # 메모리 절약
        remove_unused_columns=False,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 6. Fine-tuning 시작
    print("Fine-tuning 시작...")
    trainer.train()

    # 7. 모델 저장
    print("모델 저장 중...")
    model.save_pretrained("./qwen_vl_boat_parking")
    processor.save_pretrained("./qwen_vl_boat_parking")

    print("Fine-tuning 완료!")


if __name__ == "__main__":
    main()
