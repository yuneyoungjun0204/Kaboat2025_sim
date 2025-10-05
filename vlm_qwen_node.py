#!/usr/bin/env python3
"""
Qwen2.5-VL-7B 기반 선박 자동 주차 내비게이션 ROS2 노드

토픽:
- 구독: /camera/image_raw (sensor_msgs/Image) - 카메라 이미지
- 발행: /vlm/navigation_command (std_msgs/String) - 항법 명령
- 발행: /vlm/debug_image (sensor_msgs/Image) - 디버그용 이미지 (선택)

명령 형식:
- "Move LEFT/RIGHT/UP/DOWN/FORWARD/BACK"
- "STOP - RED SHAPE CENTERED"
- "No red shape found"
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
from PIL import Image as PILImage
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import numpy as np
from datetime import datetime


class QwenVLMNavigationNode(Node):
    def __init__(self):
        super().__init__('qwen_vlm_navigation')

        # 파라미터 선언
        self.declare_parameter('model_id', 'Qwen/Qwen2.5-VL-7B-Instruct')
        self.declare_parameter('use_quantization', True)
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('processing_rate', 2.0)  # Hz

        # 파라미터 가져오기
        model_id = self.get_parameter('model_id').value
        use_quantization = self.get_parameter('use_quantization').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        processing_rate = self.get_parameter('processing_rate').value

        self.get_logger().info(f'Qwen VLM Navigation Node 초기화 중...')
        self.get_logger().info(f'모델: {model_id}')
        self.get_logger().info(f'양자화: {use_quantization}')

        # VLM 모델 초기화
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f'디바이스: {self.device}')

        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(model_id)

        # 양자화 설정
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

        self.get_logger().info('모델 로딩 완료!')

        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.get_logger().info(f'GPU 메모리: {allocated:.2f}GB (예약: {reserved:.2f}GB)')

        # CV Bridge
        self.bridge = CvBridge()

        # 최신 이미지 저장
        self.latest_image = None
        self.image_lock = False

        # ROS2 구독자/발행자
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            String,
            '/vlm/navigation_command',
            10
        )

        if self.publish_debug:
            self.debug_img_pub = self.create_publisher(
                Image,
                '/vlm/debug_image',
                10
            )

        # 주기적 처리 타이머
        self.timer = self.create_timer(1.0 / processing_rate, self.process_image)

        self.get_logger().info('Qwen VLM Navigation Node 준비 완료!')

    def image_callback(self, msg):
        """카메라 이미지 수신"""
        if not self.image_lock:
            self.latest_image = msg

    def process_image(self):
        """주기적으로 이미지 처리 및 항법 명령 발행"""
        if self.latest_image is None:
            return

        self.image_lock = True

        try:
            # ROS Image → numpy array
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "rgb8")

            # numpy → PIL Image
            pil_image = PILImage.fromarray(cv_image)

            # 임시 파일로 저장 (Qwen-VL은 파일 경로 필요)
            temp_path = f"/tmp/vlm_temp_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
            pil_image.save(temp_path)

            # VLM 분석
            prompt = "Find red shape on dock. If red shape is not at image center, tell boat direction to move: 'Move LEFT/RIGHT/UP/DOWN/FORWARD/BACK'. If centered, say 'STOP - RED SHAPE CENTERED'. Be brief."
            command = self.analyze_image(temp_path, prompt)

            # 명령 발행
            msg = String()
            msg.data = command
            self.cmd_pub.publish(msg)

            self.get_logger().info(f'항법 명령: {command}')

            # 디버그 이미지 발행 (선택)
            if self.publish_debug:
                # 결과를 이미지에 표시
                cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                cv2.putText(cv_image_bgr, command, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                debug_msg = self.bridge.cv2_to_imgmsg(cv_image_bgr, "bgr8")
                self.debug_img_pub.publish(debug_msg)

            # 임시 파일 삭제
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            self.get_logger().error(f'이미지 처리 오류: {str(e)}')
        finally:
            self.image_lock = False

    def analyze_image(self, image_path, prompt):
        """VLM 이미지 분석"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 준비
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
            )

        # 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]


def main(args=None):
    rclpy.init(args=args)
    node = QwenVLMNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
