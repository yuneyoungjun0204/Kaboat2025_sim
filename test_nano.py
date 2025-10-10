#!/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/bin/python3
"""
NanoOWL VRX 실시간 탐지 노드
VRX 시뮬레이터의 카메라 이미지를 받아 파란 +모양, 빨간 세모, 초록 원 탐지
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import time
import sys
from PIL import Image as PILImage

# NanoOWL 절대 경로 추가
sys.path.insert(0, '/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/nanoowl')
from nanoowl.owl_predictor import OwlPredictor


class NanoOWLVRXDetector(Node):
    def __init__(self):
        super().__init__('nanoowl_vrx_detector')

        self.get_logger().info("=" * 80)
        self.get_logger().info("NanoOWL VRX 실시간 탐지 노드 초기화")
        self.get_logger().info("=" * 80)

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 파라미터 선언
        self.declare_parameter('model_name', 'google/owlvit-base-patch32')
        self.declare_parameter('confidence_threshold', 0.002)
        self.declare_parameter('nms_iou_threshold', 0.5)
        self.declare_parameter('min_box_area', 1000)
        self.declare_parameter('max_box_area', 5000)
        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/image_raw')
        self.declare_parameter('display_window', True)

        # 파라미터 가져오기
        model_name = self.get_parameter('model_name').value
        self.threshold = self.get_parameter('confidence_threshold').value
        self.nms_iou_threshold = self.get_parameter('nms_iou_threshold').value
        self.min_box_area = self.get_parameter('min_box_area').value
        self.max_box_area = self.get_parameter('max_box_area').value
        camera_topic = self.get_parameter('camera_topic').value
        self.display_window = self.get_parameter('display_window').value

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # NanoOWL Predictor 로드
        self.get_logger().info(f"모델 로딩: {model_name}")
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None  # TensorRT 없이 PyTorch 사용
        )

        # 탐지 대상 쿼리 (각 대상당 4개)
        self.target_queries = [
            # 빨간 네모 (인덱스 0-3)
            "a bright red square marker",
            "a red rectangular shape",
            "a vivid red square on a surface",
            "a bold red square marker",

            # 파란 삼각형 (인덱스 4-7)
            "a bright blue triangle marker",
            "a blue triangular shape",
            "a vivid blue triangle on a surface",
            "a bold blue triangular marker",

            # 노란 원 (인덱스 8-11)
            "a bright yellow circle marker",
            "a vivid yellow circular shape",
            "a bold yellow round disc",
            "a bright yellow filled circle",
        ]

        # 라벨 간소화 매핑 (표시용)
        self.label_mapping = {
            0: "red square", 1: "red square", 2: "red square", 3: "red square",
            4: "blue triangle", 5: "blue triangle", 6: "blue triangle", 7: "blue triangle",
            8: "yellow circle", 9: "yellow circle", 10: "yellow circle", 11: "yellow circle",
        }

        # 텍스트 인코딩 미리 수행 (캐싱)
        self.get_logger().info(f"텍스트 인코딩 중... ({len(self.target_queries)}개 구체적 프롬프트)")
        self.text_encodings = self.predictor.encode_text(self.target_queries)

        self.get_logger().info(f"✓ 탐지 대상 (구체적 프롬프트):")
        self.get_logger().info(f"  - 빨간 네모: 4가지 변형")
        self.get_logger().info(f"  - 파란 세모: 4가지 변형")
        self.get_logger().info(f"  - 노란 원: 4가지 변형")
        self.get_logger().info(f"✓ 박스 크기 제한: {self.min_box_area} ~ {self.max_box_area} 픽셀")
        self.get_logger().info(f"✓ 디바이스: {self.device}")

        # ROS2 구독자/발행자 설정
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # 탐지 결과 발행자
        self.detections_pub = self.create_publisher(
            Float32MultiArray,
            '/nanoowl/detections',
            10
        )

        # 상태 발행자
        self.status_pub = self.create_publisher(
            String,
            '/nanoowl/status',
            10
        )

        # 시각화된 이미지 발행자
        self.viz_image_pub = self.create_publisher(
            Image,
            '/nanoowl/visualization',
            10
        )

        # 성능 모니터링
        self.frame_count = 0
        self.start_time = time.time()
        self.total_inference_time = 0

        # 디스플레이 창 초기화
        if self.display_window:
            cv2.namedWindow('NanoOWL VRX Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('NanoOWL VRX Detection', 1280, 720)

        self.get_logger().info("✓ 초기화 완료")
        self.get_logger().info("=" * 80)

    def _calculate_iou(self, box1, box2):
        """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        # 교집합 면적
        intersection_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

        # 각 박스의 면적
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 합집합 면적
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def _filter_by_size(self, detections):
        """박스 크기로 필터링"""
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            if self.min_box_area <= area <= self.max_box_area:
                filtered.append(det)
        return filtered

    def _select_best_per_color(self, detections):
        """색깔별로 가장 높은 신뢰도의 객체만 선택"""
        if not detections:
            return detections

        # 색깔별로 그룹화
        color_groups = {
            "blue": [],
            "red": [],
            "yellow": []
        }

        for det in detections:
            label = det['label'].lower()
            if "blue" in label:
                color_groups["blue"].append(det)
            elif "red" in label:
                color_groups["red"].append(det)
            elif "yellow" in label:
                color_groups["yellow"].append(det)

        # 각 색깔별로 가장 높은 신뢰도 선택
        best_detections = []
        for color, group in color_groups.items():
            if group:
                # 신뢰도 기준으로 정렬하여 가장 높은 것 선택
                best = max(group, key=lambda x: x['confidence'])
                best_detections.append(best)

        return best_detections

    def detect_frame(self, frame_rgb):
        """프레임에서 객체 탐지 (RGB 입력)"""
        # NumPy -> PIL (copy로 writable 경고 방지)
        image = PILImage.fromarray(frame_rgb.copy())

        # 이미지 전처리
        image_tensor = self.predictor.image_preprocessor.preprocess_pil_image(image)

        # ROI 설정 (전체 이미지)
        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device
        )

        # 이미지 인코딩 (ROI 사용)
        image_encodings = self.predictor.encode_rois(image_tensor, rois, pad_square=False)

        # 탐지 수행 (decode 사용)
        output = self.predictor.decode(
            image_output=image_encodings,
            text_output=self.text_encodings,
            threshold=self.threshold
        )

        # 결과 파싱
        detections = []
        if output.labels is not None and len(output.labels) > 0:
            for i in range(len(output.labels)):
                score = output.scores[i].item()

                if score >= self.threshold:
                    bbox = output.boxes[i].detach().cpu().numpy()
                    label_idx = output.labels[i].item()

                    if label_idx < len(self.target_queries):
                        # 간소화된 라벨 사용 (표시용)
                        label = self.label_mapping.get(label_idx, self.target_queries[label_idx])
                        detections.append({
                            "label": label,
                            "confidence": score,
                            "bbox": [int(b) for b in bbox]
                        })

        # 1. 박스 크기 필터링
        detections = self._filter_by_size(detections)

        # 2. 색깔별 최고 신뢰도 선택
        detections = self._select_best_per_color(detections)

        return detections

    def image_callback(self, msg):
        """이미지 콜백 함수"""
        self.frame_count += 1

        try:
            # ROS Image -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # 탐지 시작
        start_time = time.time()
        detections = self.detect_frame(frame_rgb)
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time

        # 시각화
        viz_frame = self._draw(cv_image.copy(), detections, inference_time)

        # 탐지 결과 발행
        self._publish_detections(detections)

        # 상태 발행
        status_msg = String()
        status_msg.data = f"Frame: {self.frame_count}, FPS: {1/inference_time:.1f}, Detections: {len(detections)}"
        self.status_pub.publish(status_msg)

        # 시각화된 이미지 발행
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(viz_frame, "bgr8")
            self.viz_image_pub.publish(viz_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

        # 디스플레이
        if self.display_window:
            cv2.imshow('NanoOWL VRX Detection', viz_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("사용자가 종료를 요청했습니다.")
                rclpy.shutdown()

        # 성능 모니터링 (30프레임마다)
        if self.frame_count % 30 == 0:
            avg_fps = self.frame_count / self.total_inference_time
            self.get_logger().info(
                f"프레임: {self.frame_count}, 평균 FPS: {avg_fps:.1f}, 탐지: {len(detections)}개"
            )

    def _publish_detections(self, detections):
        """탐지 결과 발행"""
        # Float32MultiArray 형식으로 발행
        # 형식: [num_detections, label_id1, conf1, x1, y1, x2, y2, label_id2, conf2, x2, y2, x2, y2, ...]
        msg = Float32MultiArray()
        data = [float(len(detections))]

        for det in detections:
            label = det['label']
            # 라벨을 숫자로 변환
            if "blue" in label:
                label_id = 0
            elif "red" in label:
                label_id = 1
            elif "yellow" in label:
                label_id = 2
            else:
                label_id = -1

            data.extend([
                float(label_id),
                float(det['confidence']),
                float(det['bbox'][0]),
                float(det['bbox'][1]),
                float(det['bbox'][2]),
                float(det['bbox'][3])
            ])

        msg.data = data
        self.detections_pub.publish(msg)

    def _draw(self, frame, detections, inference_time):
        """탐지 결과 그리기"""
        h, w = frame.shape[:2]

        # 색상 매핑 (BGR 형식)
        colors = {
            "blue": (255, 100, 0),    # 밝은 파란색
            "red": (0, 50, 255),      # 밝은 빨간색
            "yellow": (0, 255, 255)   # 밝은 노란색
        }

        # 탐지 박스 그리기
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            # 색상 선택
            color = (255, 255, 255)  # 기본 흰색
            for c_name, c_val in colors.items():
                if c_name in label.lower():
                    color = c_val
                    break

            # 바운딩 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 코너 강조 (L자 모양)
            corner_len = 15
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

            # 중심점 표시
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 8, color, 2)

            # 라벨 텍스트 (배경 포함)
            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # 텍스트 크기 측정
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # 라벨 위치 결정 (박스 위 또는 아래)
            if y1 - text_h - 10 > 0:
                # 박스 위에 표시
                text_x = x1
                text_y = y1 - 10
                bg_y1 = y1 - text_h - 15
                bg_y2 = y1 - 5
            else:
                # 박스 아래에 표시
                text_x = x1
                text_y = y2 + text_h + 10
                bg_y1 = y2 + 5
                bg_y2 = y2 + text_h + 15

            # 텍스트 배경 (반투명)
            overlay = frame.copy()
            cv2.rectangle(overlay, (text_x, bg_y1), (text_x + text_w + 10, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # 텍스트 그리기
            cv2.putText(frame, text, (text_x + 5, text_y), font, font_scale, color, thickness)

        # 상단 정보 표시 (반투명 배경)
        info = f"NanoOWL VRX | {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f} | Frame: {self.frame_count} | Objects: {len(detections)}"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def destroy_node(self):
        """노드 종료 시 정리"""
        if self.display_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = NanoOWLVRXDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
