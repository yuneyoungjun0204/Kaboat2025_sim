#!/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/bin/python3
"""
NanoOWL VRX 실시간 탐지 노드
- 다양한 색상의 부표 및 도형 탐지
- OpenCV 툴바로 실시간 파라미터 및 모드 제어
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
        self.get_logger().info("NanoOWL VRX 실시간 탐지 노드 (툴바 제어 버전) 초기화")
        self.get_logger().info("=" * 80)

        self.bridge = CvBridge()

        # 파라미터 선언 (초기값으로 사용)
        self.declare_parameter('model_name', 'google/owlvit-base-patch32')
        self.declare_parameter('confidence_threshold', 0.1) # 기본 임계값 조정
        self.declare_parameter('min_box_area', 1000)
        self.declare_parameter('max_box_area', 50000) # 최대 크기 상향
        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/image_raw')
        self.declare_parameter('display_window', True)
        self.declare_parameter('initial_mode', 0) # 0: 부표, 1: 도형

        # 파라미터 가져오기 (초기값 설정용)
        model_name = self.get_parameter('model_name').value
        self.initial_threshold = self.get_parameter('confidence_threshold').value
        self.initial_min_area = self.get_parameter('min_box_area').value
        self.initial_max_area = self.get_parameter('max_box_area').value
        camera_topic = self.get_parameter('camera_topic').value
        self.display_window = self.get_parameter('display_window').value
        self.initial_mode = self.get_parameter('initial_mode').value

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.get_logger().info(f"모델 로딩: {model_name}")
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None
        )

        # --- 탐지 대상 분리 ---
        # 0: 부표 탐지용 쿼리
        self.buoy_queries = [
            # 파란 부표 (원통형)
            "a blue buoy", "a floating blue marker", "a blue navigation buoy", "a round blue object in the water",
            # 빨간 고깔 부표
            "a red cone buoy", "a red conical marker", "a cone-shaped red buoy", "a floating red cone",
            # 초록 고깔 부표
            "a green cone buoy", "a green conical marker", "a cone-shaped green buoy", "a floating green cone"
        ]
        self.buoy_label_mapping = {
            0: "blue buoy", 1: "blue buoy", 2: "blue buoy", 3: "blue buoy",
            4: "red cone buoy", 5: "red cone buoy", 6: "red cone buoy", 7: "red cone buoy",
            8: "green cone buoy", 9: "green cone buoy", 10: "green cone buoy", 11: "green cone buoy"
        }

        # 1: 도형 탐지용 쿼리
        self.shape_queries = [
            "a bright red square marker", "a red rectangular shape", "a vivid red square on a surface", "a bold red square marker",
            "a bright blue triangle marker", "a blue triangular shape", "a vivid blue triangle on a surface", "a bold blue triangular marker",
            "a bright yellow circle marker", "a vivid yellow circular shape", "a bold yellow round disc", "a bright yellow filled circle",
        ]
        self.shape_label_mapping = {
            0: "red square", 1: "red square", 2: "red square", 3: "red square",
            4: "blue triangle", 5: "blue triangle", 6: "blue triangle", 7: "blue triangle",
            8: "yellow circle", 9: "yellow circle", 10: "yellow circle", 11: "yellow circle",
        }

        # 텍스트 인코딩 미리 수행
        self.get_logger().info("부표 텍스트 인코딩 중...")
        self.buoy_text_encodings = self.predictor.encode_text(self.buoy_queries)
        self.get_logger().info("도형 텍스트 인코딩 중...")
        self.shape_text_encodings = self.predictor.encode_text(self.shape_queries)

        # --- 실시간 제어용 변수 ---
        self.current_mode = self.initial_mode
        self.current_threshold = self.initial_threshold
        self.current_min_area = self.initial_min_area
        self.current_max_area = self.initial_max_area

        # ROS2 구독자/발행자
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.detections_pub = self.create_publisher(Float32MultiArray, '/nanoowl/detections', 10)
        self.status_pub = self.create_publisher(String, '/nanoowl/status', 10)
        self.viz_image_pub = self.create_publisher(Image, '/nanoowl/visualization', 10)

        # 성능 모니터링
        self.frame_count = 0
        self.start_time = time.time()
        self.total_inference_time = 0

        # 디스플레이 창 및 툴바 초기화
        if self.display_window:
            self._create_trackbars()
            cv2.namedWindow('NanoOWL VRX Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('NanoOWL VRX Detection', 1280, 720)

        self.get_logger().info("✓ 초기화 완료")
        self.get_logger().info("=" * 80)

    def _dummy_callback(self, val):
        """툴바 값 변경 시 호출되는 빈 함수"""
        pass

    def _create_trackbars(self):
        """파라미터 제어를 위한 OpenCV 툴바 생성"""
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Controls', 400, 200)

        cv2.createTrackbar('Mode', 'Controls', self.initial_mode, 1, self._dummy_callback) # 0: Buoy, 1: Shape
        cv2.createTrackbar('Threshold', 'Controls', int(self.initial_threshold * 1000), 100, self._dummy_callback)
        cv2.createTrackbar('Min Area', 'Controls', self.initial_min_area, 50000, self._dummy_callback)
        cv2.createTrackbar('Max Area', 'Controls', self.initial_max_area, 100000, self._dummy_callback)
        self.get_logger().info("✓ 제어 툴바 생성 완료")

    def _update_params_from_trackbars(self):
        """툴바에서 현재 파라미터 값을 읽어와 업데이트"""
        self.current_mode = cv2.getTrackbarPos('Mode', 'Controls')
        self.current_threshold = cv2.getTrackbarPos('Threshold', 'Controls') / 1000.0
        self.current_min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        self.current_max_area = cv2.getTrackbarPos('Max Area', 'Controls')

    def _filter_by_size(self, detections):
        """박스 크기로 필터링"""
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            if self.current_min_area <= area <= self.current_max_area:
                filtered.append(det)
        return filtered

    def _select_best_per_class(self, detections):
        """클래스별로 가장 높은 신뢰도의 객체만 선택"""
        if not detections:
            return []

        best_detections = {}
        for det in detections:
            label = det['label']
            if label not in best_detections or det['confidence'] > best_detections[label]['confidence']:
                best_detections[label] = det
        return list(best_detections.values())


    def detect_frame(self, frame_rgb, mode):
        """프레임에서 객체 탐지 (RGB 입력 및 모드 선택)"""
        image = PILImage.fromarray(frame_rgb.copy())

        # 모드에 따라 사용할 텍스트 인코딩과 라벨 맵 선택
        if mode == 0: # 부표 모드
            text_encodings = self.buoy_text_encodings
            label_mapping = self.buoy_label_mapping
            queries = self.buoy_queries
        else: # 도형 모드
            text_encodings = self.shape_text_encodings
            label_mapping = self.shape_label_mapping
            queries = self.shape_queries
            
        # 탐지 수행
        output = self.predictor.predict(
            image=image,
            text=queries,
            text_encodings=text_encodings,
            threshold=self.current_threshold
        )

        # 결과 파싱
        detections = []
        for i in range(len(output.labels)):
            score = output.scores[i].item()
            bbox = output.boxes[i].detach().cpu().numpy()
            label_idx = output.labels[i].item()
            
            label = label_mapping.get(label_idx, "unknown")
            detections.append({
                "label": label,
                "confidence": score,
                "bbox": [int(b) for b in bbox]
            })

        # 1. 박스 크기 필터링
        detections = self._filter_by_size(detections)

        # 2. 클래스별 최고 신뢰도 선택 (부표, 빨간네모 등 클래스별 1개만 남김)
        detections = self._select_best_per_class(detections)

        return detections

    def image_callback(self, msg):
        """이미지 콜백 함수"""
        self.frame_count += 1
        
        # 툴바가 있으면 값 업데이트
        if self.display_window:
            self._update_params_from_trackbars()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        detections = self.detect_frame(frame_rgb, self.current_mode)
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time

        viz_frame = self._draw(cv_image.copy(), detections, inference_time)

        self._publish_detections(detections)

        status_msg = String()
        mode_str = "Buoy" if self.current_mode == 0 else "Shape"
        status_msg.data = f"Mode: {mode_str} | FPS: {1/inference_time:.1f} | Detections: {len(detections)}"
        self.status_pub.publish(status_msg)

        try:
            viz_msg = self.bridge.cv2_to_imgmsg(viz_frame, "bgr8")
            self.viz_image_pub.publish(viz_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

        if self.display_window:
            cv2.imshow('NanoOWL VRX Detection', viz_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("사용자가 종료를 요청했습니다.")
                self.destroy_node()
                rclpy.shutdown()

        if self.frame_count % 30 == 0:
            avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
            self.get_logger().info(
                f"프레임: {self.frame_count}, 평균 FPS: {avg_fps:.1f}, 탐지: {len(detections)}개"
            )

    def _publish_detections(self, detections):
        """탐지 결과 발행"""
        msg = Float32MultiArray()
        data = [float(len(detections))]

        # 라벨을 숫자로 변환하는 통합 맵
        label_to_id = {
            "blue buoy": 0,
            "red square": 1,
            "blue triangle": 2,
            "yellow circle": 3,
            "red cone buoy": 4,
            "green cone buoy": 5,
        }

        for det in detections:
            label = det['label']
            label_id = label_to_id.get(label, -1) # 모르는 라벨은 -1

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

        colors = {
            "blue": (255, 100, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "buoy": (255, 200, 100) # 부표용 색상
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            color = (255, 255, 255)
            for c_name, c_val in colors.items():
                if c_name in label.lower():
                    color = c_val
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, color, -1)

            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_y = y1 - 10 if y1 - text_h - 15 > 0 else y2 + text_h + 10
            bg_y1 = y1 - text_h - 15 if y1 - text_h - 15 > 0 else y2 + 5
            bg_y2 = y1 - 5 if y1 - text_h - 15 > 0 else y2 + text_h + 15

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, bg_y1), (x1 + text_w + 10, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, text, (x1 + 5, text_y), font, font_scale, color, thickness)

        mode_str = "MODE: BUOY (0)" if self.current_mode == 0 else "MODE: SHAPE (1)"
        info = f"NanoOWL | {inference_time*1000:.0f}ms | {mode_str} | Objects: {len(detections)}"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame

    def destroy_node(self):
        """노드 종료 시 정리"""
        self.get_logger().info("노드를 종료합니다...")
        if self.display_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NanoOWLVRXDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node and rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()