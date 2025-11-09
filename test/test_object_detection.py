#!/usr/bin/env python3
"""
NanoOWL 객체 탐지 테스트 스크립트
- 카메라 이미지를 받아서 NanoOWL로 객체 탐지
- 트랙바로 파라미터 조정
"""

import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage

# NanoOWL 경로 추가
sys.path.insert(0, '/home/yuneyoungjun/nanoowl')
from nanoowl.owl_predictor import OwlPredictor


class ObjectDetectionTester(Node):
    """객체 탐지 테스트 노드"""

    def __init__(self):
        super().__init__('object_detection_tester')

        # CvBridge 초기화
        self.bridge = CvBridge()

        # NanoOWL 초기화
        self.get_logger().info("NanoOWL 초기화 중...")
        self.device = "cuda"
        model_name = 'google/owlvit-base-patch32'
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None
        )
        self.get_logger().info("✓ NanoOWL 초기화 완료")

        # 이미지 데이터
        self.current_image = None
        self.detections = []

        # 파라미터 (트랙바로 조정 가능)
        self.detection_threshold = 3  # 0.03 (0-100, 실제 값은 /100)
        self.min_box_area = 500  # 최소 박스 면적
        self.max_box_area = 80000  # 최대 박스 면적
        self.mission_type = 0  # 0: 부표 사이 지나가기, 1: 부표 회전

        # 탐지 쿼리 정의
        self.detection_queries = {
            0: {  # PASS_BETWEEN_BUOYS
                'queries': [
                    "a red cone buoy", "a red conical marker", "a cone-shaped red buoy",
                    "a green cone buoy", "a green conical marker", "a cone-shaped green buoy"
                ],
                'label_mapping': {
                    0: "red_cone", 1: "red_cone", 2: "red_cone",
                    3: "green_cone", 4: "green_cone", 5: "green_cone"
                },
                'colors': {
                    'red_cone': (0, 0, 255),
                    'green_cone': (0, 255, 0)
                }
            },
            1: {  # CIRCLE_BUOY
                'queries': [
                    "a blue circle"
                ],
                'label_mapping': {
                    0: "blue_buoy"
                },
                'colors': {
                    'blue_buoy': (255, 0, 0)
                }
            }
        }

        # 텍스트 인코딩 미리 수행
        for mission_type, query_info in self.detection_queries.items():
            query_info['text_encodings'] = self.predictor.encode_text(query_info['queries'])

        # 이미지 구독
        self.image_sub = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera/image_raw',
            self.image_callback,
            10
        )

        # OpenCV 윈도우 생성
        cv2.namedWindow('Object Detection Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection Test', 1280, 720)

        # 트랙바 생성
        self._create_trackbars()

        # 타이머로 시각화 업데이트
        self.timer = self.create_timer(0.1, self.update_visualization)

        self.get_logger().info("✓ 테스트 노드 초기화 완료")
        self.get_logger().info("카메라 이미지 대기 중...")

    def _create_trackbars(self):
        """트랙바 생성"""
        cv2.createTrackbar('Mission Type', 'Object Detection Test', self.mission_type, 1, self._on_mission_type_change)
        cv2.createTrackbar('Threshold (x0.01)', 'Object Detection Test', self.detection_threshold, 100, self._on_threshold_change)
        cv2.createTrackbar('Min Box Area', 'Object Detection Test', self.min_box_area, 5000, self._on_min_area_change)
        cv2.createTrackbar('Max Box Area', 'Object Detection Test', int(self.max_box_area / 100), 2000, self._on_max_area_change)

    def _on_mission_type_change(self, value):
        """미션 타입 변경 콜백"""
        self.mission_type = value

    def _on_threshold_change(self, value):
        """탐지 임계값 변경 콜백"""
        self.detection_threshold = value

    def _on_min_area_change(self, value):
        """최소 박스 면적 변경 콜백"""
        self.min_box_area = value

    def _on_max_area_change(self, value):
        """최대 박스 면적 변경 콜백"""
        self.max_box_area = value * 100

    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image

            # 객체 탐지 수행
            self.detections = self._detect_objects(cv_image)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {e}")

    def _detect_objects(self, image):
        """객체 탐지 수행"""
        if image is None:
            return []

        try:
            # RGB로 변환
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = PILImage.fromarray(frame_rgb)

            # 현재 미션에 맞는 쿼리 선택
            query_info = self.detection_queries.get(self.mission_type)
            if query_info is None:
                return []

            # NanoOWL 탐지
            threshold = self.detection_threshold / 100.0
            output = self.predictor.predict(
                image=image_pil,
                text=query_info['queries'],
                text_encodings=query_info['text_encodings'],
                threshold=threshold
            )

            # 결과 파싱
            detections = []
            for i in range(len(output.labels)):
                score = output.scores[i].item()
                bbox = output.boxes[i].detach().cpu().numpy()
                label_idx = output.labels[i].item()

                x1, y1, x2, y2 = [int(b) for b in bbox]

                # 박스 크기 필터링
                area = (x2 - x1) * (y2 - y1)
                if not (self.min_box_area <= area <= self.max_box_area):
                    continue

                # 중심점 계산
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                label = query_info['label_mapping'].get(label_idx, "unknown")
                detections.append({
                    "label": label,
                    "confidence": score,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "area": area
                })

            # 클래스별 최고 신뢰도만 선택
            detections = self._select_best_per_class(detections)

            return detections

        except Exception as e:
            self.get_logger().error(f"탐지 오류: {e}")
            return []

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

    def update_visualization(self):
        """시각화 업데이트"""
        if self.current_image is None:
            return

        try:
            # 원본 이미지 복사
            display_image = self.current_image.copy()

            # 현재 미션 정보
            query_info = self.detection_queries.get(self.mission_type)
            if query_info is None:
                return

            # 탐지 결과 그리기
            for det in self.detections:
                label = det['label']
                confidence = det['confidence']
                bbox = det['bbox']
                center = det['center']
                area = det['area']

                # 색상 선택
                color = query_info['colors'].get(label, (255, 255, 255))

                # 바운딩 박스 그리기
                cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # 중심점 그리기
                cv2.circle(display_image, center, 5, color, -1)

                # 레이블 및 신뢰도 표시
                text = f"{label}: {confidence:.2f} (area: {area})"
                cv2.putText(display_image, text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 파라미터 정보 표시
            mission_names = ["Pass Between Buoys", "Circle Buoy"]
            info_text = [
                f"Mission: {mission_names[self.mission_type]}",
                f"Threshold: {self.detection_threshold / 100.0:.2f}",
                f"Box Area: {self.min_box_area} - {self.max_box_area}",
                f"Detections: {len(self.detections)}"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(display_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 30

            # 표시
            cv2.imshow('Object Detection Test', display_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"시각화 오류: {e}")

    def destroy_node(self):
        """노드 종료"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = ObjectDetectionTester()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
