#!/usr/bin/env python3
"""
NanoOWL + MiDaS 통합 테스트 스크립트
- 카메라 이미지를 받아서 객체 탐지 + 깊이 필터링
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

from utils.depth_estimation import MiDaSHybridDepthEstimator
from utils.detection_system import DetectionSystem, MissionType


class IntegratedDetectionTester(Node):
    """통합 탐지 테스트 노드 (객체 탐지 + 깊이 필터링)"""

    def __init__(self):
        super().__init__('integrated_detection_tester')

        # CvBridge 초기화
        self.bridge = CvBridge()

        # MiDaS 깊이 추정기 초기화
        self.get_logger().info("MiDaS 깊이 추정기 초기화 중...")
        self.depth_estimator = MiDaSHybridDepthEstimator()
        self.get_logger().info("✓ 깊이 추정기 초기화 완료")

        # 탐지 시스템 초기화
        self.get_logger().info("탐지 시스템 초기화 중...")
        self.detection_system = DetectionSystem(
            depth_estimator=self.depth_estimator,
            device="cuda",
            detection_threshold=0.03,
            min_box_area=500,
            max_box_area=80000,
            min_depth=0.0,
            max_depth=50.0
        )
        self.get_logger().info("✓ 탐지 시스템 초기화 완료")

        # 이미지 데이터
        self.current_image = None
        self.depth_map = None
        self.detections = []

        # 파라미터 (트랙바로 조정 가능)
        self.detection_threshold = 3  # 0.03 (0-100, 실제 값은 /100)
        self.min_box_area = 500  # 최소 박스 면적
        self.max_box_area = 80000  # 최대 박스 면적
        self.min_depth = 0  # 최소 깊이 (0-100미터)
        self.max_depth = 50  # 최대 깊이 (0-100미터)
        self.mission_type = 0  # 0: 부표 사이 지나가기, 1: 부표 회전
        self.show_depth = 1  # 0: 깊이 맵 숨김, 1: 깊이 맵 표시

        # 색상 정의
        self.colors = {
            'red_cone': (0, 0, 255),
            'green_cone': (0, 255, 0),
            'blue_buoy': (255, 0, 0)
        }

        # 이미지 구독
        self.image_sub = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera/image_raw',
            self.image_callback,
            10
        )

        # OpenCV 윈도우 생성
        cv2.namedWindow('Integrated Detection Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Integrated Detection Test', 1600, 720)

        # 트랙바 생성
        self._create_trackbars()

        # 타이머로 시각화 업데이트
        self.timer = self.create_timer(0.1, self.update_visualization)

        self.get_logger().info("✓ 테스트 노드 초기화 완료")
        self.get_logger().info("카메라 이미지 대기 중...")

    def _create_trackbars(self):
        """트랙바 생성"""
        cv2.createTrackbar('Mission Type', 'Integrated Detection Test', self.mission_type, 1, self._on_mission_type_change)
        cv2.createTrackbar('Threshold (x0.01)', 'Integrated Detection Test', self.detection_threshold, 100, self._on_threshold_change)
        cv2.createTrackbar('Min Box Area', 'Integrated Detection Test', self.min_box_area, 5000, self._on_min_area_change)
        cv2.createTrackbar('Max Box Area (x100)', 'Integrated Detection Test', int(self.max_box_area / 100), 2000, self._on_max_area_change)
        cv2.createTrackbar('Min Depth (m)', 'Integrated Detection Test', self.min_depth, 100, self._on_min_depth_change)
        cv2.createTrackbar('Max Depth (m)', 'Integrated Detection Test', self.max_depth, 100, self._on_max_depth_change)
        cv2.createTrackbar('Show Depth', 'Integrated Detection Test', self.show_depth, 1, self._on_show_depth_change)

    def _on_mission_type_change(self, value):
        """미션 타입 변경 콜백"""
        self.mission_type = value
        self._update_detection_parameters()

    def _on_threshold_change(self, value):
        """탐지 임계값 변경 콜백"""
        self.detection_threshold = value
        self._update_detection_parameters()

    def _on_min_area_change(self, value):
        """최소 박스 면적 변경 콜백"""
        self.min_box_area = value
        self._update_detection_parameters()

    def _on_max_area_change(self, value):
        """최대 박스 면적 변경 콜백"""
        self.max_box_area = value * 100
        self._update_detection_parameters()

    def _on_min_depth_change(self, value):
        """최소 깊이 변경 콜백"""
        self.min_depth = value
        self._update_detection_parameters()

    def _on_max_depth_change(self, value):
        """최대 깊이 변경 콜백"""
        self.max_depth = value
        self._update_detection_parameters()

    def _on_show_depth_change(self, value):
        """깊이 맵 표시 변경 콜백"""
        self.show_depth = value

    def _update_detection_parameters(self):
        """탐지 시스템 파라미터 업데이트"""
        self.detection_system.update_parameters(
            detection_threshold=self.detection_threshold / 100.0,
            min_box_area=self.min_box_area,
            max_box_area=self.max_box_area,
            min_depth=float(self.min_depth),
            max_depth=float(self.max_depth)
        )

    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image

            # 미션 타입 변환
            mission_type_map = {
                0: MissionType.PASS_BETWEEN_BUOYS,
                1: MissionType.CIRCLE_BUOY
            }
            current_mission = mission_type_map.get(self.mission_type, MissionType.PASS_BETWEEN_BUOYS)

            # 객체 탐지 + 깊이 필터링 수행
            self.detections = self.detection_system.detect_objects(cv_image, current_mission)

            # 깊이 맵 저장 (시각화용)
            self.depth_map = self.depth_estimator.estimate_depth(cv_image)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {e}")

    def update_visualization(self):
        """시각화 업데이트"""
        if self.current_image is None:
            return

        try:
            # 원본 이미지 복사
            display_image = self.current_image.copy()

            # 탐지 결과 그리기
            for det in self.detections:
                label = det['label']
                confidence = det['confidence']
                bbox = det['bbox']
                center = det['center']
                depth = det.get('depth', 0.0)

                # 색상 선택
                color = self.colors.get(label, (255, 255, 255))

                # 바운딩 박스 그리기
                cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # 중심점 그리기
                cv2.circle(display_image, center, 5, color, -1)

                # 레이블, 신뢰도, 깊이 표시
                text = f"{label}: {confidence:.2f}"
                depth_text = f"Depth: {depth:.1f}m"

                cv2.putText(display_image, text, (bbox[0], bbox[1] - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display_image, depth_text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 파라미터 정보 표시
            mission_names = ["Pass Between Buoys", "Circle Buoy"]
            info_text = [
                f"Mission: {mission_names[self.mission_type]}",
                f"Threshold: {self.detection_threshold / 100.0:.2f}",
                f"Box Area: {self.min_box_area} - {self.max_box_area}",
                f"Depth Range: {self.min_depth}m - {self.max_depth}m",
                f"Detections: {len(self.detections)}"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(display_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 30

            # 깊이 맵 표시 여부에 따라 결합
            if self.show_depth and self.depth_map is not None:
                # 깊이 맵을 컬러맵으로 변환
                depth_normalized = (self.depth_map * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                depth_resized = cv2.resize(depth_colored, (display_image.shape[1], display_image.shape[0]))

                # 가로로 결합
                combined = np.hstack([display_image, depth_resized])
            else:
                combined = display_image

            # 표시
            cv2.imshow('Integrated Detection Test', combined)
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
        node = IntegratedDetectionTester()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
