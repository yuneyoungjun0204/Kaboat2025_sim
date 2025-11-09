#!/usr/bin/env python3
"""
MiDaS 깊이 추정 테스트 스크립트
- 카메라 이미지를 받아서 깊이 맵 생성
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


class DepthEstimationTester(Node):
    """깊이 추정 테스트 노드"""

    def __init__(self):
        super().__init__('depth_estimation_tester')

        # CvBridge 초기화
        self.bridge = CvBridge()

        # MiDaS 깊이 추정기 초기화
        self.get_logger().info("MiDaS 깊이 추정기 초기화 중...")
        self.depth_estimator = MiDaSHybridDepthEstimator()
        self.get_logger().info("✓ 깊이 추정기 초기화 완료")

        # 이미지 데이터
        self.current_image = None
        self.depth_map = None

        # 파라미터 (트랙바로 조정 가능)
        self.depth_colormap = cv2.COLORMAP_INFERNO  # 깊이 맵 컬러맵
        self.blend_alpha = 50  # 원본 이미지와 블렌딩 비율 (0-100)
        self.min_depth_display = 0  # 표시할 최소 깊이 (0-100)
        self.max_depth_display = 100  # 표시할 최대 깊이 (0-100)

        # 이미지 구독
        self.image_sub = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera/image_raw',
            self.image_callback,
            10
        )

        # OpenCV 윈도우 생성
        cv2.namedWindow('Depth Estimation Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth Estimation Test', 1600, 600)

        # 트랙바 생성
        self._create_trackbars()

        # 타이머로 시각화 업데이트
        self.timer = self.create_timer(0.1, self.update_visualization)

        self.get_logger().info("✓ 테스트 노드 초기화 완료")
        self.get_logger().info("카메라 이미지 대기 중...")

    def _create_trackbars(self):
        """트랙바 생성"""
        cv2.createTrackbar('Colormap', 'Depth Estimation Test', 11, 20, self._on_colormap_change)
        cv2.createTrackbar('Blend Alpha (%)', 'Depth Estimation Test', self.blend_alpha, 100, self._on_blend_alpha_change)
        cv2.createTrackbar('Min Depth (%)', 'Depth Estimation Test', self.min_depth_display, 100, self._on_min_depth_change)
        cv2.createTrackbar('Max Depth (%)', 'Depth Estimation Test', self.max_depth_display, 100, self._on_max_depth_change)

    def _on_colormap_change(self, value):
        """컬러맵 변경 콜백"""
        colormap_options = [
            cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET,
            cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
            cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL,
            cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_INFERNO,
            cv2.COLORMAP_MAGMA, cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS,
            cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TWILIGHT_SHIFTED,
            cv2.COLORMAP_TURBO, cv2.COLORMAP_DEEPGREEN
        ]
        if 0 <= value < len(colormap_options):
            self.depth_colormap = colormap_options[value]

    def _on_blend_alpha_change(self, value):
        """블렌딩 알파 변경 콜백"""
        self.blend_alpha = value

    def _on_min_depth_change(self, value):
        """최소 깊이 변경 콜백"""
        self.min_depth_display = value

    def _on_max_depth_change(self, value):
        """최대 깊이 변경 콜백"""
        self.max_depth_display = value

    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image

            # 깊이 맵 추정
            self.depth_map = self.depth_estimator.estimate_depth(cv_image)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {e}")

    def update_visualization(self):
        """시각화 업데이트"""
        if self.current_image is None or self.depth_map is None:
            return

        try:
            # 원본 이미지 복사
            display_image = self.current_image.copy()

            # 깊이 맵을 0-255 범위로 정규화 (사용자 지정 범위 적용)
            depth_normalized = self.depth_map.copy()

            # 깊이 범위 필터링
            min_val = self.min_depth_display / 100.0
            max_val = self.max_depth_display / 100.0

            if max_val > min_val:
                depth_normalized = np.clip(depth_normalized, min_val, max_val)
                depth_normalized = (depth_normalized - min_val) / (max_val - min_val)

            depth_normalized = (depth_normalized * 255).astype(np.uint8)

            # 컬러맵 적용
            depth_colored = cv2.applyColorMap(depth_normalized, self.depth_colormap)

            # 원본 이미지와 블렌딩
            alpha = self.blend_alpha / 100.0
            blended = cv2.addWeighted(display_image, 1 - alpha, depth_colored, alpha, 0)

            # 깊이 맵 단독 표시를 위해 리사이즈
            depth_resized = cv2.resize(depth_colored, (display_image.shape[1], display_image.shape[0]))

            # 텍스트 정보 추가
            info_text = [
                f"Colormap: {self._get_colormap_name()}",
                f"Blend: {self.blend_alpha}%",
                f"Depth Range: {self.min_depth_display}-{self.max_depth_display}%"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(blended, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(blended, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 30

            # 깊이 맵에도 정보 추가
            for i, text in enumerate(info_text):
                cv2.putText(depth_resized, text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(depth_resized, text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 가로로 결합 (원본+블렌딩 | 깊이맵)
            combined = np.hstack([blended, depth_resized])

            # 표시
            cv2.imshow('Depth Estimation Test', combined)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"시각화 오류: {e}")

    def _get_colormap_name(self):
        """현재 컬러맵 이름 반환"""
        colormap_names = [
            "AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN",
            "SUMMER", "SPRING", "COOL", "HSV", "PINK", "INFERNO",
            "MAGMA", "PLASMA", "VIRIDIS", "CIVIDIS", "TWILIGHT",
            "TWILIGHT_SHIFTED", "TURBO", "DEEPGREEN"
        ]
        colormap_values = [
            cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET,
            cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
            cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL,
            cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_INFERNO,
            cv2.COLORMAP_MAGMA, cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS,
            cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TWILIGHT_SHIFTED,
            cv2.COLORMAP_TURBO, cv2.COLORMAP_DEEPGREEN
        ]

        try:
            idx = colormap_values.index(self.depth_colormap)
            return colormap_names[idx]
        except:
            return "UNKNOWN"

    def destroy_node(self):
        """노드 종료"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = DepthEstimationTester()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
