#!/usr/bin/env python3
"""
시각화 시스템 모듈
- OpenCV 기반 탐지 결과 및 깊이 맵 시각화
- 트랙바 제어
"""

import cv2
import numpy as np
from typing import List, Dict, Callable
from .detection_system import MissionType


class VisualizationSystem:
    """통합 시각화 시스템"""

    def __init__(self, detection_threshold=0.03, min_box_area=500, max_box_area=80000,
                 min_depth=0, max_depth=50, thrust_scale=1000):
        """
        Args:
            detection_threshold: 탐지 임계값
            min_box_area: 최소 박스 면적
            max_box_area: 최대 박스 면적
            min_depth: 최소 깊이 (미터)
            max_depth: 최대 깊이 (미터)
            thrust_scale: 스러스터 스케일
        """
        # 파라미터
        self.detection_threshold = detection_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_depth_threshold = min_depth
        self.max_depth_threshold = max_depth
        self.thrust_scale = thrust_scale

        # 색상 매핑
        self.colors = {
            "red_cone": (0, 0, 255),      # 빨강
            "green_cone": (0, 255, 0),     # 초록
            "blue_buoy": (255, 0, 0)       # 파랑
        }

        # 시각화 창 설정
        self._setup_windows()

    def _setup_windows(self):
        """시각화 창 및 트랙바 설정"""
        # 메인 시각화 창
        cv2.namedWindow('VRX Mission Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VRX Mission Control', 1280, 720)

        # 깊이 맵 시각화 창
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth Map', 640, 480)

        # 제어 파라미터 트랙바 창
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parameters', 600, 400)

        # 탐지 파라미터 트랙바
        cv2.createTrackbar('Detect Threshold', 'Parameters',
                          int(self.detection_threshold * 1000), 1000, self._dummy_callback)
        cv2.createTrackbar('Min Box Area', 'Parameters',
                          self.min_box_area, 50000, self._dummy_callback)
        cv2.createTrackbar('Max Box Area', 'Parameters',
                          self.max_box_area, 150000, self._dummy_callback)
        cv2.createTrackbar('Min Depth (m)', 'Parameters',
                          int(self.min_depth_threshold), 100, self._dummy_callback)
        cv2.createTrackbar('Max Depth (m)', 'Parameters',
                          int(self.max_depth_threshold), 200, self._dummy_callback)

        # 제어 파라미터 트랙바
        cv2.createTrackbar('Thrust Scale', 'Parameters',
                          int(self.thrust_scale), 3000, self._dummy_callback)
        cv2.createTrackbar('Forward Speed x100', 'Parameters',
                          50, 100, self._dummy_callback)
        cv2.createTrackbar('Steering Gain x1000', 'Parameters',
                          3, 20, self._dummy_callback)

    def _dummy_callback(self, val):
        """트랙바 콜백 (빈 함수)"""
        pass

    def update_parameters_from_trackbars(self) -> Dict:
        """
        트랙바에서 파라미터 업데이트

        Returns:
            업데이트된 파라미터 딕셔너리
        """
        self.detection_threshold = cv2.getTrackbarPos('Detect Threshold', 'Parameters') / 1000.0
        self.min_box_area = cv2.getTrackbarPos('Min Box Area', 'Parameters')
        self.max_box_area = cv2.getTrackbarPos('Max Box Area', 'Parameters')
        self.min_depth_threshold = float(cv2.getTrackbarPos('Min Depth (m)', 'Parameters')) / 20.0
        self.max_depth_threshold = float(cv2.getTrackbarPos('Max Depth (m)', 'Parameters'))
        self.thrust_scale = float(cv2.getTrackbarPos('Thrust Scale', 'Parameters'))

        return {
            'detection_threshold': self.detection_threshold,
            'min_box_area': self.min_box_area,
            'max_box_area': self.max_box_area,
            'min_depth': self.min_depth_threshold,
            'max_depth': self.max_depth_threshold,
            'thrust_scale': self.thrust_scale
        }

    def visualize_depth_map(self, depth_map: np.ndarray, mission_name: str):
        """깊이 맵 시각화"""
        # 깊이 맵을 컬러맵으로 변환
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # 정보 표시
        info_text = [
            f"Mission: {mission_name}",
            f"Max Depth: {self.max_depth_threshold:.1f}m",
            f"Min Depth: {self.min_depth_threshold:.1f}m"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(depth_colored, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

        cv2.imshow('Depth Map', depth_colored)

    def visualize_detections(self, image: np.ndarray, detections: List[Dict],
                            mission_name: str, waypoint_index: int, total_waypoints: int,
                            bridge=None, viz_image_pub=None):
        """
        탐지 결과 시각화

        Args:
            image: BGR 이미지
            detections: 탐지 결과 리스트
            mission_name: 현재 미션 이름
            waypoint_index: 현재 웨이포인트 인덱스
            total_waypoints: 전체 웨이포인트 수
            bridge: CvBridge 인스턴스 (선택)
            viz_image_pub: 시각화 이미지 퍼블리셔 (선택)
        """
        if image is None:
            return

        vis_image = image.copy()

        # 탐지 결과 그리기
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            depth = det["depth"]
            cx, cy = det["center"]

            color = self.colors.get(label, (255, 255, 255))

            # 바운딩 박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

            # 중심점
            cv2.circle(vis_image, (cx, cy), 5, color, -1)

            # 라벨 및 정보
            text = f"{label}: {conf:.2f} | {depth:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # 배경 박스
            cv2.rectangle(vis_image, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1 - 5),
                         (0, 0, 0), -1)
            cv2.putText(vis_image, text, (x1 + 5, y1 - 10),
                       font, font_scale, color, thickness)

        # 미션 정보 표시
        mission_info = [
            f"Mission: {mission_name}",
            f"Waypoint: {waypoint_index + 1}/{total_waypoints}",
            f"Detections: {len(detections)}",
            f"Threshold: {self.detection_threshold:.3f}",
            f"Box Area: {self.min_box_area}-{self.max_box_area}",
            f"Max Depth: {self.max_depth_threshold:.1f}m"
        ]

        # 반투명 배경
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0, vis_image)

        # 정보 텍스트
        y_offset = 25
        for text in mission_info:
            cv2.putText(vis_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

        # 화면 표시
        cv2.imshow('VRX Mission Control', vis_image)
        cv2.waitKey(1)

        # ROS 메시지로 발행 (선택)
        if bridge and viz_image_pub:
            try:
                viz_msg = bridge.cv2_to_imgmsg(vis_image, "bgr8")
                viz_image_pub.publish(viz_msg)
            except Exception as e:
                pass

    def cleanup(self):
        """시각화 창 정리"""
        cv2.destroyAllWindows()
