#!/usr/bin/env python3
"""
시각화 시스템 모듈
- OpenCV 기반 탐지 결과 및 깊이 맵 시각화
- 트랙바 제어
- 원본 탐지 vs 추적 결과 구분 표시
"""

import cv2
import numpy as np
from typing import List, Dict, Callable, Optional
from .detection_system import MissionType


class VisualizationSystem:
    """통합 시각화 시스템"""

    def __init__(self, detection_threshold=0.026, min_box_area=500, max_box_area=80000,
                 min_depth=0, max_depth=50, thrust_scale=700):
        """
        Args:
            detection_threshold: 탐지 임계값
            min_box_area: 최소 박스 면적
            max_box_area: 최대 박스 면적
            min_depth: 최소 깊이 (미터)
            max_depth: 최대 깊이 (미터)
            thrust_scale: 스러스터 스케일
        """
        # Jetson 최적화: 고정된 파라미터 (트랙바 제거)
        self.detection_threshold = 0.026
        self.min_box_area = 500
        self.max_box_area = 80000
        self.min_depth_threshold = 0.0
        self.max_depth_threshold = 50.0
        self.thrust_scale = 700.0

        # IMM-PDAF 파라미터 (고정)
        self.max_coast_frames = 4
        self.gate_threshold = 9.0  # 원래 90/10

        # 색상 매핑
        self.colors = {
            "red_cone": (0, 0, 255),      # 빨강
            "green_cone": (0, 255, 0),     # 초록
            "blue_buoy": (255, 0, 0)       # 파랑
        }

        # Jetson 최적화: 메인 시각화 창만 생성 (Parameters 창 제거)
        cv2.namedWindow('VRX Mission Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VRX Mission Control', 640, 480)

    # Jetson 최적화: 트랙바 시스템 완전 제거 (고정 파라미터 사용)
    def update_parameters_from_trackbars(self) -> Dict:
        """
        고정 파라미터 반환 (트랙바 제거됨)

        Returns:
            고정된 파라미터 딕셔너리
        """
        return {
            'detection_threshold': self.detection_threshold,
            'min_box_area': self.min_box_area,
            'max_box_area': self.max_box_area,
            'min_depth': self.min_depth_threshold,
            'max_depth': self.max_depth_threshold,
            'thrust_scale': self.thrust_scale,
            'max_coast_frames': self.max_coast_frames,
            'gate_threshold': self.gate_threshold,
            'circle_rotation_dir': 1,  # 시계방향 고정
            'circle_base_speed': 150.0,
            'circle_min_speed': 50.0,
            'circle_max_turn': 150.0,
            'circle_pid_kp': 0.8,
            'circle_tx_base_x': 1040.0,
            'circle_tx_slope': 700.0,
            'circle_tx_min_x': 800.0,
            'circle_tx_max_x': 1200.0,
            'pass_max_depth_diff': 1.0,
            'force_mission_mode': 0,  # 일반 모드 고정
            'force_obstacle_avoid': False
        }

    # visualize_depth_map 메서드 제거 (성능 최적화)
    # 깊이 맵 시각화는 사용되지 않으므로 제거됨

    def visualize_detections(self, image: np.ndarray, detections: List[Dict],
                            mission_name: str, waypoint_index: int, total_waypoints: int,
                            raw_detections: Optional[List[Dict]] = None,
                            bridge=None, viz_image_pub=None):
        """
        탐지 결과 시각화 (Jetson 최적화: 정보 텍스트 제거, 박스만 표시)

        Args:
            image: BGR 이미지
            detections: 추적 결과 리스트 (IMM-PDAF 출력)
            mission_name: 현재 미션 이름
            waypoint_index: 현재 웨이포인트 인덱스
            total_waypoints: 전체 웨이포인트 수
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력)
            bridge: CvBridge 인스턴스 (선택)
            viz_image_pub: 시각화 이미지 퍼블리셔 (선택)
        """
        if image is None:
            return

        # Jetson 최적화: 불필요한 복사 제거, 원본에 직접 그리기
        vis_image = image

        # 추적 결과만 그리기 (굵은 실선) - raw_detections 제거로 성능 향상
        for det in detections:
            # pass_max_depth_diff로 필터링된 객체는 그리지 않음
            if det.get('filtered_by_depth_diff', False):
                continue

            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            cx, cy = det["center"]

            color = self.colors.get(label, (255, 255, 255))

            # 박스와 중심점만 그리기 (텍스트 제거로 성능 향상)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
            cv2.circle(vis_image, (cx, cy), 7, color, -1)

        # 모든 정보 텍스트 제거 (Jetson 최적화)
        # 화면 표시
        cv2.imshow('VRX Mission Control', vis_image)
        cv2.waitKey(1)

    # _draw_dashed_rectangle 및 _draw_dashed_line 메서드 제거 (사용되지 않음, Jetson 최적화)

    def cleanup(self):
        """시각화 창 정리"""
        cv2.destroyAllWindows()
