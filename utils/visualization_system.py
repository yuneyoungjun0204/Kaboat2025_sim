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
        # 파라미터
        self.detection_threshold = detection_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_depth_threshold = min_depth
        self.max_depth_threshold = max_depth
        self.thrust_scale = thrust_scale

        # IMM-PDAF 파라미터
        self.max_coast_frames = 4
        self.gate_threshold = 90  # 9.21 * 10

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
        cv2.resizeWindow('VRX Mission Control', 640, 480)

        # # 깊이 맵 시각화 창
        # cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Depth Map', 640, 480)

        # 제어 파라미터 트랙바 창
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parameters', 600, 500)

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

        # IMM-PDAF 파라미터 트랙바
        cv2.createTrackbar('Max Coast Frames', 'Parameters',
                          self.max_coast_frames, 30, self._dummy_callback)
        cv2.createTrackbar('Gate Threshold x10', 'Parameters',
                          self.gate_threshold, 200, self._dummy_callback)

        # 선회 미션 파라미터 트랙바
        cv2.createTrackbar('Circle: Rotation Dir', 'Parameters',
                          1, 2, self._dummy_callback)  # 1=시계방향, 2=반시계방향
        cv2.createTrackbar('Circle: Base Speed', 'Parameters',
                          150, 300, self._dummy_callback)
        cv2.createTrackbar('Circle: Min Speed', 'Parameters',
                          50, 200, self._dummy_callback)
        cv2.createTrackbar('Circle: Max Turn', 'Parameters',
                          150, 250, self._dummy_callback)
        cv2.createTrackbar('Circle: PID Kp x10', 'Parameters',
                          8, 50, self._dummy_callback)  # 0.8-5.0

        # target_x 결정식 파라미터 트랙바 (main_circle.py와 동일)
        cv2.createTrackbar('Circle: TX BaseX', 'Parameters',
                          1040, 2000, self._dummy_callback)
        cv2.createTrackbar('Circle: TX Slope', 'Parameters',
                          700, 10000, self._dummy_callback)
        cv2.createTrackbar('Circle: TX MinX', 'Parameters',
                          800, 2000, self._dummy_callback)
        cv2.createTrackbar('Circle: TX MaxX', 'Parameters',
                          1200, 2000, self._dummy_callback)

        # 부표 사이 통과 미션 파라미터 트랙바
        cv2.createTrackbar('Pass: Max Depth Diff x10', 'Parameters',
                          10, 200, self._dummy_callback)  # 기본값 5.0m (50/10)

        # 강제 미션 모드 트랙바 (0=일반, 1=장애물회피, 2=부표사이지나기, 3=부표한바퀴)
        cv2.createTrackbar('Force Mission Mode', 'Parameters',
                          0, 3, self._dummy_callback)

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

        # IMM-PDAF 파라미터
        self.max_coast_frames = cv2.getTrackbarPos('Max Coast Frames', 'Parameters')
        self.gate_threshold = cv2.getTrackbarPos('Gate Threshold x10', 'Parameters')

        # 선회 미션 파라미터
        circle_rotation_dir = cv2.getTrackbarPos('Circle: Rotation Dir', 'Parameters')
        circle_base_speed = float(cv2.getTrackbarPos('Circle: Base Speed', 'Parameters'))
        circle_min_speed = float(cv2.getTrackbarPos('Circle: Min Speed', 'Parameters'))
        circle_max_turn = float(cv2.getTrackbarPos('Circle: Max Turn', 'Parameters'))
        circle_pid_kp = cv2.getTrackbarPos('Circle: PID Kp x10', 'Parameters') / 10.0

        # target_x 결정식 파라미터 (main_circle.py와 동일)
        circle_tx_base_x = float(cv2.getTrackbarPos('Circle: TX BaseX', 'Parameters'))
        circle_tx_slope = float(cv2.getTrackbarPos('Circle: TX Slope', 'Parameters'))
        circle_tx_min_x = float(cv2.getTrackbarPos('Circle: TX MinX', 'Parameters'))
        circle_tx_max_x = float(cv2.getTrackbarPos('Circle: TX MaxX', 'Parameters'))

        # 미션 모드 선택
        # 부표 사이 통과 미션 파라미터
        pass_max_depth_diff = cv2.getTrackbarPos('Pass: Max Depth Diff x10', 'Parameters') / 10.0

        # 강제 미션 모드 (0=일반, 1=장애물회피, 2=부표사이지나기, 3=부표한바퀴)
        force_mission_mode = cv2.getTrackbarPos('Force Mission Mode', 'Parameters')

        return {
            'detection_threshold': self.detection_threshold,
            'min_box_area': self.min_box_area,
            'max_box_area': self.max_box_area,
            'min_depth': self.min_depth_threshold,
            'max_depth': self.max_depth_threshold,
            'thrust_scale': self.thrust_scale,
            'max_coast_frames': self.max_coast_frames,
            'gate_threshold': self.gate_threshold / 10.0,
            'circle_rotation_dir': circle_rotation_dir,
            'circle_base_speed': circle_base_speed,
            'circle_min_speed': circle_min_speed,
            'circle_max_turn': circle_max_turn,
            'circle_pid_kp': circle_pid_kp,
            'circle_tx_base_x': circle_tx_base_x,
            'circle_tx_slope': circle_tx_slope,
            'circle_tx_min_x': circle_tx_min_x,
            'circle_tx_max_x': circle_tx_max_x,
            'pass_max_depth_diff': pass_max_depth_diff,
            'force_mission_mode': force_mission_mode,
            # 하위 호환성을 위해 유지
            'force_obstacle_avoid': bool(force_mission_mode == 1)
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
