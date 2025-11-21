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
from .config import Constants


class VisualizationSystem:
    """통합 시각화 시스템"""

    def __init__(self, detection_threshold=None, min_box_area=None, max_box_area=None,
                 min_depth=None, max_depth=None, thrust_scale=None):
        """
        Args:
            모든 파라미터는 config.py에서 가져옵니다 (하위 호환성 유지)
        """
        # Config에서 파라미터 가져오기
        vp = Constants.VisualizationParams

        self.detection_threshold = vp.DETECTION_THRESHOLD
        self.min_box_area = vp.MIN_BOX_AREA
        self.max_box_area = vp.MAX_BOX_AREA
        self.min_depth_threshold = vp.MIN_DEPTH_THRESHOLD
        self.max_depth_threshold = vp.MAX_DEPTH_THRESHOLD
        self.thrust_scale = Constants.DEFAULT_THRUST_SCALE

        # IMM-PDAF 파라미터
        self.max_coast_frames = vp.MAX_COAST_FRAMES
        self.gate_threshold = vp.GATE_THRESHOLD

        # 색상 매핑
        self.colors = {
            "red_cone": vp.COLOR_RED_CONE,
            "green_cone": vp.COLOR_GREEN_CONE,
            "blue_buoy": vp.COLOR_BLUE_BUOY
        }

        # 시각화 창 생성 및 초기화
        cv2.namedWindow(vp.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(vp.WINDOW_NAME, vp.WINDOW_WIDTH, vp.WINDOW_HEIGHT)

        # Depth map 창도 미리 생성
        cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Map", 640, 480)

        # 초기 화면 표시 (검은 화면)
        initial_image = np.zeros((vp.WINDOW_HEIGHT, vp.WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.putText(initial_image, "VRX Mission Control Initialized", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(initial_image, "Waiting for data...", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(vp.WINDOW_NAME, initial_image)
        cv2.waitKey(1)

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
            'min_depth_threshold': self.min_depth_threshold,
            'max_depth_threshold': self.max_depth_threshold,
            'thrust_scale': self.thrust_scale,
            'max_coast_frames': self.max_coast_frames,
            'gate_threshold': self.gate_threshold,
            'circle_rotation_dir': Constants.CIRCLE_DEFAULT_ROTATION_DIR,
            'circle_base_speed': Constants.CIRCLE_BASE_SPEED,
            'circle_min_speed': Constants.CIRCLE_MIN_SPEED,
            'circle_max_turn': Constants.CIRCLE_MAX_TURN_THRUST,
            'circle_pid_kp': Constants.CIRCLE_PID_KP,
            'circle_tx_base_x': Constants.CIRCLE_TX_BASE_X,
            'circle_tx_slope': Constants.CIRCLE_TX_SLOPE,
            'circle_tx_min_x': Constants.CIRCLE_TX_MIN_X,
            'circle_tx_max_x': Constants.CIRCLE_TX_MAX_X,
            'pass_max_depth_diff': Constants.PASS_BETWEEN_MAX_DEPTH_DIFF,
            'rotation_gain': Constants.ROTATION_GAIN,
            'rotation_tolerance': Constants.ROTATION_TOLERANCE,
            'rotation_stable_frames': Constants.ROTATION_STABLE_FRAMES,
            'rotation_max_thrust': Constants.ROTATION_MAX_THRUST,
            'rotation_default_angle': Constants.ROTATION_DEFAULT_TARGET
        }

    def visualize_depth_map(self, depth_map: np.ndarray, window_name: str = "Depth Map"):
        """
        깊이 맵 시각화

        Args:
            depth_map: 깊이 맵 (0-1 정규화된 numpy 배열)
            window_name: 창 이름
        """
        if depth_map is None:
            return

        # 깊이 맵을 컬러맵으로 변환 (더 보기 좋게)
        depth_colormap = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO  # 또는 COLORMAP_JET, COLORMAP_TURBO
        )

        # 화면 표시
        cv2.imshow(window_name, depth_colormap)
        cv2.waitKey(1)

    def visualize_detections(self, image: np.ndarray, detections: List[Dict],
                            mission_name: str, waypoint_index: int, total_waypoints: int,
                            raw_detections: Optional[List[Dict]] = None,
                            bridge=None, viz_image_pub=None,
                            accumulated_angle: Optional[float] = None,
                            depth_map: Optional[np.ndarray] = None):
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
            accumulated_angle: 도킹 미션 누적 각도 (선택)
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

        # 도킹 미션일 경우 누적 각도 표시
        if mission_name == "DOCK_MODE" and accumulated_angle is not None:
            # 화면 상단에 누적 각도 표시
            vp = Constants.VisualizationParams
            text = f"Accumulated Angle: {accumulated_angle:.1f} deg"
            cv2.putText(vis_image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, vp.COLOR_ACCUMULATED_ANGLE, 2)

        # 화면 표시
        cv2.imshow(Constants.VisualizationParams.WINDOW_NAME, vis_image)

        # Depth map 시각화 (선택적)
        if depth_map is not None:
            self.visualize_depth_map(depth_map)

        cv2.waitKey(1)

    # _draw_dashed_rectangle 및 _draw_dashed_line 메서드 제거 (사용되지 않음, Jetson 최적화)

    def cleanup(self):
        """시각화 창 정리"""
        cv2.destroyWindow(Constants.VisualizationParams.WINDOW_NAME)
        cv2.destroyWindow("Depth Map")
        cv2.destroyAllWindows()
