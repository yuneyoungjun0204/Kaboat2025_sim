#!/usr/bin/env python3
"""
파라미터 관리 모듈
"""

from typing import Dict, Any, Optional
from utils.detection_system import MissionType
from utils.visualization_system import VisualizationSystem


class ParameterManager:
    """시스템 파라미터 관리 클래스"""

    def __init__(self, visualization_system: VisualizationSystem):
        """
        Args:
            visualization_system: VisualizationSystem 인스턴스
        """
        self.visualization = visualization_system
        self._cached_params = {}

    def update_all_parameters(self) -> Dict[str, Any]:
        """모든 파라미터 업데이트 및 반환"""
        self._cached_params = self.visualization.update_parameters_from_trackbars()
        return self._cached_params

    def get_detection_parameters(self) -> Dict[str, Any]:
        """탐지 시스템용 파라미터 추출"""
        params = self._cached_params
        return {
            'detection_threshold': params.get('detection_threshold'),
            'min_box_area': params.get('min_box_area'),
            'max_box_area': params.get('max_box_area'),
            'min_depth': params.get('min_depth_threshold'),
            'max_depth': params.get('max_depth_threshold'),
        }

    def get_thrust_scale(self) -> Optional[float]:
        """Thrust scale 파라미터 반환"""
        return self._cached_params.get('thrust_scale')

    def get_tracker_parameters(self) -> Dict[str, Any]:
        """트래커용 파라미터 추출"""
        return {
            'max_coast_frames': self._cached_params.get('max_coast_frames'),
            'gate_threshold': self._cached_params.get('gate_threshold'),
        }

    def get_mission_parameters(self, mission_type: MissionType,
                              waypoint_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        미션별 파라미터 추출 및 병합

        Args:
            mission_type: 현재 미션 타입
            waypoint_params: 웨이포인트에 정의된 파라미터

        Returns:
            병합된 미션 파라미터
        """
        trackbar_params = self._cached_params
        waypoint_params = waypoint_params or {}

        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return {
                'pass_max_depth_diff': trackbar_params.get('pass_max_depth_diff', 5.0)
            }

        elif mission_type == MissionType.CIRCLE_BUOY:
            # 선회 미션 파라미터만 추출하여 병합
            circle_params = {k: v for k, v in trackbar_params.items()
                           if k.startswith('circle_')}
            return {**waypoint_params, **circle_params}

        return {}

    def is_force_obstacle_avoid(self) -> bool:
        """강제 장애물 회피 모드 확인"""
        return self._cached_params.get('force_obstacle_avoid', False)
