#!/usr/bin/env python3
"""
파라미터 관리 모듈
"""

from typing import Dict, Any, Optional
from utils.detection_system import MissionType
from utils.visualization_system import VisualizationSystem
from utils.config import Constants


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

        elif mission_type == MissionType.ROTATION:
            rotation_params = {
                'desired_angle': trackbar_params.get(
                    'rotation_default_angle', Constants.ROTATION_DEFAULT_TARGET
                ),
                'rotation_gain': trackbar_params.get('rotation_gain', Constants.ROTATION_GAIN),
                'rotation_tolerance': trackbar_params.get('rotation_tolerance', Constants.ROTATION_TOLERANCE),
                'rotation_stable_frames': trackbar_params.get(
                    'rotation_stable_frames', Constants.ROTATION_STABLE_FRAMES
                ),
                'rotation_max_thrust': trackbar_params.get('rotation_max_thrust', Constants.ROTATION_MAX_THRUST)
            }
            rotation_params.update(waypoint_params)
            return rotation_params

        elif mission_type == MissionType.DOCK_MODE:
            # 도킹 미션 파라미터: trackbar 파라미터 + waypoint 파라미터 병합
            dock_params = {k: v for k, v in trackbar_params.items()
                          if k.startswith('dock_')}
            # waypoint_params를 먼저 넣고 trackbar로 덮어쓰기 (trackbar 우선순위)
            # 하지만 target_shape은 waypoint_params 우선
            result = {**dock_params, **waypoint_params}
            return result

        return {}

    def get_force_mission_mode(self) -> int:
        """
        강제 미션 모드 값 반환

        Returns:
            0: 일반 모드 (웨이포인트 기반)
            1: 강제 장애물 회피 모드
            2: 강제 부표 사이 지나기 미션
            3: 강제 부표 한바퀴 돌기 미션
        """
        return self._cached_params.get('force_mission_mode', Constants.ForceMissionMode.NORMAL)

    def is_force_obstacle_avoid(self) -> bool:
        """강제 장애물 회피 모드 확인 (하위 호환성)"""
        return self.get_force_mission_mode() == Constants.ForceMissionMode.OBSTACLE_AVOID

    def get_forced_mission_type(self) -> Optional[MissionType]:
        """
        트랙바로 설정된 강제 미션 타입 반환

        Returns:
            강제 미션 타입 또는 None (일반 모드일 때)
        """
        force_mode = self.get_force_mission_mode()

        if force_mode == Constants.ForceMissionMode.OBSTACLE_AVOID:
            return MissionType.OBSTACLE_AVOID
        elif force_mode == Constants.ForceMissionMode.PASS_BETWEEN_BUOYS:
            return MissionType.PASS_BETWEEN_BUOYS
        elif force_mode == Constants.ForceMissionMode.CIRCLE_BUOY:
            return MissionType.CIRCLE_BUOY
        else:
            return None
