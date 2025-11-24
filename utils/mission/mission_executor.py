#!/usr/bin/env python3
"""
미션 실행 헬퍼 모듈
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from ..detection.detection_system import MissionType
from .mission_strategies_new import MissionManager
from .waypoint_manager import WaypointManager


class MissionExecutor:
    """미션 실행 헬퍼 클래스"""

    def __init__(self, mission_manager: MissionManager, waypoint_manager: WaypointManager):
        """
        Args:
            mission_manager: MissionManager 인스턴스
            waypoint_manager: WaypointManager 인스턴스
        """
        self.mission_manager = mission_manager
        self.waypoint_manager = waypoint_manager

    def execute_pass_between_buoys(self, detected_objects: list, current_image,
                                  raw_detections: list, mission_params: Dict[str, Any],
                                  logger, agent_position: Optional[np.ndarray] = None,
                                  agent_heading: Optional[float] = None) -> Tuple[float, float]:
        """
        부표 사이 지나가기 미션 실행

        Args:
            detected_objects: 탐지된 객체 리스트
            current_image: 현재 이미지
            raw_detections: 원본 탐지 결과
            mission_params: 미션 파라미터
            logger: 로거
            agent_position: 에이전트 현재 위치 (옵션, 부표 미탐지 시 LOS guidance 사용)
            agent_heading: 에이전트 현재 헤딩 (옵션, 부표 미탐지 시 LOS guidance 사용)
        """
        # 현재 웨이포인트 정보 가져오기 (LOS guidance로 추종할 목표)
        current_idx = self.waypoint_manager.get_waypoint_index()
        current_waypoint = None
        previous_waypoint = None

        if agent_position is not None:
            # 현재 목표 웨이포인트
            if current_idx < len(self.waypoint_manager.waypoints):
                curr_wp = self.waypoint_manager.waypoints[current_idx]
                current_waypoint = {'x': curr_wp['x'], 'y': curr_wp['y']}

            # 이전 웨이포인트 (호환성 유지, 미사용)
            if current_idx > 0:
                prev_wp = self.waypoint_manager.waypoints[current_idx - 1]
                previous_waypoint = {'x': prev_wp['x'], 'y': prev_wp['y']}

        return self.mission_manager.execute_mission(
            MissionType.PASS_BETWEEN_BUOYS,
            detected_objects=detected_objects,
            current_image=current_image,
            raw_detections=raw_detections,
            mission_params=mission_params,
            logger=logger,
            agent_position=agent_position,
            agent_heading=agent_heading,
            previous_waypoint=previous_waypoint,
            current_waypoint=current_waypoint
        )

    def execute_circle_buoy(self, agent_position: Optional[np.ndarray],
                            agent_heading: Optional[float],
                            mission_params: Dict[str, Any],
                            logger) -> Tuple[float, float, float, float]:
        """
        웨이포인트 기반 원 궤적 그리기 미션 실행

        Args:
            agent_position: 로봇 현재 위치 [x, y]
            agent_heading: 로봇 헤딩 (도)
            mission_params: 미션 파라미터 (rotation_direction, length, radius_reach, max_duration)
            logger: 로거

        Returns:
            Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos, right_pos)
        """
        return self.mission_manager.execute_mission(
            MissionType.CIRCLE_BUOY,
            agent_position=agent_position,
            agent_heading=agent_heading,
            mission_params=mission_params,
            logger=logger
        )

    def execute_rotation(self, agent_heading: float, mission_params: Dict[str, Any],
                         logger) -> Tuple[float, float]:
        """제자리 선회 미션 실행"""
        return self.mission_manager.execute_mission(
            MissionType.ROTATION,
            agent_heading=agent_heading,
            mission_params=mission_params,
            logger=logger
        )

    def execute_stop(self, mission_params: Dict[str, Any],
                     logger) -> Tuple[float, float]:
        """정지 미션 실행"""
        result = self.mission_manager.execute_mission(
            MissionType.STOP,
            mission_params=mission_params,
            logger=logger
        )
        # StopMission은 body_forces를 반환하므로 thruster 명령으로 변환 필요
        # 하지만 execute_mission이 (left_thrust, right_thrust)를 반환하도록 해야 함
        # 일단 result가 tuple이면 그대로 반환, 아니면 변환
        if isinstance(result, tuple) and len(result) >= 2:
            return result[0], result[1]
        return 0.0, 0.0

    def execute_dock_mission(self, detected_objects: list, current_image,
                             agent_heading: float,
                             agent_position: Optional[np.ndarray],
                             raw_detections: list, mission_params: Dict[str, Any],
                             logger) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
        """
        도킹 미션 실행

        Args:
            detected_objects: 추적된 객체
            current_image: 현재 이미지
            agent_heading: 로봇 헤딩 (도, -180~180)
            agent_position: 로봇 현재 위치 [x, y]
            raw_detections: 원본 탐지 결과
            mission_params: 미션 파라미터
            logger: ROS2 logger

        Returns:
            Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
                (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        return self.mission_manager.execute_mission(
            MissionType.DOCK_MODE,
            detected_objects=detected_objects,
            current_image=current_image,
            agent_heading=agent_heading,
            agent_position=agent_position,
            raw_detections=raw_detections,
            mission_params=mission_params,
            logger=logger
        )

    def execute_obstacle_avoid(self, agent_position: np.ndarray, agent_heading: float,
                               lidar_distances: np.ndarray,
                               get_lidar_distance_func: Callable,
                               get_onnx_control_func: Callable,
                               force_obstacle_avoid: bool,
                               manual_target_x: Optional[float],
                               manual_target_y: Optional[float],
                               logger) -> Tuple[float, float]:
        """
        장애물 회피 미션 실행

        Args:
            agent_position: 에이전트 위치
            agent_heading: 에이전트 방향
            lidar_distances: LiDAR 거리 배열
            get_lidar_distance_func: LiDAR 거리 조회 함수
            get_onnx_control_func: ONNX 제어 함수
            force_obstacle_avoid: 강제 장애물 회피 모드
            manual_target_x: 수동 목표 X 좌표
            manual_target_y: 수동 목표 Y 좌표
            logger: ROS2 logger

        Returns:
            (left_thrust, right_thrust) 튜플
        """
        waypoints = self.waypoint_manager.waypoints
        waypoint_index = self.waypoint_manager.get_waypoint_index()

        # 강제 장애물 회피 모드에서는 수동 목표 사용
        if force_obstacle_avoid and manual_target_x is not None and manual_target_y is not None:
            manual_waypoint = {
                'x': manual_target_x,
                'y': manual_target_y,
                'mission_type': MissionType.OBSTACLE_AVOID,
                'radius': 15.0,
                'params': {}
            }
            waypoints = [manual_waypoint]
            waypoint_index = 0
            logger.info(
                f"🎯 강제 모드: 수동 목표 사용 ({manual_target_x:.1f}, {manual_target_y:.1f})"
            )

        return self.mission_manager.execute_mission(
            MissionType.OBSTACLE_AVOID,
            agent_position=agent_position,
            agent_heading=agent_heading,
            waypoints=waypoints,
            current_waypoint_index=waypoint_index,
            lidar_distances=lidar_distances,
            get_lidar_distance_func=get_lidar_distance_func,
            get_onnx_control_func=get_onnx_control_func,
            logger=logger
        )

    def get_waypoint_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """웨이포인트 위치 반환 (ONNX 제어용) - 미션 시작 위치 기준 상대 좌표"""
        waypoints = self.waypoint_manager.waypoints
        current_idx = self.waypoint_manager.get_waypoint_index()

        zeros = np.zeros(2, dtype=np.float32)

        if current_idx < len(waypoints):
            # 미션 시작 위치 기준 상대 좌표 사용
            x, y = self.waypoint_manager.get_waypoint_relative_to_initial(current_idx)
            current_target = np.array([x, y], dtype=np.float32)
        else:
            current_target = zeros

        if current_idx > 0:
            x, y = self.waypoint_manager.get_waypoint_relative_to_initial(current_idx - 1)
            previous_target = np.array([x, y], dtype=np.float32)
        else:
            previous_target = zeros

        if current_idx + 1 < len(waypoints):
            x, y = self.waypoint_manager.get_waypoint_relative_to_initial(current_idx + 1)
            next_target = np.array([x, y], dtype=np.float32)
        else:
            next_target = current_target.copy()

        return current_target, previous_target, next_target