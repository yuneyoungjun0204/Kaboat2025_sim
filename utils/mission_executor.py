#!/usr/bin/env python3
"""
미션 실행 헬퍼 모듈
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from utils.detection_system import MissionType
from utils.mission_strategies_new import MissionManager
from utils.waypoint_manager import WaypointManager


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
                                   logger) -> Tuple[float, float]:
        """부표 사이 지나가기 미션 실행"""
        return self.mission_manager.execute_mission(
            MissionType.PASS_BETWEEN_BUOYS,
            detected_objects=detected_objects,
            current_image=current_image,
            raw_detections=raw_detections,
            mission_params=mission_params,
            logger=logger
        )

    def execute_circle_buoy(self, detected_objects: list, current_image,
                           agent_heading: float, mission_params: Dict[str, Any],
                           raw_detections: list, logger) -> Tuple[float, float]:
        """부표 회전 미션 실행"""
        return self.mission_manager.execute_mission(
            MissionType.CIRCLE_BUOY,
            detected_objects=detected_objects,
            current_image=current_image,
            agent_heading=agent_heading,
            mission_params=mission_params,
            raw_detections=raw_detections,
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
        """웨이포인트 위치 반환 (ONNX 제어용)"""
        waypoints = self.waypoint_manager.waypoints
        current_idx = self.waypoint_manager.get_waypoint_index()

        zeros = np.zeros(2, dtype=np.float32)

        if current_idx < len(waypoints):
            current = waypoints[current_idx]
            current_target = np.array([current['x'], current['y']], dtype=np.float32)
        else:
            current_target = zeros

        if current_idx > 0:
            prev = waypoints[current_idx - 1]
            previous_target = np.array([prev['x'], prev['y']], dtype=np.float32)
        else:
            previous_target = zeros

        if current_idx + 1 < len(waypoints):
            next_wp = waypoints[current_idx + 1]
            next_target = np.array([next_wp['x'], next_wp['y']], dtype=np.float32)
        else:
            next_target = current_target.copy()

        return current_target, previous_target, next_target
