#!/usr/bin/env python3
"""
장애물 회피 미션
- LOS guidance 기반 경로 추종
- 장애물 감지 시 ONNX 모델 사용
- 장애물 없으면 직접 제어
"""

from typing import Tuple, Callable
import numpy as np
from .base_mission import BaseMission
from .avoid_control import AvoidanceController


class AvoidMission(BaseMission):
    """장애물 회피 미션"""
    
    def __init__(self, waypoints, onnx_control_func: Callable, get_lidar_distance_func: Callable,
                 thrust_scale=800, completion_threshold=15.0):
        """
        Args:
            waypoints: 웨이포인트 리스트
            onnx_control_func: ONNX 모델 제어 함수
            get_lidar_distance_func: LiDAR 거리 조회 함수
            thrust_scale: 스러스터 스케일
            completion_threshold: 목표 도달 판정 거리
        """
        super().__init__("Obstacle Avoidance", waypoints, completion_threshold)
        
        # 장애물 회피 컨트롤러
        self.avoidance_controller = AvoidanceController(
            boat_width=2.2,
            boat_height=50.0,
            max_lidar_distance=100.0,
            los_delta=10.0,
            los_lookahead_min=30.0,
            los_lookahead_max=80.0,
            filter_alpha=0.35
        )
        
        self.onnx_control_func = onnx_control_func
        self.get_lidar_distance_func = get_lidar_distance_func
        self.thrust_scale = thrust_scale
        
        # 제어 상태
        self.use_direct_control = False
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        # 시각화용
        self.check_area_points = []
        self.los_target = None
    
    def update(self, current_pos: np.ndarray, agent_heading: float, 
               lidar_distances: np.ndarray) -> Tuple[float, float]:
        """
        미션 업데이트
        
        Args:
            current_pos: 현재 위치 [y, x]
            agent_heading: 현재 헤딩 (도)
            lidar_distances: LiDAR 거리 데이터
            
        Returns:
            (left_thrust, right_thrust)
        """
        # 미션이 실행 중이 아니거나 목표가 없으면 정지
        if not self.is_running() or self.target_position is None:
            return 0.0, 0.0
        
        # 웨이포인트 도달 확인
        if self.check_waypoint_reached(current_pos):
            return 0.0, 0.0
        
        # LOS target 계산
        self.los_target = self.avoidance_controller.get_los_target(
            current_pos, self.waypoints, self.current_target_index
        )
        
        # 장애물 확인 및 제어 명령 계산
        self.use_direct_control, linear_velocity, angular_velocity, self.check_area_points = \
            self.avoidance_controller.check_obstacles_and_get_control(
                current_pos, self.los_target, agent_heading, lidar_distances,
                self.get_lidar_distance_func, self.onnx_control_func
            )
        
        # 필터 적용
        filtered_linear, filtered_angular = self.avoidance_controller.apply_filters(
            linear_velocity, angular_velocity
        )
        
        # 이전 명령 업데이트
        self.previous_moment_input = filtered_angular
        self.previous_force_input = filtered_linear
        
        # 스러스터 명령 계산
        forward_thrust = filtered_linear * self.thrust_scale
        turn_thrust = filtered_angular * self.thrust_scale
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        
        # 스러스터 제한
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        
        # 필터 적용
        left_thrust, right_thrust = self.avoidance_controller.apply_thrust_filters(
            left_thrust, right_thrust
        )
        
        return left_thrust, right_thrust
    
    def get_control_mode(self) -> str:
        """현재 제어 모드 반환"""
        if self.use_direct_control:
            return "DIRECT_CONTROL"
        else:
            return "ONNX_MODEL"
    
    def get_visualization_data(self):
        """시각화 데이터 반환"""
        return {
            'check_area_points': self.check_area_points,
            'los_target': self.los_target,
            'control_mode': self.get_control_mode(),
            'linear_velocity': self.previous_force_input,
            'angular_velocity': self.previous_moment_input
        }

