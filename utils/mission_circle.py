#!/usr/bin/env python3
"""
부표 선회 미션 (Circle Mission)
- 특정 부표를 중심으로 원을 그리며 선회
- 일정 반경 유지
"""

from typing import Tuple
import numpy as np
from .base_mission import BaseMission


class CircleMission(BaseMission):
    """부표 선회 미션"""
    
    def __init__(self, waypoints, circle_radius=10.0, circle_direction='clockwise',
                 thrust_scale=800, completion_threshold=15.0):
        """
        Args:
            waypoints: 웨이포인트 리스트 (선회 중심점, 선회 종료점)
            circle_radius: 선회 반경 (미터)
            circle_direction: 선회 방향 ('clockwise' or 'counterclockwise')
            thrust_scale: 스러스터 스케일
            completion_threshold: 목표 도달 판정 거리
        """
        super().__init__("Circle Navigation", waypoints, completion_threshold)
        self.circle_radius = circle_radius
        self.circle_direction = circle_direction
        self.thrust_scale = thrust_scale
        
        # 선회 중심점 (첫 번째 웨이포인트)
        self.circle_center = None
        if len(waypoints) > 0:
            self.circle_center = np.array(waypoints[0], dtype=np.float32)
        
        # 선회 시작 여부
        self.circling_started = False
        self.circling_complete = False
    
    def calculate_tangent_direction(self, current_pos: np.ndarray, center: np.ndarray) -> float:
        """원의 접선 방향 계산"""
        # 중심에서 현재 위치로의 벡터
        to_current = current_pos - center
        distance_to_center = np.linalg.norm(to_current)
        
        if distance_to_center < 0.1:
            return 0.0
        
        # 접선 방향 (반시계방향이 기본)
        tangent_angle = np.arctan2(to_current[0], to_current[1])
        
        if self.circle_direction == 'clockwise':
            tangent_angle -= np.pi / 2  # 시계방향
        else:
            tangent_angle += np.pi / 2  # 반시계방향
        
        return tangent_angle
    
    def calculate_radius_error(self, current_pos: np.ndarray, center: np.ndarray) -> float:
        """현재 위치와 목표 반경의 차이"""
        distance_to_center = np.linalg.norm(current_pos - center)
        return self.circle_radius - distance_to_center
    
    def update(self, current_pos: np.ndarray, agent_heading: float) -> Tuple[float, float]:
        """
        미션 업데이트
        
        Args:
            current_pos: 현재 위치 [y, x]
            agent_heading: 현재 헤딩 (도)
            
        Returns:
            (left_thrust, right_thrust)
        """
        # 미션이 실행 중이 아니거나 중심점이 없으면 정지
        if not self.is_running() or self.circle_center is None:
            return 0.0, 0.0
        
        # 선회가 완료되었는지 확인 (마지막 웨이포인트 도달)
        if self.current_target_index >= len(self.waypoints) - 1:
            if self.check_waypoint_reached(current_pos):
                return 0.0, 0.0
        
        # 선회 중심점까지의 거리
        distance_to_center = np.linalg.norm(current_pos - self.circle_center)
        
        # 목표 반경에 도달했으면 선회 시작
        if not self.circling_started:
            if abs(distance_to_center - self.circle_radius) < 5.0:
                self.circling_started = True
                print(f"🔄 [{self.mission_name}] 선회 시작!")
        
        # 선회 중
        if self.circling_started:
            # 접선 방향 계산
            tangent_angle = self.calculate_tangent_direction(current_pos, self.circle_center)
            
            # 현재 헤딩과의 차이
            current_heading_rad = np.radians(agent_heading)
            heading_error = tangent_angle - current_heading_rad
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # 반경 에러 계산 (중심으로부터 멀어지면 안쪽으로 조정)
            radius_error = self.calculate_radius_error(current_pos, self.circle_center)
            
            # 제어 명령 계산
            angular_speed = 2.0 * heading_error + 0.5 * radius_error / self.circle_radius
            angular_speed = np.clip(angular_speed, -1.0, 1.0)
            
            forward_speed = 0.6  # 선회 시 일정 속도 유지
            forward_speed = forward_speed * (1.0 - abs(angular_speed) * 0.2)
            forward_speed = np.clip(forward_speed, 0.3, 0.8)
        else:
            # 목표 반경으로 접근 중
            dx = self.circle_center[0] - current_pos[0]
            dy = self.circle_center[1] - current_pos[1]
            target_heading_rad = np.arctan2(dx, dy)
            current_heading_rad = np.radians(agent_heading)
            heading_error = target_heading_rad - current_heading_rad
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            angular_speed = 2.0 * heading_error
            angular_speed = np.clip(angular_speed, -1.0, 1.0)
            
            # 거리에 따른 속도 조절
            approach_distance = distance_to_center - self.circle_radius
            forward_speed = 0.5 * np.tanh(approach_distance / 10.0)
            forward_speed = np.clip(forward_speed, 0.1, 0.7)
        
        # 스러스터 명령 계산
        forward_thrust = forward_speed * self.thrust_scale
        turn_thrust = angular_speed * self.thrust_scale
        
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        
        # 스러스터 제한
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        
        return left_thrust, right_thrust
    
    def get_control_mode(self) -> str:
        """현재 제어 모드 반환"""
        if self.circling_started:
            return "CIRCLE_NAVIGATION"
        else:
            return "CIRCLE_APPROACH"

