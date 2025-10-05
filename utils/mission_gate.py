#!/usr/bin/env python3
"""
부표 사이 통과 미션 (Gate Mission)
- 두 부표 사이를 정확히 통과
- 직선 경로 추종
"""

from typing import Tuple
import numpy as np
from .base_mission import BaseMission


class GateMission(BaseMission):
    """부표 사이 통과 미션"""
    
    def __init__(self, waypoints, thrust_scale=800, completion_threshold=15.0):
        """
        Args:
            waypoints: 웨이포인트 리스트 (게이트 중심점들)
            thrust_scale: 스러스터 스케일
            completion_threshold: 목표 도달 판정 거리
        """
        super().__init__("Gate Passing", waypoints, completion_threshold)
        self.thrust_scale = thrust_scale
        
        # PID 제어 파라미터
        self.kp_heading = 2.0
        self.kp_distance = 0.5
    
    def calculate_heading_error(self, current_pos: np.ndarray, target_pos: np.ndarray,
                               agent_heading: float) -> float:
        """목표 방향과 현재 헤딩의 차이 계산"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_heading_rad = np.arctan2(dx, dy)
        current_heading_rad = np.radians(agent_heading)
        heading_error = target_heading_rad - current_heading_rad
        
        # -π ~ π 범위로 정규화
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        return heading_error
    
    def update(self, current_pos: np.ndarray, agent_heading: float) -> Tuple[float, float]:
        """
        미션 업데이트
        
        Args:
            current_pos: 현재 위치 [y, x]
            agent_heading: 현재 헤딩 (도)
            
        Returns:
            (left_thrust, right_thrust)
        """
        # 미션이 실행 중이 아니거나 목표가 없으면 정지
        if not self.is_running() or self.target_position is None:
            return 0.0, 0.0
        
        # 웨이포인트 도달 확인
        if self.check_waypoint_reached(current_pos):
            return 0.0, 0.0
        
        # 목표까지 거리 계산
        distance = np.sqrt(
            (current_pos[0] - self.target_position[0])**2 + 
            (current_pos[1] - self.target_position[1])**2
        )
        
        # 헤딩 에러 계산
        heading_error = self.calculate_heading_error(current_pos, self.target_position, agent_heading)
        
        # 제어 명령 계산
        # 거리에 따른 속도 조절
        if distance > 30.0:
            forward_speed = 1.0
        elif distance > 15.0:
            forward_speed = 0.7
        else:
            forward_speed = 0.4
        
        # 헤딩 에러에 따른 회전 속도
        angular_speed = self.kp_heading * heading_error
        angular_speed = np.clip(angular_speed, -1.0, 1.0)
        
        # 회전 시 전진 속도 감소
        forward_speed = forward_speed * (1.0 - abs(angular_speed) * 0.3)
        forward_speed = np.clip(forward_speed, 0.1, 1.0)
        
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
        return "GATE_DIRECT"

