#!/usr/bin/env python3
"""
미션 베이스 클래스
- 모든 미션의 공통 인터페이스 정의
- 웨이포인트 관리, 미션 상태 관리
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class MissionStatus(Enum):
    """미션 상태"""
    IDLE = "idle"  # 대기
    RUNNING = "running"  # 실행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패


class BaseMission(ABC):
    """미션 베이스 클래스"""
    
    def __init__(self, mission_name: str, waypoints: List[List[float]], 
                 completion_threshold: float = 15.0):
        """
        Args:
            mission_name: 미션 이름
            waypoints: 웨이포인트 리스트 [[y1, x1], [y2, x2], ...]
            completion_threshold: 목표 도달 판정 거리 (미터)
        """
        self.mission_name = mission_name
        self.waypoints = waypoints
        self.completion_threshold = completion_threshold
        self.status = MissionStatus.IDLE
        self.current_target_index = 0
        self.target_position = None
        self.waypoint_reached = False
        
        if len(waypoints) > 0:
            self.target_position = np.array(waypoints[0], dtype=np.float32)
    
    def start(self):
        """미션 시작"""
        self.status = MissionStatus.RUNNING
        self.current_target_index = 0
        if len(self.waypoints) > 0:
            self.target_position = np.array(self.waypoints[0], dtype=np.float32)
        self.waypoint_reached = False
        print(f"✅ [{self.mission_name}] 미션 시작!")
    
    def check_waypoint_reached(self, current_pos: np.ndarray) -> bool:
        """
        현재 웨이포인트 도달 여부 확인
        
        Returns:
            웨이포인트 도달 여부
        """
        if self.target_position is None:
            return False
        
        distance = np.sqrt(
            (current_pos[0] - self.target_position[0])**2 + 
            (current_pos[1] - self.target_position[1])**2
        )
        
        if distance < self.completion_threshold:
            if not self.waypoint_reached:
                self.waypoint_reached = True
                self.current_target_index += 1
                
                if self.current_target_index < len(self.waypoints):
                    # 다음 웨이포인트로 이동
                    next_waypoint = self.waypoints[self.current_target_index]
                    self.target_position = np.array(next_waypoint, dtype=np.float32)
                    self.waypoint_reached = False
                    print(f"🎯 [{self.mission_name}] 웨이포인트 {self.current_target_index}/{len(self.waypoints)} 도달")
                    return False
                else:
                    # 모든 웨이포인트 완료
                    self.target_position = None
                    self.status = MissionStatus.COMPLETED
                    print(f"🎉 [{self.mission_name}] 미션 완료!")
                    return True
        
        return False
    
    def is_completed(self) -> bool:
        """미션 완료 여부"""
        return self.status == MissionStatus.COMPLETED
    
    def is_running(self) -> bool:
        """미션 실행 중 여부"""
        return self.status == MissionStatus.RUNNING
    
    def get_current_target(self) -> Optional[np.ndarray]:
        """현재 목표 위치 반환"""
        return self.target_position
    
    def get_waypoint_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        현재, 이전, 다음 웨이포인트 위치 반환
        
        Returns:
            (current_target, previous_target, next_target)
        """
        if len(self.waypoints) == 0:
            zeros = np.zeros(2, dtype=np.float32)
            return zeros, zeros, zeros
        
        # 현재 목표
        if self.current_target_index < len(self.waypoints):
            current_target = np.array(self.waypoints[self.current_target_index], dtype=np.float32)
        else:
            current_target = np.zeros(2, dtype=np.float32)
        
        # 이전 목표
        if self.current_target_index > 0:
            previous_target = np.array(self.waypoints[self.current_target_index - 1], dtype=np.float32)
        else:
            previous_target = np.zeros(2, dtype=np.float32)
        
        # 다음 목표
        if self.current_target_index + 1 < len(self.waypoints):
            next_target = np.array(self.waypoints[self.current_target_index + 1], dtype=np.float32)
        else:
            next_target = current_target.copy()
        
        return current_target, previous_target, next_target
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Tuple[float, float]:
        """
        미션 업데이트 (각 미션마다 구현)
        
        Returns:
            (left_thrust, right_thrust)
        """
        pass
    
    @abstractmethod
    def get_control_mode(self) -> str:
        """
        현재 제어 모드 반환
        
        Returns:
            제어 모드 문자열
        """
        pass

