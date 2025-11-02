#!/usr/bin/env python3
"""
웨이포인트 관리 모듈
- 웨이포인트 추가/관리
- 미션 전환 로직
"""

import numpy as np
from typing import List, Dict, Optional
from .detection_system import MissionType
from .config import Constants


class WaypointManager:
    """웨이포인트 관리 시스템"""

    def __init__(self):
        self.waypoints = []
        self.current_waypoint_index = 0

    def add_waypoint(self, x: float, y: float, mission_type: MissionType,
                    radius: float = None, params: Optional[Dict] = None):
        """
        웨이포인트 추가

        Args:
            x: X 좌표
            y: Y 좌표
            mission_type: 미션 타입
            radius: 도달 판정 반경 (미터), None이면 기본값 사용
            params: 미션별 파라미터
        """
        waypoint = {
            'x': x,
            'y': y,
            'mission_type': mission_type,
            'radius': radius if radius is not None else Constants.DEFAULT_WAYPOINT_RADIUS,
            'params': params if params else {}
        }
        self.waypoints.append(waypoint)

    def setup_predefined_waypoints(self):
        """미리 정의된 웨이포인트 설정 (config에서 가져오기)"""
        # mission_type 문자열을 MissionType enum으로 변환하는 매핑
        mission_type_map = {
            'PASS_BETWEEN_BUOYS': MissionType.PASS_BETWEEN_BUOYS,
            'CIRCLE_BUOY': MissionType.CIRCLE_BUOY,
            'WAYPOINT_FOLLOW': MissionType.WAYPOINT_FOLLOW,
            'OBSTACLE_AVOID': MissionType.OBSTACLE_AVOID
        }

        for x, y, mission_type_str, radius, params in Constants.PREDEFINED_WAYPOINTS:
            mission_type = mission_type_map.get(mission_type_str)
            if mission_type:
                self.add_waypoint(x, y, mission_type, radius, params)

    def check_waypoint_reached(self, agent_position: np.ndarray) -> Optional[Dict]:
        """
        웨이포인트 도달 확인

        Args:
            agent_position: 현재 위치

        Returns:
            도달한 경우 다음 웨이포인트 정보, 아니면 None
        """
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return None

        current_waypoint = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([current_waypoint['x'], current_waypoint['y']], dtype=np.float32)

        # 현재 위치에서 웨이포인트까지 거리 계산
        distance = np.linalg.norm(agent_position - target_pos)

        # 웨이포인트 도달 확인
        if distance < current_waypoint['radius']:
            # 다음 웨이포인트로 이동
            self.current_waypoint_index += 1

            if self.current_waypoint_index < len(self.waypoints):
                return self.waypoints[self.current_waypoint_index]
            else:
                return {'completed': True}  # 모든 웨이포인트 완료

        return None

    def get_current_waypoint(self) -> Optional[Dict]:
        """현재 웨이포인트 반환"""
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None

    def get_current_mission_type(self) -> Optional[MissionType]:
        """현재 미션 타입 반환"""
        waypoint = self.get_current_waypoint()
        if waypoint:
            return waypoint['mission_type']
        return None

    def get_current_mission_params(self) -> Dict:
        """현재 미션 파라미터 반환"""
        waypoint = self.get_current_waypoint()
        if waypoint:
            return waypoint['params']
        return {}

    def get_waypoint_list(self) -> List[List[float]]:
        """웨이포인트 리스트 반환 (장애물 회피용)"""
        return [[wp['x'], wp['y']] for wp in self.waypoints]

    def get_total_waypoints(self) -> int:
        """전체 웨이포인트 수 반환"""
        return len(self.waypoints)

    def get_waypoint_index(self) -> int:
        """현재 웨이포인트 인덱스 반환"""
        return self.current_waypoint_index

    def is_mission_completed(self) -> bool:
        """모든 미션 완료 여부 확인"""
        return self.current_waypoint_index >= len(self.waypoints)
