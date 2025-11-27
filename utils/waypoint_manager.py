#!/usr/bin/env python3
"""
웨이포인트 관리 모듈
- 웨이포인트 추가/관리
- 미션 전환 로직
- GPS 좌표 <-> 로컬 좌표 변환 지원
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .detection_system import MissionType
from .config import Constants


def gps_to_local(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    GPS 좌표(위도/경도)를 로컬 좌표(미터)로 변환

    Args:
        lat: 목표 위도
        lon: 목표 경도
        ref_lat: 기준점 위도
        ref_lon: 기준점 경도

    Returns:
        (x, y): 로컬 좌표 (미터) - x=Easting, y=Northing
    """
    # 위도/경도를 미터 단위로 근사 변환
    # 1도 ≈ 111,320m (위도), 1도 ≈ 111,320 * cos(위도) m (경도)
    lat_m = (lat - ref_lat) * 111320.0
    lon_m = (lon - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    return lon_m, lat_m  # x=Easting, y=Northing


def local_to_gps(x: float, y: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    로컬 좌표(미터)를 GPS 좌표(위도/경도)로 변환

    Args:
        x: 로컬 X 좌표 (Easting, 미터)
        y: 로컬 Y 좌표 (Northing, 미터)
        ref_lat: 기준점 위도
        ref_lon: 기준점 경도

    Returns:
        (lat, lon): GPS 좌표 (위도, 경도)
    """
    lat = ref_lat + (y / 111320.0)
    lon = ref_lon + (x / (111320.0 * np.cos(np.radians(ref_lat))))
    return lat, lon


class WaypointManager:
    """
    웨이포인트 관리 시스템
    - 로컬 좌표 (MODE=0) 또는 GPS 좌표 (MODE=1) 지원
    """

    def __init__(self, waypoint_mode: int = None):
        """
        Args:
            waypoint_mode: 웨이포인트 좌표계 모드
                - 0: 로컬 좌표 (UTM 상대 좌표, 미터)
                - 1: GPS 좌표 (위도/경도)
                - None: Constants.WAYPOINT_MODE 사용
        """
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_mode = waypoint_mode if waypoint_mode is not None else Constants.WAYPOINT_MODE
        self.gps_reference_lat = Constants.GPS_REFERENCE_LAT
        self.gps_reference_lon = Constants.GPS_REFERENCE_LON

        # 미션 시작 시 첫 번째 현재 위치를 기준점으로 저장
        self.initial_position_lat = None
        self.initial_position_lon = None
        self.initial_position_set = False

    def add_waypoint(self, x: float, y: float, mission_type: MissionType,
                    radius: float = None, params: Optional[Dict] = None,
                    is_gps: bool = None):
        """
        웨이포인트 추가 (로컬 좌표 또는 GPS 좌표)

        Args:
            x: X 좌표 (미터) 또는 위도
            y: Y 좌표 (미터) 또는 경도
            mission_type: 미션 타입
            radius: 도달 판정 반경 (미터), None이면 기본값 사용
            params: 미션별 파라미터
            is_gps: True이면 x,y를 GPS 좌표(위도/경도)로 처리,
                   None이면 self.waypoint_mode 사용
        """
        # GPS 좌표인지 확인
        use_gps = is_gps if is_gps is not None else (self.waypoint_mode == 1)

        # GPS 좌표를 로컬 좌표로 변환
        if use_gps:
            lat, lon = x, y
            x_local, y_local = gps_to_local(lat, lon, self.gps_reference_lat, self.gps_reference_lon)
            waypoint = {
                'x': x_local,
                'y': y_local,
                'lat': lat,  # 원본 GPS 좌표 저장
                'lon': lon,
                'mission_type': mission_type,
                'radius': radius if radius is not None else Constants.DEFAULT_WAYPOINT_RADIUS,
                'params': params if params else {},
                'is_gps': True
            }
        else:
            waypoint = {
                'x': x,
                'y': y,
                'mission_type': mission_type,
                'radius': radius if radius is not None else Constants.DEFAULT_WAYPOINT_RADIUS,
                'params': params if params else {},
                'is_gps': False
            }

        self.waypoints.append(waypoint)

    def setup_predefined_waypoints(self):
        """
        미리 정의된 웨이포인트 설정 (config에서 가져오기)

        WAYPOINT_MODE에 따라 자동으로 좌표계 변환:
        - MODE=0: x,y를 로컬 좌표(미터)로 처리
        - MODE=1: x,y를 GPS 좌표(위도/경도)로 처리
        """
        # mission_type 문자열을 MissionType enum으로 변환하는 매핑
        mission_type_map = {
            'PASS_BETWEEN_BUOYS': MissionType.PASS_BETWEEN_BUOYS,
            'CIRCLE_BUOY': MissionType.CIRCLE_BUOY,
            'WAYPOINT_FOLLOW': MissionType.WAYPOINT_FOLLOW,
            'OBSTACLE_AVOID': MissionType.OBSTACLE_AVOID,
            'HEADING_ALIGN': MissionType.HEADING_ALIGN,
            'DOCK_MODE': MissionType.DOCK_MODE,
            'ROTATION': MissionType.ROTATION
        }

        for x, y, mission_type_str, radius, params in Constants.PREDEFINED_WAYPOINTS:
            mission_type = mission_type_map.get(mission_type_str)
            if mission_type:
                # waypoint_mode에 따라 is_gps 자동 설정
                self.add_waypoint(x, y, mission_type, radius, params, is_gps=None)

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

    def get_waypoint_gps_info(self, index: int = None) -> Optional[Dict]:
        """
        웨이포인트의 GPS 정보 반환

        Args:
            index: 웨이포인트 인덱스 (None이면 현재 웨이포인트)

        Returns:
            GPS 정보 딕셔너리 또는 None
            - 'lat': 위도
            - 'lon': 경도
            - 'is_gps': GPS 좌표로 추가되었는지 여부
        """
        if index is None:
            index = self.current_waypoint_index

        if index < 0 or index >= len(self.waypoints):
            return None

        waypoint = self.waypoints[index]

        if waypoint.get('is_gps', False):
            return {
                'lat': waypoint['lat'],
                'lon': waypoint['lon'],
                'is_gps': True
            }
        else:
            # 로컬 좌표를 GPS 좌표로 변환
            lat, lon = local_to_gps(
                waypoint['x'], waypoint['y'],
                self.gps_reference_lat, self.gps_reference_lon
            )
            return {
                'lat': lat,
                'lon': lon,
                'is_gps': False
            }

    def set_gps_reference(self, lat: float, lon: float):
        """
        GPS 기준점 설정 (런타임에 변경 가능)

        Args:
            lat: 기준점 위도
            lon: 기준점 경도
        """
        self.gps_reference_lat = lat
        self.gps_reference_lon = lon

    def get_waypoint_mode_str(self) -> str:
        """현재 웨이포인트 모드 문자열 반환"""
        return "GPS 좌표" if self.waypoint_mode == 1 else "로컬 좌표"

    def set_initial_position(self, lat: float, lon: float):
        """
        미션 시작 시 첫 번째 현재 위치를 기준점으로 설정
        모든 GPS 웨이포인트의 x, y를 미션 시작 위치 기준으로 재계산

        Args:
            lat: 현재 위도
            lon: 현재 경도
        """
        if not self.initial_position_set:
            self.initial_position_lat = lat
            self.initial_position_lon = lon
            self.initial_position_set = True

            # 모든 GPS 웨이포인트의 x, y를 미션 시작 위치 기준으로 재계산
            self._recalculate_all_waypoints_from_initial()

    def _recalculate_all_waypoints_from_initial(self):
        """
        모든 GPS 웨이포인트의 x, y를 미션 시작 위치 기준으로 재계산
        """
        if not self.initial_position_set:
            return

        for waypoint in self.waypoints:
            if waypoint.get('is_gps', False):
                # GPS 좌표를 미션 시작 위치 기준 로컬 좌표로 재계산
                x_local, y_local = gps_to_local(
                    waypoint['lat'], waypoint['lon'],
                    self.initial_position_lat, self.initial_position_lon
                )
                waypoint['x'] = x_local
                waypoint['y'] = y_local

    def get_waypoint_relative_to_initial(self, waypoint_index: int = None) -> Tuple[float, float]:
        """
        현재 웨이포인트를 미션 시작 위치 기준 상대 좌표(m)로 반환

        Args:
            waypoint_index: 웨이포인트 인덱스 (None이면 현재 웨이포인트)

        Returns:
            (x, y): 미션 시작 위치 기준 상대 좌표 (미터)
        """
        if waypoint_index is None:
            waypoint_index = self.current_waypoint_index

        if waypoint_index < 0 or waypoint_index >= len(self.waypoints):
            return 0.0, 0.0

        waypoint = self.waypoints[waypoint_index]

        # 미션 시작 위치가 설정되지 않은 경우 기존 방식 사용
        if not self.initial_position_set:
            return waypoint['x'], waypoint['y']

        # GPS 좌표로 추가된 웨이포인트인 경우
        if waypoint.get('is_gps', False):
            # 미션 시작 위치를 기준으로 m 변환
            x_local, y_local = gps_to_local(
                waypoint['lat'], waypoint['lon'],
                self.initial_position_lat, self.initial_position_lon
            )
            return x_local, y_local
        else:
            # 로컬 좌표로 추가된 경우 그대로 반환
            return waypoint['x'], waypoint['y']

    def is_initial_position_set(self) -> bool:
        """미션 시작 위치가 설정되었는지 확인"""
        return self.initial_position_set

    def get_initial_position(self) -> Optional[Tuple[float, float]]:
        """미션 시작 위치 반환 (lat, lon)"""
        if self.initial_position_set:
            return self.initial_position_lat, self.initial_position_lon
        return None
