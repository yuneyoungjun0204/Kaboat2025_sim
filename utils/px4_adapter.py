#!/usr/bin/env python3
"""
PX4 어댑터 모듈
- 센서 메시지 변환 (PX4 → 기존 시스템 형식)
- 명령 변환 (기존 시스템 → PX4 형식)
- 좌표계 변환 (LLA ↔ NED)
"""

import numpy as np
from math import sin, cos, atan2, pi, sqrt
from typing import Tuple, Optional
from dataclasses import dataclass
from pyproj import Transformer, Proj

from .config import Constants


@dataclass
class NEDPosition:
    """NED 좌표계 위치"""
    north: float
    east: float
    down: float


@dataclass
class VelocityYawCommand:
    """PX4 속도/yaw 명령"""
    velocity: float      # m/s
    yaw: float          # rad
    use_position: bool  # True=위치제어, False=속도제어


class CoordinateConverter:
    """LLA ↔ NED 좌표 변환기"""

    def __init__(self, lat_origin: float = None, lon_origin: float = None, alt_origin: float = None):
        """
        Args:
            lat_origin: 기준점 위도 (degrees)
            lon_origin: 기준점 경도 (degrees)
            alt_origin: 기준점 고도 (meters)
        """
        self.lat_origin = lat_origin or Constants.PX4.LAT_ORIGIN
        self.lon_origin = lon_origin or Constants.PX4.LON_ORIGIN
        self.alt_origin = alt_origin or Constants.PX4.ALT_ORIGIN

        # Projection 설정
        self.proj_lla = Proj(proj='latlong', ellps='WGS84')
        self.proj_ecef = Proj(proj='geocent', ellps='WGS84')
        self.transformer_to_ecef = Transformer.from_proj(self.proj_lla, self.proj_ecef)

        # 기준점 ECEF 좌표
        self.ref_x, self.ref_y, self.ref_z = self.transformer_to_ecef.transform(
            self.lon_origin, self.lat_origin, self.alt_origin
        )

        # ECEF to NED 회전 행렬
        lat_rad = np.radians(self.lat_origin)
        lon_rad = np.radians(self.lon_origin)
        self.R_ecef_to_ned = np.array([
            [-sin(lat_rad) * cos(lon_rad), -sin(lat_rad) * sin(lon_rad), cos(lat_rad)],
            [-sin(lon_rad), cos(lon_rad), 0],
            [-cos(lat_rad) * cos(lon_rad), -cos(lat_rad) * sin(lon_rad), -sin(lat_rad)]
        ])

    def lla_to_ned(self, lat: float, lon: float, alt: float) -> NEDPosition:
        """
        LLA 좌표를 NED 좌표로 변환

        Args:
            lat: 위도 (degrees)
            lon: 경도 (degrees)
            alt: 고도 (meters)

        Returns:
            NEDPosition: NED 좌표
        """
        x, y, z = self.transformer_to_ecef.transform(lon, lat, alt)
        ned = self.R_ecef_to_ned @ np.array([x - self.ref_x, y - self.ref_y, z - self.ref_z])
        return NEDPosition(north=ned[0], east=ned[1], down=ned[2])

    def ned_to_lla(self, north: float, east: float, down: float) -> Tuple[float, float, float]:
        """
        NED 좌표를 LLA 좌표로 변환

        Args:
            north: 북쪽 방향 (meters)
            east: 동쪽 방향 (meters)
            down: 아래 방향 (meters)

        Returns:
            Tuple[lat, lon, alt]: LLA 좌표
        """
        # NED to ECEF
        ned = np.array([north, east, down])
        ecef_diff = np.linalg.inv(self.R_ecef_to_ned) @ ned
        x = ecef_diff[0] + self.ref_x
        y = ecef_diff[1] + self.ref_y
        z = ecef_diff[2] + self.ref_z

        # ECEF to LLA
        transformer_to_lla = Transformer.from_proj(self.proj_ecef, self.proj_lla)
        lon, lat, alt = transformer_to_lla.transform(x, y, z)
        return lat, lon, alt


class PX4CommandConverter:
    """기존 시스템 명령 → PX4 명령 변환기"""

    def __init__(self):
        self.max_velocity = Constants.PX4.MAX_VELOCITY
        self.max_yaw_rate = Constants.PX4.MAX_YAW_RATE
        self.velocity_scale = Constants.PX4.VELOCITY_SCALE
        self.yaw_rate_scale = Constants.PX4.YAW_RATE_SCALE

        # 상태 변수
        self._current_yaw = 0.0
        self._last_time = None

    def update_current_yaw(self, yaw: float):
        """현재 yaw 값 업데이트 (라디안)"""
        self._current_yaw = yaw

    def convert_velocity_command(
        self,
        desired_speed: float,
        desired_moment: float,
        current_time_ns: int = None
    ) -> VelocityYawCommand:
        """
        기존 시스템의 속도/모멘트 명령을 PX4 형식으로 변환

        Args:
            desired_speed: 정규화된 전진 속도 (-1 ~ 1)
            desired_moment: 정규화된 yaw moment (-1 ~ 1)
            current_time_ns: 현재 시간 (나노초)

        Returns:
            VelocityYawCommand: PX4용 속도/yaw 명령
        """
        # 속도 변환: 정규화된 값 → m/s
        velocity = np.clip(desired_speed, -1.0, 1.0) * self.velocity_scale
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)

        # Yaw rate 계산
        yaw_rate = np.clip(desired_moment, -1.0, 1.0) * self.yaw_rate_scale
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # Yaw 적분 (시간 기반)
        if current_time_ns is not None and self._last_time is not None:
            dt = (current_time_ns - self._last_time) / 1e9
            if dt > 0 and dt < 1.0:  # 유효한 dt 범위
                self._current_yaw += yaw_rate * dt
                # [-pi, pi] 범위로 정규화
                self._current_yaw = self._normalize_angle(self._current_yaw)

        if current_time_ns is not None:
            self._last_time = current_time_ns

        return VelocityYawCommand(
            velocity=velocity,
            yaw=self._current_yaw,
            use_position=False
        )

    def convert_position_command(
        self,
        x_error: float,
        y_error: float,
        target_yaw: float = None
    ) -> Tuple[float, float, float]:
        """
        위치 오차를 PX4 위치 setpoint로 변환

        Args:
            x_error: X 방향 위치 오차 (m)
            y_error: Y 방향 위치 오차 (m)
            target_yaw: 목표 yaw (rad), None이면 현재 yaw 유지

        Returns:
            Tuple[x, y, yaw]: 위치 setpoint
        """
        yaw = target_yaw if target_yaw is not None else self._current_yaw
        return x_error, y_error, yaw

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """각도를 [-pi, pi] 범위로 정규화"""
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def heading_deg_to_yaw_rad(self, heading_deg: float) -> float:
        """
        Heading (도) → Yaw (라디안) 변환

        Heading: 북쪽 기준 시계방향 (0-360)
        Yaw: 동쪽 기준 반시계방향 (-pi ~ pi)
        """
        # Heading to Yaw 변환
        yaw_rad = np.radians(90.0 - heading_deg)
        return self._normalize_angle(yaw_rad)

    def yaw_rad_to_heading_deg(self, yaw_rad: float) -> float:
        """
        Yaw (라디안) → Heading (도) 변환
        """
        heading_deg = 90.0 - np.degrees(yaw_rad)
        while heading_deg < 0:
            heading_deg += 360
        while heading_deg >= 360:
            heading_deg -= 360
        return heading_deg


class PX4SensorAdapter:
    """PX4 센서 데이터 어댑터"""

    def __init__(self):
        self.coord_converter = CoordinateConverter()

        # 캐시된 센서 데이터
        self._position_ned: Optional[NEDPosition] = None
        self._heading_rad: float = 0.0
        self._velocity_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def process_global_position(self, lat: float, lon: float, alt: float) -> NEDPosition:
        """
        VehicleGlobalPosition 메시지 처리

        Args:
            lat: 위도 (degrees)
            lon: 경도 (degrees)
            alt: 고도 (meters)

        Returns:
            NEDPosition: NED 좌표
        """
        self._position_ned = self.coord_converter.lla_to_ned(lat, lon, alt)
        return self._position_ned

    def process_local_position(self, heading: float, vx: float, vy: float, vz: float = 0.0):
        """
        VehicleLocalPosition 메시지 처리

        Args:
            heading: 헤딩 (radians)
            vx: X 속도 (m/s)
            vy: Y 속도 (m/s)
            vz: Z 속도 (m/s)
        """
        self._heading_rad = heading
        self._velocity_ned = (vx, vy, vz)

    def get_position_ned(self) -> Optional[NEDPosition]:
        """현재 NED 위치 반환"""
        return self._position_ned

    def get_heading_rad(self) -> float:
        """현재 헤딩 반환 (라디안)"""
        return self._heading_rad

    def get_heading_deg(self) -> float:
        """현재 헤딩 반환 (도)"""
        return np.degrees(self._heading_rad)

    def get_velocity_ned(self) -> Tuple[float, float, float]:
        """현재 NED 속도 반환"""
        return self._velocity_ned

    def get_speed(self) -> float:
        """현재 속력 반환 (m/s)"""
        vx, vy, _ = self._velocity_ned
        return sqrt(vx**2 + vy**2)

    def convert_pointcloud_to_laserscan(
        self,
        points: np.ndarray,
        angle_min: float = -pi,
        angle_max: float = pi,
        angle_increment: float = None,
        range_min: float = 0.1,
        range_max: float = 100.0,
        height_min: float = -0.5,
        height_max: float = 2.0
    ) -> np.ndarray:
        """
        PointCloud2 데이터를 LaserScan 형식으로 변환

        Args:
            points: Nx3 점군 배열 (x, y, z)
            angle_min: 최소 각도 (rad)
            angle_max: 최대 각도 (rad)
            angle_increment: 각도 증분 (rad), None이면 자동 계산
            range_min: 최소 거리 (m)
            range_max: 최대 거리 (m)
            height_min: 최소 높이 필터 (m)
            height_max: 최대 높이 필터 (m)

        Returns:
            np.ndarray: LaserScan ranges 배열
        """
        if angle_increment is None:
            angle_increment = np.radians(1.0)  # 기본 1도

        num_bins = int((angle_max - angle_min) / angle_increment)
        ranges = np.full(num_bins, range_max)

        if points.shape[0] == 0:
            return ranges

        # 높이 필터링
        mask = (points[:, 2] > height_min) & (points[:, 2] < height_max)
        filtered_points = points[mask]

        if filtered_points.shape[0] == 0:
            return ranges

        # 2D 거리 및 각도 계산
        distances = np.hypot(filtered_points[:, 0], filtered_points[:, 1])
        angles = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])

        # 거리 필터링
        valid_mask = (distances >= range_min) & (distances <= range_max)
        distances = distances[valid_mask]
        angles = angles[valid_mask]

        # 각도를 bin 인덱스로 변환
        bin_indices = ((angles - angle_min) / angle_increment).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # 각 bin에서 최소 거리 선택
        for i, dist in enumerate(distances):
            bin_idx = bin_indices[i]
            if dist < ranges[bin_idx]:
                ranges[bin_idx] = dist

        return ranges


class PX4BridgeData:
    """PX4 브릿지 토픽 데이터 구조체"""

    def __init__(self):
        self.velocity_yaw_cmd = [0.0, 0.0]  # [velocity, yaw]
        self.position_error = [0.0, 0.0]    # [x_error, y_error]
        self.control_flag = False           # False=속도제어, True=위치제어

    def set_velocity_command(self, velocity: float, yaw: float):
        """속도 명령 설정"""
        self.velocity_yaw_cmd = [float(velocity), float(yaw)]
        self.control_flag = False

    def set_position_command(self, x_error: float, y_error: float):
        """위치 명령 설정"""
        self.position_error = [float(x_error), float(y_error)]
        self.control_flag = True
