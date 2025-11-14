#!/usr/bin/env python3
"""
센서 콜백 핸들러 모듈
"""

import numpy as np
from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from typing import Optional, Callable
from utils.config import Constants
from utils.sensor_preprocessing import SensorDataManager
from utils.geometry import normalize_angle_180  # 통합된 유틸리티 사용
from utils.detection_system import MissionType


class SensorCallbackHandler:
    """센서 콜백 처리 클래스"""

    def __init__(self, bridge: CvBridge, sensor_manager: SensorDataManager, logger):
        """
        Args:
            bridge: CvBridge 인스턴스
            sensor_manager: SensorDataManager 인스턴스
            logger: ROS2 logger
        """
        self.bridge = bridge
        self.sensor_manager = sensor_manager
        self.logger = logger

        # 센서 데이터
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.lidar_distances = np.zeros(Constants.LIDAR_ARRAY_SIZE, dtype=np.float32)
        self.current_image = None
        self.reference_point_set = False

        # LiDAR 직교좌표 (sensor_preprocessing.py에서 가공된 데이터)
        self.lidar_cartesian_x = np.array([])
        self.lidar_cartesian_y = np.array([])

        # 수동 목표 위치
        self.manual_target_x = None
        self.manual_target_y = None

    def image_callback(self, msg: Image) -> None:
        """이미지 콜백"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.logger.error(f'CvBridge Error: {e}')

    def gps_callback(self, msg: NavSatFix) -> None:
        """GPS 콜백"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            self.agent_position = np.array([gps_data['utm_y'], gps_data['utm_x']], dtype=np.float32)
            if not self.reference_point_set:
                self.reference_point_set = True

    def imu_callback(self, msg: Imu) -> None:
        """
        IMU 콜백 - NED 좌표계 기준

        Heading: -180~180도 (0° = North, +90° = East, -90° = West, ±180° = South)
        Angular velocity: rad/s → deg/s, Z축 (+ = CCW, - = CW)
        """
        imu_data = self.sensor_manager.process_imu_data(msg)

        # Yaw 각도를 -180~180도 범위로 정규화 (NED 좌표계)
        self.agent_heading = normalize_angle_180(imu_data['yaw_degrees'])

        # 각속도 처리 (Z축, rad/s → deg/s)
        # NED 좌표계: + = 반시계방향(CCW), - = 시계방향(CW)
        current_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Z축 각속도를 deg/s로 변환 후 클리핑
        angular_velocity_z_deg = np.degrees(current_angular_velocity[2])
        self.angular_velocity_y = np.clip(
            angular_velocity_z_deg,
            Constants.ANGULAR_VELOCITY_LIMIT[0],
            Constants.ANGULAR_VELOCITY_LIMIT[1]
        )

    def lidar_callback(self, msg: LaserScan) -> None:
        """
        LiDAR 콜백 - sensor_preprocessing.py의 LiDARProcessor 사용

        전처리 과정:
        1. LiDARProcessor로 유효 범위 필터링 및 노이즈 제거
        2. 가공된 데이터를 기존 배열 형식으로 변환
        3. 직교좌표도 함께 저장
        """
        # sensor_preprocessing.py의 LiDARProcessor 사용
        lidar_data = self.sensor_manager.process_lidar_data(msg)

        # 가공된 데이터를 기존 배열 형식으로 변환
        raw_ranges = np.full(
            Constants.LIDAR_ARRAY_SIZE,
            Constants.MAX_LIDAR_DISTANCE,
            dtype=np.float32
        )

        # 필터링된 ranges와 angles를 배열에 매핑
        # 주의: 스케일 팩터는 이미 sensor_preprocessing.py에서 적용됨
        if len(lidar_data['ranges']) > 0:
            for distance, angle_rad in zip(lidar_data['ranges'], lidar_data['angles']):
                angle_deg = np.degrees(angle_rad)

                if Constants.LIDAR_ANGLE_RANGE[0] <= angle_deg <= Constants.LIDAR_ANGLE_RANGE[1]:
                    # 최대 거리 제한
                    if distance >= Constants.MAX_LIDAR_DISTANCE:
                        distance = Constants.MAX_LIDAR_DISTANCE

                    idx = int(angle_deg + 100)
                    idx = max(0, min(Constants.LIDAR_ARRAY_SIZE - 1, idx))
                    raw_ranges[idx] = distance

        self.lidar_distances = raw_ranges.astype(np.float32)

        # 직교좌표도 저장 (필요시 사용)
        self.lidar_cartesian_x, self.lidar_cartesian_y = self.sensor_manager.get_lidar_cartesian()

    def waypoint_callback(self, msg: Point, waypoint_add_callback: Optional[Callable] = None) -> None:
        """
        웨이포인트 콜백 - 수동 목표 설정 및 자동 웨이포인트 추가

        Args:
            msg: Point 메시지
            waypoint_add_callback: 웨이포인트 추가 콜백 함수
        """
        # trajectory_viz에서 클릭한 좌표를 수동 목표로 저장
        self.manual_target_x = float(msg.x)
        self.manual_target_y = float(msg.y)
        self.logger.info(f"수동 목표 설정: ({self.manual_target_x:.1f}, {self.manual_target_y:.1f})")

        # 웨이포인트 추가 콜백 실행
        if waypoint_add_callback:
            waypoint_add_callback(msg)

    def has_valid_data(self) -> bool:
        """
        센서 데이터가 유효한지 확인

        Returns:
            bool: GPS 데이터가 초기화되었는지 여부
        """
        return self.reference_point_set

    def get_lidar_distance_at_angle(self, angle_deg: float) -> float:
        """주어진 각도에서 LiDAR 거리 가져오기"""
        # 각도 정규화
        while angle_deg > Constants.LIDAR_ANGLE_RANGE[1]:
            angle_deg -= 360
        while angle_deg < Constants.LIDAR_ANGLE_RANGE[0]:
            angle_deg += 360

        if -180 <= angle_deg <= 180:
            idx = int(angle_deg + 100)
            idx = max(0, min(Constants.LIDAR_ARRAY_SIZE - 1, idx))
            return self.lidar_distances[idx]
        return Constants.MAX_LIDAR_DISTANCE

    # Alias for compatibility
    def get_lidar_distance_at_angle_degrees(self, angle_deg: float) -> float:
        """주어진 각도에서 LiDAR 거리 가져오기 (호환성 유지용 alias)"""
        return self.get_lidar_distance_at_angle(angle_deg)
