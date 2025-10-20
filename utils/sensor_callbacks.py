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
            self.agent_position = np.array([gps_data['utm_x'], gps_data['utm_y']], dtype=np.float32)
            if not self.reference_point_set:
                self.reference_point_set = True

    def imu_callback(self, msg: Imu) -> None:
        """IMU 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.agent_heading = imu_data['yaw_degrees']
        if self.agent_heading < 0:
            self.agent_heading += 360.0

        current_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        self.angular_velocity_y = np.clip(
            current_angular_velocity[2],
            Constants.ANGULAR_VELOCITY_LIMIT[0],
            Constants.ANGULAR_VELOCITY_LIMIT[1]
        )

    def lidar_callback(self, msg: LaserScan) -> None:
        """LiDAR 콜백"""
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        raw_ranges = np.full(
            Constants.LIDAR_ARRAY_SIZE,
            Constants.MAX_LIDAR_DISTANCE,
            dtype=np.float32
        )

        for i in range(len(ranges)):
            angle_rad = angle_min + i * angle_increment
            angle_deg = np.degrees(angle_rad)

            if Constants.LIDAR_ANGLE_RANGE[0] <= angle_deg <= Constants.LIDAR_ANGLE_RANGE[1]:
                distance = ranges[i]
                if np.isinf(distance) or np.isnan(distance) or distance >= Constants.MAX_LIDAR_DISTANCE:
                    distance = Constants.MAX_LIDAR_DISTANCE

                idx = int(angle_deg + 100)
                idx = max(0, min(Constants.LIDAR_ARRAY_SIZE - 1, idx))
                raw_ranges[idx] = distance

        self.lidar_distances = raw_ranges.astype(np.float32)

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
