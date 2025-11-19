#!/usr/bin/env python3
"""
센서 콜백 핸들러 모듈
"""

import numpy as np
from scipy.ndimage import median_filter
from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from typing import Optional, Callable
from utils.config import Constants
from utils.sensor_preprocessing import SensorDataManager, normalize_angle_180
from utils.detection_system import MissionType


class LidarFilter:
    """
    LiDAR 데이터 필터링 클래스
    - Range Filter: 유효 범위 검증
    - Median Filter: 공간적 이상치 제거 (Spike noise 제거)
    - Temporal EMA Filter: 시간적 부드러움 (이전 프레임과 가중 평균)
    """

    def __init__(self,
                 min_valid_distance: float = 0.1,
                 max_valid_distance: float = 100.0,
                 median_window: int = 5,
                 temporal_alpha: float = 0.4,
                 array_size: int = 201):
        """
        Args:
            min_valid_distance: 최소 유효 거리 (미터)
            max_valid_distance: 최대 유효 거리 (미터)
            median_window: Median 필터 윈도우 크기 (홀수 권장)
            temporal_alpha: Temporal EMA 필터 계수 (0~1, 낮을수록 부드러움)
            array_size: LiDAR 배열 크기
        """
        self.min_valid_distance = min_valid_distance
        self.max_valid_distance = max_valid_distance
        self.median_window = median_window if median_window % 2 == 1 else median_window + 1  # 홀수로 보장
        self.temporal_alpha = temporal_alpha
        self.array_size = array_size

        # 이전 프레임 데이터 (Temporal filtering용)
        self.previous_distances = np.full(array_size, max_valid_distance, dtype=np.float32)
        self.initialized = False

    def apply_range_filter(self, distances: np.ndarray) -> np.ndarray:
        """
        Range Filter: 유효 범위를 벗어난 값을 최대값으로 대체

        Args:
            distances: 입력 거리 배열

        Returns:
            필터링된 거리 배열
        """
        filtered = distances.copy()

        # inf, nan, 0 등의 잘못된 값 처리
        invalid_mask = np.isnan(filtered) | np.isinf(filtered) | (filtered <= 0.0)
        filtered[invalid_mask] = self.max_valid_distance

        # 최소값보다 작은 값 처리
        too_close_mask = filtered < self.min_valid_distance
        filtered[too_close_mask] = self.max_valid_distance

        # 최대값보다 큰 값 처리
        too_far_mask = filtered > self.max_valid_distance
        filtered[too_far_mask] = self.max_valid_distance

        return filtered

    def apply_median_filter(self, distances: np.ndarray) -> np.ndarray:
        """
        Median Filter: 공간적 이상치 제거
        인접한 각도의 중간값으로 튀는 값 제거

        Args:
            distances: 입력 거리 배열

        Returns:
            필터링된 거리 배열
        """
        # scipy의 median_filter 사용 (경계 처리 자동)
        # mode='reflect': 경계에서 반사 패딩 사용
        filtered = median_filter(distances, size=self.median_window, mode='reflect')
        return filtered.astype(np.float32)

    def apply_temporal_filter(self, distances: np.ndarray) -> np.ndarray:
        """
        Temporal EMA Filter: 시간적 부드러움 추가
        현재 프레임과 이전 프레임의 가중 평균

        Formula: filtered = alpha * current + (1 - alpha) * previous

        Args:
            distances: 입력 거리 배열

        Returns:
            필터링된 거리 배열
        """
        if not self.initialized:
            # 첫 프레임은 그대로 사용
            self.previous_distances = distances.copy()
            self.initialized = True
            return distances

        # EMA 필터 적용
        filtered = self.temporal_alpha * distances + (1.0 - self.temporal_alpha) * self.previous_distances

        # 다음 프레임을 위해 저장
        self.previous_distances = filtered.copy()

        return filtered.astype(np.float32)

    def filter(self, distances: np.ndarray) -> np.ndarray:
        """
        전체 필터링 파이프라인 적용
        1. Range Filter
        2. Median Filter
        3. Temporal EMA Filter

        Args:
            distances: 입력 거리 배열

        Returns:
            필터링된 거리 배열
        """
        # 1단계: Range Filter (유효하지 않은 값 제거)
        filtered = self.apply_range_filter(distances)

        # 2단계: Median Filter (공간적 이상치 제거)
        filtered = self.apply_median_filter(filtered)

        # 3단계: Temporal Filter (시간적 부드러움)
        filtered = self.apply_temporal_filter(filtered)

        return filtered

    def reset(self):
        """필터 상태 초기화"""
        self.previous_distances.fill(self.max_valid_distance)
        self.initialized = False

    def get_filter_stats(self, raw: np.ndarray, filtered: np.ndarray) -> dict:
        """
        필터링 전후 통계 정보 계산

        Args:
            raw: 원본 데이터
            filtered: 필터링된 데이터

        Returns:
            통계 정보 딕셔너리
        """
        # 유효한 값만 선택 (max_distance가 아닌 값)
        valid_raw_mask = raw < self.max_valid_distance
        valid_filtered_mask = filtered < self.max_valid_distance

        stats = {
            'raw_mean': np.mean(raw[valid_raw_mask]) if valid_raw_mask.any() else 0.0,
            'filtered_mean': np.mean(filtered[valid_filtered_mask]) if valid_filtered_mask.any() else 0.0,
            'raw_std': np.std(raw[valid_raw_mask]) if valid_raw_mask.any() else 0.0,
            'filtered_std': np.std(filtered[valid_filtered_mask]) if valid_filtered_mask.any() else 0.0,
            'raw_min': np.min(raw[valid_raw_mask]) if valid_raw_mask.any() else 0.0,
            'filtered_min': np.min(filtered[valid_filtered_mask]) if valid_filtered_mask.any() else 0.0,
            'max_change': np.max(np.abs(filtered - raw)),
            'valid_points_raw': np.sum(valid_raw_mask),
            'valid_points_filtered': np.sum(valid_filtered_mask),
        }

        return stats


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

        # 웨이포인트 관리자 (나중에 설정됨)
        self.waypoint_manager = None

        # PX4 모드용 초기 GPS 위치 (기준점)
        self.initial_lat = None
        self.initial_lon = None

        # LiDAR 필터 초기화
        self.lidar_filter = None
        self.lidar_frame_count = 0  # 필터링 통계용 프레임 카운터
        if Constants.LIDAR_FILTER_ENABLED:
            self.lidar_filter = LidarFilter(
                min_valid_distance=Constants.LIDAR_MIN_VALID_DISTANCE,
                max_valid_distance=Constants.LIDAR_MAX_VALID_DISTANCE,
                median_window=Constants.LIDAR_MEDIAN_FILTER_WINDOW,
                temporal_alpha=Constants.LIDAR_TEMPORAL_FILTER_ALPHA,
                array_size=Constants.LIDAR_ARRAY_SIZE
            )
            self.logger.info(
                f"✓ LiDAR 필터 활성화 - Median window: {Constants.LIDAR_MEDIAN_FILTER_WINDOW}, "
                f"Temporal alpha: {Constants.LIDAR_TEMPORAL_FILTER_ALPHA}, "
                f"Range: [{Constants.LIDAR_MIN_VALID_DISTANCE}, {Constants.LIDAR_MAX_VALID_DISTANCE}]m"
            )

    def set_waypoint_manager(self, waypoint_manager):
        """웨이포인트 관리자 설정"""
        self.waypoint_manager = waypoint_manager

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
                # 웨이포인트 관리자에 초기 위치 설정 (웨이포인트 재계산)
                if self.waypoint_manager is not None:
                    self.waypoint_manager.set_initial_position(msg.latitude, msg.longitude)
                    self.logger.info(
                        f"초기 위치 설정: lat={msg.latitude:.8f}, lon={msg.longitude:.8f}"
                    )

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

    def px4_odometry_callback(self, msg) -> None:
        """
        PX4 Odometry 콜백 - 각속도와 heading 업데이트

        VehicleOdometry 메시지에서:
        - angular_velocity[2]: z축 각속도 (yaw rate, rad/s)
        - q: quaternion (orientation)

        Note: PX4 모드에서만 사용됨
        """
        import math

        # 각속도 처리 (z축, rad/s → deg/s)
        angular_velocity_z_rad = msg.angular_velocity[2]
        angular_velocity_z_deg = np.degrees(angular_velocity_z_rad)
        self.angular_velocity_y = np.clip(
            angular_velocity_z_deg,
            Constants.ANGULAR_VELOCITY_LIMIT[0],
            Constants.ANGULAR_VELOCITY_LIMIT[1]
        )

        # Quaternion에서 Yaw 추출
        # q = [w, x, y, z] (PX4 순서)
        q = msg.q
        # Yaw (z축 회전)
        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = np.degrees(yaw_rad)

        # NED 좌표계로 정규화 (-180 ~ 180)
        self.agent_heading = normalize_angle_180(yaw_deg)

    def px4_global_position_callback(self, msg) -> None:
        """
        PX4 VehicleGlobalPosition 콜백 - GPS 위치 업데이트

        VehicleGlobalPosition 메시지에서:
        - lat, lon: 위도/경도 (degrees)
        - alt: 고도 (meters)

        Note: PX4 모드에서 기존 GPS 토픽 대신 사용됨
        """
        from utils.waypoint_manager import gps_to_local

        # 미션 시작 시 첫 번째 현재 위치를 기준점으로 설정
        if self.initial_lat is None:
            self.initial_lat = msg.lat
            self.initial_lon = msg.lon
            self.reference_point_set = True

            self.logger.info(
                f"PX4 GPS 초기 위치 설정: lat={msg.lat:.8f}, lon={msg.lon:.8f}"
            )

            # waypoint_manager에도 초기 위치 설정
            if self.waypoint_manager is not None:
                self.waypoint_manager.set_initial_position(msg.lat, msg.lon)
                self.logger.info(f"waypoint_manager 초기 위치 설정 완료")

                # 재계산된 웨이포인트 정보 로깅
                for i, wp in enumerate(self.waypoint_manager.waypoints):
                    self.logger.info(
                        f"  웨이포인트 {i}: x={wp['x']:.2f}m, y={wp['y']:.2f}m, "
                        f"미션={wp['mission_type'].name}"
                    )
            else:
                self.logger.warn("waypoint_manager가 None입니다! 웨이포인트 재계산 불가")

        # 초기 위치 기준으로 현재 위치를 로컬 좌표(m)로 변환
        x_local, y_local = gps_to_local(msg.lat, msg.lon, self.initial_lat, self.initial_lon)
        self.agent_position = np.array(
            [x_local, y_local],  # [x, y] = [Easting, Northing]
            dtype=np.float32
        )

    def px4_local_position_callback(self, msg) -> None:
        """
        PX4 VehicleLocalPosition 콜백 - heading 업데이트만 담당

        VehicleLocalPosition 메시지에서:
        - heading: yaw 각도 (라디안)

        Note: PX4 모드에서만 사용됨
        Note: 위치 업데이트는 px4_global_position_callback에서 담당
              (초기 GPS 위치를 기준점으로 사용하기 위함)
        """
        # Heading 업데이트 (heading_good_for_control과 관계없이 항상 업데이트)
        # 위치 업데이트는 px4_global_position_callback에서 처리
        yaw_deg = np.degrees(msg.heading)
        self.agent_heading = normalize_angle_180(yaw_deg)

    def livox_imu_callback(self, msg: Imu) -> None:
        """
        Livox LiDAR IMU 콜백 - 각속도 업데이트 (ONNX 모델용)

        Livox LiDAR 내장 IMU에서 각속도 데이터를 가져옴
        ONNX 모델 입력에 사용되는 angular_velocity_y (Z축 각속도) 업데이트

        각속도: rad/s → deg/s, Z축 (+ = CCW, - = CW)
        """
        # Z축 각속도를 deg/s로 변환 후 클리핑
        angular_velocity_z_deg = np.degrees(msg.angular_velocity.z)
        self.angular_velocity_y = np.clip(
            angular_velocity_z_deg,
            Constants.ANGULAR_VELOCITY_LIMIT[0],
            Constants.ANGULAR_VELOCITY_LIMIT[1]
        )

    def lidar_callback(self, msg: LaserScan) -> None:
        """LiDAR 콜백 (필터링 포함)"""
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        raw_ranges = np.full(
            Constants.LIDAR_ARRAY_SIZE,
            Constants.MAX_LIDAR_DISTANCE,
            dtype=np.float32
        )

        # 1. 원본 데이터 변환 및 스케일 적용
        for i in range(len(ranges)):
            angle_rad = angle_min + i * angle_increment
            angle_deg = np.degrees(angle_rad)

            if Constants.LIDAR_ANGLE_RANGE[0] <= angle_deg <= Constants.LIDAR_ANGLE_RANGE[1]:
                distance = ranges[i]
                if np.isinf(distance) or np.isnan(distance) or distance >= Constants.MAX_LIDAR_DISTANCE:
                    distance = Constants.MAX_LIDAR_DISTANCE
                else:
                    distance = distance * Constants.LIDAR_SCALE_FACTOR

                idx = int(angle_deg + 100)
                idx = max(0, min(Constants.LIDAR_ARRAY_SIZE - 1, idx))
                raw_ranges[idx] = distance

        # 2. 필터 적용 (활성화된 경우)
        if self.lidar_filter is not None:
            filtered_ranges = self.lidar_filter.filter(raw_ranges)
            self.lidar_distances = filtered_ranges

            # 디버그 로깅 (주기적)
            self.lidar_frame_count += 1
            if (Constants.LIDAR_FILTER_DEBUG_LOG_INTERVAL > 0 and
                self.lidar_frame_count % Constants.LIDAR_FILTER_DEBUG_LOG_INTERVAL == 0):
                stats = self.lidar_filter.get_filter_stats(raw_ranges, filtered_ranges)
                self.logger.info(
                    f"📊 LiDAR 필터링 통계 (프레임 {self.lidar_frame_count}): "
                    f"평균 {stats['raw_mean']:.2f}→{stats['filtered_mean']:.2f}m, "
                    f"표준편차 {stats['raw_std']:.2f}→{stats['filtered_std']:.2f}m, "
                    f"최대 변화 {stats['max_change']:.2f}m, "
                    f"유효점 {stats['valid_points_raw']}→{stats['valid_points_filtered']}"
                )
        else:
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
