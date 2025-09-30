#!/usr/bin/env python3
"""
센서 데이터 전처리 모듈
- GPS to UTM 변환
- LiDAR 데이터 처리 및 좌표 변환
- IMU 데이터 처리
"""

import numpy as np
import time
from typing import Tuple, Dict
from sensor_msgs.msg import LaserScan, NavSatFix, Imu

# UTM 변환을 위한 간단한 구현 (utm 모듈 대신)
def simple_utm_conversion(lat, lon, ref_lat, ref_lon):
    """간단한 UTM 변환 (정확도는 떨어지지만 기본적인 변환)"""
    # 위도/경도를 미터 단위로 근사 변환
    # 1도 ≈ 111,320m (위도), 1도 ≈ 111,320 * cos(위도) m (경도)
    lat_m = (lat - ref_lat) * 111320.0
    lon_m = (lon - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    return lon_m, lat_m  # Easting, Northing

class GPSTransformer:
    """GPS 데이터를 UTM 좌표로 변환하는 클래스"""
    
    def __init__(self, ref_lat=-33.8568, ref_lon=151.2153):  # Sydney Regatta 기준점
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_easting, self.ref_northing = simple_utm_conversion(ref_lat, ref_lon, ref_lat, ref_lon)
        self.gps_data = None
        
        # 첫 번째 GPS 값을 기준점으로 설정
        self.first_gps_set = False
        self.first_utm_x = None
        self.first_utm_y = None
    
    def process_gps_data(self, msg: NavSatFix) -> Dict:
        """GPS 데이터를 UTM 좌표로 변환 (첫 번째 값을 기준점으로 설정)"""
        if msg.latitude == 0.0 or msg.longitude == 0.0:
            return None
        
        # 간단한 UTM 변환
        utm_easting, utm_northing = simple_utm_conversion(msg.latitude, msg.longitude, self.ref_lat, self.ref_lon)
        
        # 첫 번째 GPS 값을 기준점으로 설정
        if not self.first_gps_set:
            self.first_utm_x = utm_easting
            self.first_utm_y = utm_northing
            self.first_gps_set = True
            print(f"📍 기준점 설정: UTM X={self.first_utm_x:.2f}m, Y={self.first_utm_y:.2f}m")
        
        # 첫 번째 GPS 값을 기준으로 한 상대 좌표 계산
        relative_x = utm_easting - self.first_utm_x
        relative_y = utm_northing - self.first_utm_y
        
        self.gps_data = {
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'utm_x': relative_x,
            'utm_y': relative_y,
            'altitude': msg.altitude,
            'timestamp': time.time()
        }
        return self.gps_data

class LiDARProcessor:
    """LiDAR 데이터 전처리 클래스"""
    
    def __init__(self, max_range: float = 100.0, min_range: float = 0.0):
        self.max_range = max_range
        self.min_range = min_range
        self.lidar_data = {'ranges': np.array([]), 'angles': np.array([])}
    
    def process_lidar_data(self, msg: LaserScan) -> Dict:
        """LiDAR 데이터 전처리"""
        ranges = np.array(msg.ranges)
        
        # 유효한 범위만 필터링 (더 관대한 조건)
        valid_mask = (np.isfinite(ranges) & 
                     (ranges >= self.min_range) & 
                     (ranges <= self.max_range) &
                     (ranges > 0.0))  # 0보다 큰 값만
        
        valid_ranges = ranges[valid_mask]
        valid_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))[valid_mask]
        
        # 노이즈 필터링 비활성화 (안정성을 위해)
        # filtered_ranges, filtered_angles = self._filter_noise(valid_ranges, valid_angles)
        filtered_ranges, filtered_angles = valid_ranges, valid_angles
        
        # 디버깅 정보 추가
        if len(filtered_ranges) < len(ranges) * 0.3:  # 30% 미만이면 경고
            print(f"⚠️ LiDAR 필터링 경고: {len(filtered_ranges)}/{len(ranges)} 포인트만 유효 ({len(filtered_ranges)/len(ranges)*100:.1f}%)")
        
        self.lidar_data = {
            'ranges': filtered_ranges,
            'angles': filtered_angles,
            'timestamp': time.time(),
            'raw_count': len(ranges),
            'valid_count': len(filtered_ranges),
            'filter_ratio': len(filtered_ranges) / len(ranges) if len(ranges) > 0 else 0
        }
        return self.lidar_data
    
    def _filter_noise(self, ranges: np.ndarray, angles: np.ndarray, 
                     noise_threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """노이즈 필터링"""
        if len(ranges) < 3:
            return ranges, angles
        
        try:
            # 이웃 포인트와의 거리 차이 계산
            diff = np.abs(np.diff(ranges))
            
            # 노이즈 임계값보다 큰 차이를 보이는 포인트 제거
            # diff 배열 크기는 ranges보다 1 작으므로, 첫 번째와 마지막 포인트는 유지
            noise_mask = np.concatenate([[True], diff < noise_threshold, [True]])
            
            # 배열 크기 확인
            if len(noise_mask) == len(ranges):
                return ranges[noise_mask], angles[noise_mask]
            else:
                # 크기가 맞지 않으면 원본 반환
                return ranges, angles
                
        except Exception as e:
            # 오류 발생 시 원본 반환
            print(f"노이즈 필터링 오류: {e}")
            return ranges, angles

    def lidar_to_cartesian(self, ranges: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LiDAR 데이터를 직교좌표계로 변환 (회전 방향 수정)"""
        # LiDAR 좌표계: 0도 = 전방, 시계방향 회전
        # 로봇 좌표계: X = 전방, Y = 좌측
        # 각도를 로봇 좌표계에 맞게 조정
        x = ranges * np.sin(-angles)  # 전방 방향 (X축)
        y = ranges * np.cos(angles)  # 좌측 방향 (Y축)
        return x, y

class IMUProcessor:
    """IMU 데이터 처리 클래스"""
    
    def __init__(self):
        self.imu_data = None
    
    def process_imu_data(self, msg: Imu) -> Dict:
        """IMU 데이터 처리"""
        # Quaternion을 Euler 각도로 변환
        yaw_rad = self.quaternion_to_yaw(msg.orientation)
        yaw_degrees = np.degrees(yaw_rad)
        
        self.imu_data = {
            'orientation': msg.orientation,
            'yaw_rad': yaw_rad,
            'yaw_degrees': yaw_degrees,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration,
            'timestamp': time.time()
        }
        return self.imu_data
    
    def quaternion_to_yaw(self, orientation) -> float:
        """Quaternion을 Yaw 각도로 변환 (라디안)"""
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

class SensorDataManager:
    """센서 데이터 통합 관리 클래스"""
    
    def __init__(self):
        self.gps_transformer = GPSTransformer()
        self.lidar_processor = LiDARProcessor()
        self.imu_processor = IMUProcessor()
        
        # LiDAR 직교좌표 저장
        self.lidar_cartesian_x = np.array([])
        self.lidar_cartesian_y = np.array([])
    
    def process_gps_data(self, msg: NavSatFix) -> Dict:
        """GPS 데이터 처리"""
        return self.gps_transformer.process_gps_data(msg)
    
    def process_lidar_data(self, msg: LaserScan) -> Dict:
        """LiDAR 데이터 처리"""
        lidar_data = self.lidar_processor.process_lidar_data(msg)
        
        # 직교좌표 변환
        if len(lidar_data['ranges']) > 0:
            self.lidar_cartesian_x, self.lidar_cartesian_y = self.lidar_processor.lidar_to_cartesian(
                lidar_data['ranges'], lidar_data['angles']
            )
        
        return lidar_data
    
    def process_imu_data(self, msg: Imu) -> Dict:
        """IMU 데이터 처리"""
        return self.imu_processor.process_imu_data(msg)
    
    def get_lidar_cartesian(self) -> Tuple[np.ndarray, np.ndarray]:
        """LiDAR 직교좌표 데이터 반환"""
        return self.lidar_cartesian_x, self.lidar_cartesian_y
