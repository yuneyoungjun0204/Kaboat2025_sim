#!/usr/bin/env python3
"""
VRX ONNX 모델 기반 선박 제어 시스템 v6
- Unity NED 좌표계에서 학습된 모델을 Gazebo ENU 환경에서 사용
- 좌표계 변환 모듈화 (Unity NED ↔ Gazebo ENU ↔ Body-fixed)
- LiDAR 데이터 좌표계 변환 포함
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point
import time
from collections import deque
from utils import SensorDataManager

class CoordinateTransformer:
    """좌표계 변환 클래스"""
    
    @staticmethod
    def unity_ned_to_gazebo_enu(unity_pos):
        """
        Unity NED → Gazebo ENU 좌표계 변환
        Unity: X=동서(Easting), Y=남북(Northing), Z=아래(Depth)
        Gazebo: X=동(East), Y=북(North), Z=위(Up)
        """
        if len(unity_pos) >= 2:
            # Unity NED (X=동서, Y=남북) → Gazebo ENU (X=동, Y=북)
            gazebo_x = unity_pos[0]   # 동서 → 동
            gazebo_y = unity_pos[1]   # 남북 → 북
            return np.array([gazebo_x, gazebo_y])
        return np.array([0.0, 0.0])
    
    @staticmethod
    def gazebo_enu_to_unity_ned(gazebo_pos):
        """
        Gazebo ENU → Unity NED 좌표계 변환
        """
        if len(gazebo_pos) >= 2:
            # Gazebo ENU (X=동, Y=북) → Unity NED (X=동서, Y=남북)
            unity_x = gazebo_pos[0]   # 동 → 동서
            unity_y = gazebo_pos[1]   # 북 → 남북
            return np.array([unity_x, unity_y])
        return np.array([0.0, 0.0])
    
    @staticmethod
    def unity_heading_to_gazebo_heading(unity_heading_deg):
        """
        Unity 헤딩 → Gazebo 헤딩 변환
        Unity: 0도=서쪽, 90도=북쪽, 180도=동쪽, 270도=남쪽
        Gazebo: 0도=동쪽, 90도=북쪽, 180도=서쪽, 270도=남쪽
        """
        # Unity → Gazebo: 180도 회전
        gazebo_heading = unity_heading_deg + 180.0
        return CoordinateTransformer.normalize_angle_0_360(gazebo_heading)
    
    @staticmethod
    def gazebo_heading_to_unity_heading(gazebo_heading_deg):
        """
        Gazebo 헤딩 → Unity 헤딩 변환
        """
        # Gazebo → Unity: -180도 회전
        unity_heading = gazebo_heading_deg - 180.0
        return CoordinateTransformer.normalize_angle_0_360(unity_heading)
    
    @staticmethod
    def body_fixed_lidar_to_unity_ned(lidar_dist, lidar_angle_deg, robot_heading_deg, robot_pos):
        """
        Body-fixed LiDAR → Unity NED 좌표계 변환
        Body-fixed: 로봇 기준 상대 좌표
        Unity NED: 절대 좌표계
        """
        # 1. Body-fixed LiDAR를 절대 각도로 변환
        absolute_angle_deg = lidar_angle_deg + robot_heading_deg
        
        # 2. Unity NED 좌표계로 변환 (Unity 헤딩 사용)
        unity_heading = CoordinateTransformer.gazebo_heading_to_unity_heading(robot_heading_deg)
        unity_angle_deg = lidar_angle_deg + unity_heading
        
        # 3. Unity NED 좌표계에서의 위치 계산
        unity_angle_rad = np.radians(unity_angle_deg)
        unity_x = robot_pos[0] + lidar_dist * np.cos(unity_angle_rad)
        unity_y = robot_pos[1] + lidar_dist * np.sin(unity_angle_rad)
        
        return np.array([unity_x, unity_y])
    
    @staticmethod
    def normalize_angle_0_360(angle_deg):
        """각도를 0~360도 범위로 정규화"""
        while angle_deg < 0:
            angle_deg += 360.0
        while angle_deg >= 360.0:
            angle_deg -= 360.0
        return angle_deg
    
    @staticmethod
    def normalize_angle_pi_pi(angle_rad):
        """각도를 -π ~ π 범위로 정규화"""
        while angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        while angle_rad < -np.pi:
            angle_rad += 2 * np.pi
        return angle_rad

class VRXONNXControllerV6(Node):
    """VRX ONNX 모델 기반 제어 노드 v6 - 좌표계 변환 모듈화"""
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v6')
        
        # 좌표계 변환기 초기화
        self.coord_transformer = CoordinateTransformer()
        
        # ONNX 모델 로드
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-12499862.onnx'
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.get_logger().info(f"ONNX Model Loaded: {self.model_path}")

        # 센서 데이터 관리자
        self.sensor_manager = SensorDataManager()
        
        # ROS2 서브스크라이버
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan', self.lidar_callback, 10)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        
        # 웨이포인트 서브스크라이버 (robot_visualizer에서 받음)
        self.waypoint_sub = self.create_subscription(
            Point, 
            '/vrx/waypoint', 
            self.waypoint_callback, 
            10
        )
        
        # ROS2 퍼블리셔 (스러스터 제어)
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        # Unity 관찰값 구조에 맞는 변수들 (Unity NED 좌표계 기준)
        self.lidar_distances = np.zeros(201, dtype=np.float32)  # LiDAR 거리 (201개)
        self.lidar_distances_360 = np.zeros(360, dtype=np.float32)  # LiDAR 거리 (360도 전체)
        self.agent_heading = 0.0                                # 에이전트 헤딩 (Unity 좌표계)
        self.angular_velocity_y = 0.0                           # IMU 각속도 (Y축)
        
        # 위치 관련 변수들 (Unity NED 좌표계)
        self.agent_position_unity = np.zeros(2, dtype=np.float32)     # Unity NED 좌표계 위치
        self.agent_position_gazebo = np.zeros(2, dtype=np.float32)    # Gazebo ENU 좌표계 위치
        self.current_target_position = np.zeros(2, dtype=np.float32)  # 현재 웨이포인트 (Unity NED)
        self.previous_target_position = np.zeros(2, dtype=np.float32) # 이전 웨이포인트 (Unity NED)
        self.next_target_position = np.zeros(2, dtype=np.float32)     # 다음 웨이포인트 (Unity NED)
        
        # 이전 명령 저장
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        # 웨이포인트 관리
        self.waypoints = []  # 웨이포인트 리스트 (Unity NED 좌표계)
        self.current_target_index = 0
        self.target_position = None
        
        # 각속도 추적용 변수 (Unity와 동일)
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.current_angular_acceleration = np.zeros(3)
        
        # 스케일링 (TurtleBot과 동일)
        self.v_scale = 0.4   # linear velocity scale
        self.w_scale = -0.6  # angular velocity scale

        # 최근 스러스터 명령 저장용 변수
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        # 기준점 설정 상태
        self.reference_point_set = False
        
        # 웨이포인트 도달 상태
        self.waypoint_reached = False
        
        # v5 추가: 직접 경로 모드 플래그
        self.use_direct_navigation = False
        
        # v5 모드 정보 퍼블리셔
        self.mode_pub = self.create_publisher(String, '/vrx/current_mode', 10)
        
        # goal_check 영역 정보 퍼블리셔
        self.goal_check_pub = self.create_publisher(Float64MultiArray, '/vrx/goal_check_areas', 10)

        # 10Hz 주기로 스러스터 제어
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('🚢 VRX ONNX Controller v6 시작! (좌표계 변환 모듈화)')
        self.get_logger().info('📍 웨이포인트를 robot_visualizer에서 클릭하여 설정하세요')

    def waypoint_callback(self, msg):
        """웨이포인트 콜백 - robot_visualizer에서 클릭한 점을 받음 (Gazebo ENU 좌표계)"""
        # Gazebo ENU → Unity NED 좌표계 변환
        gazebo_waypoint = np.array([msg.x, msg.y])
        unity_waypoint = self.coord_transformer.gazebo_enu_to_unity_ned(gazebo_waypoint)
        
        self.waypoints.append(unity_waypoint.tolist())
        self.current_target_index = len(self.waypoints) - 1
        self.target_position = unity_waypoint
        self.waypoint_reached = False
        
        self.get_logger().info(f'🎯 새 웨이포인트 추가: Gazebo({msg.x:.1f}, {msg.y:.1f}) → Unity({unity_waypoint[0]:.1f}, {unity_waypoint[1]:.1f}) - 총 {len(self.waypoints)}개')

    def gps_callback(self, msg):
        """GPS 데이터 콜백 - 로봇 위치 업데이트 (좌표계 변환 포함)"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            # Gazebo ENU 좌표계 (UTM 좌표)
            self.agent_position_gazebo = np.array([
                gps_data['utm_x'],  # Gazebo X (동)
                gps_data['utm_y']   # Gazebo Y (북)
            ], dtype=np.float32)
            
            # Unity NED 좌표계로 변환
            self.agent_position_unity = self.coord_transformer.gazebo_enu_to_unity_ned(self.agent_position_gazebo)
            
            # 첫 번째 GPS 데이터 기준점 설정 확인
            if not self.reference_point_set:
                self.reference_point_set = True
                self.get_logger().info(f'📍 기준점 설정 완료: Gazebo({self.agent_position_gazebo[0]:.1f}, {self.agent_position_gazebo[1]:.1f}) → Unity({self.agent_position_unity[0]:.1f}, {self.agent_position_unity[1]:.1f})')

    def imu_callback(self, msg):
        """IMU 데이터 콜백 - 헤딩과 각속도 업데이트 (좌표계 변환 포함)"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        
        # Gazebo 헤딩 (ENU 좌표계) - 0~360도
        gazebo_heading = imu_data['yaw_degrees']
        if gazebo_heading < 0:
            gazebo_heading += 360.0
            
        # Unity 헤딩으로 변환 (NED 좌표계)
        self.agent_heading = self.coord_transformer.gazebo_heading_to_unity_heading(gazebo_heading)
            
        # 각속도 계산 및 업데이트 (Unity와 동일한 로직)
        current_time = time.time()
        current_angular_velocity = np.array([
            msg.angular_velocity.x,  # Roll (X축 회전)
            msg.angular_velocity.y,  # Pitch (Y축 회전)
            msg.angular_velocity.z   # Yaw (Z축 회전) - 헤딩 방향
        ])
        
        # 각가속도 계산
        if self.last_angular_velocity_update_time > 0:
            delta_time = current_time - self.last_angular_velocity_update_time
            if delta_time > 0:
                self.current_angular_acceleration = (current_angular_velocity - self.previous_angular_velocity) / delta_time
        
        # 이전 값 업데이트
        self.previous_angular_velocity = current_angular_velocity
        self.last_angular_velocity_update_time = current_time
        
        # Z축 각속도 (헤딩 방향 - Yaw)
        self.angular_velocity_y = current_angular_velocity[2]  # Z축이 헤딩 방향

    def check_obstacles_in_forward_path(self):
        """v5 추가: 전방 60도 범위에 40m 미만 장애물이 있는지 확인"""
        # 전방 60도 범위 (중앙 ±30도)에 해당하는 LiDAR 인덱스 계산
        # LiDAR: -100도 ~ 100도, 201개 (중앙이 인덱스 100)
        center_index = 100  # 0도 (전방) 인덱스
        range_degrees = 60.0  # 60도 범위
        half_range = range_degrees / 2.0  # ±30도
        
        # 60도 범위에 해당하는 인덱스 범위 계산
        # 각도당 인덱스: 200도 / 200인덱스 = 1도당 1인덱스
        half_range_indices = int(half_range)  # ±30도 → ±30인덱스
        start_index = max(0, center_index - half_range_indices)
        end_index = min(201, center_index + half_range_indices + 1)
        
        # 해당 범위에서 40m 미만 장애물 검사
        obstacle_threshold = 40.0  # 40m 임계값
        has_obstacles = False
        
        for i in range(start_index, end_index):
            if self.lidar_distances[i] < obstacle_threshold:
                has_obstacles = True
                self.get_logger().debug(f'🚧 전방 60도 범위에 장애물 감지: 인덱스={i}, 거리={self.lidar_distances[i]:.1f}m')
                break
        
        return has_obstacles

    def goal_check(self, goal_distance, goal_psi):
        """
        목적지까지 경로에 장애물이 있는지 판단하는 함수 (Unity NED 좌표계 기준)
        """
        l = goal_distance
        boat_width = 10  # 배 폭 (미터)
        
        # Unity NED 좌표계에서 목표 방향 벡터 계산
        dx = np.cos(np.radians(goal_psi))
        dy = np.sin(np.radians(goal_psi))
        
        # 직사각형의 4개 모서리 점들 계산 (Unity NED 좌표계)
        half_width = boat_width / 2.0
        
        # 수직 방향 벡터 (목표 방향에 수직)
        perp_x = -dy  # 수직 방향
        perp_y = dx   # 수직 방향
        
        # 직사각형의 4개 모서리 (시계방향) - Unity NED 좌표계
        corners = [
            # 왼쪽 앞 모서리 (목표 방향 + 왼쪽)
            [self.agent_position_unity[0] + l * dx - half_width * perp_x, 
             self.agent_position_unity[1] + l * dy - half_width * perp_y],
            # 오른쪽 앞 모서리 (목표 방향 + 오른쪽)
            [self.agent_position_unity[0] + l * dx + half_width * perp_x,
             self.agent_position_unity[1] + l * dy + half_width * perp_y],
            # 오른쪽 뒤 모서리 (현재 위치 + 오른쪽)
            [self.agent_position_unity[0] + half_width * perp_x,
             self.agent_position_unity[1] + half_width * perp_y],
            # 왼쪽 뒤 모서리 (현재 위치 + 왼쪽)
            [self.agent_position_unity[0] - half_width * perp_x,
             self.agent_position_unity[1] - half_width * perp_y]
        ]
        
        # 직사각형 영역 정보 저장 [type, x1, y1, x2, y2, x3, y3, x4, y4]
        area_info = [4.0]  # type 4 = 직사각형
        for corner in corners:
            area_info.extend([float(corner[0]), float(corner[1])])
        
        # 직사각형 경로 내의 LiDAR 포인트들 체크 (360도 전체 사용)
        isAble = True
        for i, lidar_dist in enumerate(self.lidar_distances_360):
            if lidar_dist >= 50.0 or lidar_dist <= 0.0:  # 유효하지 않은 데이터 스킵
                continue
                
            # LiDAR 각도 계산 (360도 전체: 0도 ~ 360도)
            lidar_angle = i  # 0도부터 359도까지
            lidar_angle_rad = np.radians(lidar_angle + self.agent_heading)  # Unity 헤딩 기준
            
            # LiDAR 포인트의 실제 위치 (Unity NED 좌표계)
            lidar_x = self.agent_position_unity[0] + lidar_dist * np.cos(lidar_angle_rad)
            lidar_y = self.agent_position_unity[1] + lidar_dist * np.sin(lidar_angle_rad)
            
            # 직사각형 내부에 있는지 체크 (점-다각형 포함 테스트)
            if self.point_in_polygon(lidar_x, lidar_y, corners):
                isAble = False
                self.get_logger().debug(f'🚧 직사각형 경로 내 장애물: 거리={lidar_dist:.1f}m, 각도={lidar_angle:.1f}°')
                break
        
        # 영역 정보를 ROS 메시지로 발행 (Gazebo ENU 좌표계로 변환하여 발행)
        if len(area_info) > 1:
            # Unity NED → Gazebo ENU 좌표계 변환
            gazebo_area_info = [area_info[0]]  # type은 그대로
            for i in range(1, len(area_info), 2):
                unity_pos = np.array([area_info[i], area_info[i+1]])
                gazebo_pos = self.coord_transformer.unity_ned_to_gazebo_enu(unity_pos)
                gazebo_area_info.extend([float(gazebo_pos[0]), float(gazebo_pos[1])])
            
            area_msg = Float64MultiArray()
            area_msg.data = gazebo_area_info
            self.goal_check_pub.publish(area_msg)
        
        return isAble
    
    def normalize_angle(self, angle):
        """각도를 0-359도 범위로 정규화"""
        return self.coord_transformer.normalize_angle_0_360(angle)
    
    def point_in_polygon(self, x, y, polygon):
        """점이 다각형 내부에 있는지 확인 (Ray casting algorithm)"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def calculate_direct_heading(self, current_pos, target_pos):
        """v5 추가: atan을 이용한 직접 헤딩 계산 (Unity NED 좌표계)"""
        # Unity NED 좌표계: X=동서(Easting), Y=남북(Northing)
        dx = target_pos[0] - current_pos[0]  # X 방향 차이 (동서)
        dy = target_pos[1] - current_pos[1]  # Y 방향 차이 (남북)
        
        # Unity 좌표계에서 목표 방향 계산 (라디안)
        # Unity: 0도=서쪽, 90도=북쪽, 180도=동쪽, 270도=남쪽
        target_heading_rad = np.arctan2(dx, dy)  # Unity 좌표계에 맞게
        
        # 현재 헤딩과의 차이 계산 (라디안)
        current_heading_rad = np.radians(self.agent_heading)
        heading_diff_rad = target_heading_rad - current_heading_rad
        
        # -π ~ π 범위로 정규화
        heading_diff_rad = self.coord_transformer.normalize_angle_pi_pi(heading_diff_rad)
            
        return heading_diff_rad

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 - 좌표계 변환 포함"""
        lidar_data = self.sensor_manager.process_lidar_data(msg)
        
        # LiDAR 데이터를 -100도~100도 범위로 필터링하고 201개로 리샘플링
        lidar_ranges = lidar_data['ranges']
        lidar_angles = lidar_data['angles']
        
        # -100도 ~ 100도 범위의 LiDAR 데이터 필터링
        angle_mask = (lidar_angles >= np.radians(-100)) & (lidar_angles <= np.radians(100))
        filtered_ranges = lidar_ranges[angle_mask]
        filtered_angles = lidar_angles[angle_mask]
        
        # 필터링된 데이터가 비어있는지 확인
        if len(filtered_ranges) == 0 or len(filtered_angles) == 0:
            # 필터링된 데이터가 없으면 기본값으로 채움
            self.lidar_distances = np.full(201, 50.0, dtype=np.float32)
            self.get_logger().warn('LiDAR 필터링 데이터 없음 - 기본값 사용')
            return
        
        # 201개로 리샘플링
        target_angles = np.linspace(np.radians(-100), np.radians(100), 201)
        
        # 중복된 각도 제거 및 정렬
        if len(filtered_angles) > 1:
            # 각도를 정렬하고 중복 제거
            sort_indices = np.argsort(filtered_angles)
            filtered_angles = filtered_angles[sort_indices]
            filtered_ranges = filtered_ranges[sort_indices]
            
            # 중복 각도 제거 (np.interp가 요구하는 조건)
            unique_mask = np.concatenate(([True], np.diff(filtered_angles) != 0))
            filtered_angles = filtered_angles[unique_mask]
            filtered_ranges = filtered_ranges[unique_mask]
        
        # 리샘플링 실행
        if len(filtered_ranges) >= 2:
            resampled_ranges = np.interp(target_angles, filtered_angles, filtered_ranges)
        else:
            # 데이터가 부족하면 기본값 사용
            resampled_ranges = np.full(201, 50.0, dtype=np.float32)
        
        # 무한대 값 방지 (Unity와 동일한 로직)
        for i in range(len(resampled_ranges)):
            if np.isinf(resampled_ranges[i]) or np.isnan(resampled_ranges[i]) or resampled_ranges[i]>=50.0:
                resampled_ranges[i] = 50.0  # 최대 거리로 설정
        
        self.lidar_distances = resampled_ranges.astype(np.float32)
        
        # 360도 LiDAR 데이터도 저장 (goal_check용)
        if len(msg.ranges) >= 360:
            # 360도 데이터가 충분하면 그대로 사용
            self.lidar_distances_360 = np.array(msg.ranges[:360], dtype=np.float32)
        else:
            # 360도 데이터가 부족하면 기본값으로 채움
            self.lidar_distances_360 = np.full(360, 50.0, dtype=np.float32)
        
        # 제어 실행
        self.control_vrx()

    def control_vrx(self):
        """Unity 관찰값 구조 기반 제어 및 ONNX 모델 실행 + v5 직접 경로 모드 (좌표계 변환 포함)"""
        # 웨이포인트가 없으면 모터 정지
        if self.target_position is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            self.get_logger().debug('웨이포인트 없음 - 모터 정지')
            
            # 모드 정보 발행
            mode_msg = String()
            mode_msg.data = "STOP"
            self.mode_pub.publish(mode_msg)
            return

        # 웨이포인트 도달 확인 (Unity NED 좌표계 기준)
        current_pos_unity = self.agent_position_unity
        distance = np.sqrt((current_pos_unity[0] - self.target_position[0])**2 + (current_pos_unity[1] - self.target_position[1])**2)
        
        # 디버깅 로그 추가
        self.get_logger().debug(f'현재위치 Unity: ({current_pos_unity[0]:.1f}, {current_pos_unity[1]:.1f}) | 목표 Unity: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) | 거리: {distance:.1f}m')
        
        if distance < 15.0:  # 15m 이내 도달
            if not self.waypoint_reached:
                self.waypoint_reached = True
                self.get_logger().info(f'🎯 웨이포인트 도달! 거리: {distance:.1f}m')
                # 다음 웨이포인트로 이동
                self.current_target_index += 1
                if self.current_target_index < len(self.waypoints):
                    # 다음 웨이포인트가 있으면 이동
                    next_waypoint = self.waypoints[self.current_target_index]
                    self.target_position = np.array(next_waypoint, dtype=np.float32)
                    self.waypoint_reached = False
                    self.get_logger().info(f'🎯 다음 웨이포인트로 이동: Unity({next_waypoint[0]:.1f}, {next_waypoint[1]:.1f})')
                else:
                    # 모든 웨이포인트 완료
                    self.target_position = None
                    self.get_logger().info('🏁 모든 웨이포인트 완료! 정지합니다.')
            # 도달했으면 모터 정지
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            
            # 모드 정보 발행
            mode_msg = String()
            mode_msg.data = "REACHED"
            self.mode_pub.publish(mode_msg)
            return

        # v5 추가: 전방 60도 범위에 장애물이 있는지 확인 (기존 방식)
        has_obstacles_forward = self.check_obstacles_in_forward_path()
        
        # v5 추가: 목적지까지 경로에 장애물이 있는지 확인 (Unity NED 좌표계 기준)
        dx = self.target_position[0] - current_pos_unity[0]  # X 방향 차이 (동서)
        dy = self.target_position[1] - current_pos_unity[1]  # Y 방향 차이 (남북)
        goal_psi = np.degrees(np.arctan2(dx, dy))  # Unity NED 좌표계에서 목적지 방향 각도
        goal_psi = self.normalize_angle(int(goal_psi))
        has_obstacles_path = not self.goal_check(distance, goal_psi)
        
        # v5 추가: 두 방식 모두 장애물이 없으면 직접 경로 모드 사용
        if not has_obstacles_forward and not has_obstacles_path and distance > 20.0:  # 20m 이상 떨어져 있을 때만
            self.use_direct_navigation = True
            self.get_logger().info('🛤️ 직접 경로 모드 활성화 (경로에 장애물 없음)')
            
            # 모드 정보 발행
            mode_msg = String()
            mode_msg.data = "DIRECT_CLEAR"
            self.mode_pub.publish(mode_msg)
            
            # atan을 이용한 직접 헤딩 계산 (Unity NED 좌표계)
            heading_diff_rad = self.calculate_direct_heading(current_pos_unity, self.target_position)
            
            # 선형 속도 최대 고정
            linear_velocity = 0.3  # 최대 속도 고정
            
            # 각속도 (헤딩 차이에 비례)
            angular_velocity = np.clip(heading_diff_rad / np.pi, -0.3, 0.3)
            
            # 스러스터 명령으로 변환
            self.left_thrust, self.right_thrust = self.calculate_thruster_commands(linear_velocity, angular_velocity)
            
            # trajectory_viz.py로 출력값 전송
            if not hasattr(self, 'control_output_pub'):
                self.control_output_pub = self.create_publisher(Float64MultiArray, '/vrx/control_output', 10)
            
            output_msg = Float64MultiArray()
            output_msg.data = [linear_velocity, angular_velocity]
            self.control_output_pub.publish(output_msg)
            
            self.get_logger().info(
                f"직접경로: 거리={distance:.1f}m | "
                f"헤딩차이={np.degrees(heading_diff_rad):.1f}° | "
                f"제어값: Linear={linear_velocity:.3f}, Angular={angular_velocity:.3f} | "
                f"스러스터: L={self.left_thrust:.1f}, R={self.right_thrust:.1f}"
            )
            return
        else:
            self.use_direct_navigation = False
            if has_obstacles_forward:
                self.get_logger().info('🚧 ONNX 모델 모드 (전방 60도 범위 장애물 감지)')
            if has_obstacles_path:
                self.get_logger().info('🚧 ONNX 모델 모드 (목적지 경로에 장애물 감지)')
            
            # 모드 정보 발행
            mode_msg = String()
            if has_obstacles_forward and has_obstacles_path:
                mode_msg.data = "ONNX_BOTH"
            elif has_obstacles_forward:
                mode_msg.data = "ONNX_FORWARD"
            elif has_obstacles_path:
                mode_msg.data = "ONNX_PATH"
            else:
                mode_msg.data = "ONNX_CLOSE"
            self.mode_pub.publish(mode_msg)

        # ONNX 모델 모드 (장애물이 있는 경우 또는 가까운 거리)
        # 웨이포인트 위치 업데이트
        self.update_waypoint_positions()

        # Unity 관찰값 구조에 맞게 입력 벡터 생성 (Unity NED 좌표계 기준)
        observation_values = []
        
        # 1. LiDAR 거리 (201개)
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))
        
        # 2. 에이전트 헤딩 (1개) - 무한대 값 방지
        if np.isinf(self.agent_heading) or np.isnan(self.agent_heading):
            self.agent_heading = 0.0
        observation_values.append(float(self.agent_heading))
        
        # 3. IMU 각속도 Y축 (1개) - 무한대 값 방지
        if np.isinf(self.angular_velocity_y) or np.isnan(self.angular_velocity_y):
            self.angular_velocity_y = 0.0
        observation_values.append(float(self.angular_velocity_y))
        
        # 4. 현재 에이전트 위치 (2개: X, Y) - Unity NED 좌표계 - 무한대 값 방지
        for i in range(2):
            val = float(self.agent_position_unity[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 5. 현재 웨이포인트 위치 (2개: X, Y) - Unity NED 좌표계 - 무한대 값 방지
        for i in range(2):
            val = float(self.current_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 6. 이전 웨이포인트 위치 (2개: X, Y) - Unity NED 좌표계 - 무한대 값 방지
        for i in range(2):
            val = float(self.previous_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 7. 다음 웨이포인트 위치 (2개: X, Y) - Unity NED 좌표계 - 무한대 값 방지
        for i in range(2):
            val = float(self.next_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 8. 이전 명령 (2개: moment_input, force_input)
        observation_values.append(float(self.previous_moment_input))
        observation_values.append(float(self.previous_force_input))

        # 관찰값을 numpy 배열로 변환 (Unity와 동일한 크기: 211개)
        observation_array = np.array(observation_values, dtype=np.float32)
        
        # Stacked 입력 생성 (2번의 211개 데이터 = 422개)
        stacked_input = np.concatenate([
            observation_array,  # 첫 번째 211개 데이터
            observation_array   # 두 번째 211개 데이터 (단순화)
        ]).reshape(1, 422)

        # ONNX 모델 추론
        outputs = self.session.run(None, {self.input_name: stacked_input})
        
        # TurtleBot 스타일 출력 처리
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), 0.02)
            angular_velocity = max(min(outputs[2][0][0] * self.w_scale, 1.0), -1.0)
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        # 이전 명령 업데이트 (Unity와 동일)
        self.previous_moment_input = angular_velocity
        self.previous_force_input = linear_velocity

        # 스러스터 명령으로 변환
        self.left_thrust, self.right_thrust = self.calculate_thruster_commands(linear_velocity, angular_velocity)

        # trajectory_viz.py로 출력값 전송을 위한 퍼블리셔
        if not hasattr(self, 'control_output_pub'):
            self.control_output_pub = self.create_publisher(Float64MultiArray, '/vrx/control_output', 10)
        
        # 제어 출력값 발행 [linear_velocity, angular_velocity]
        output_msg = Float64MultiArray()
        output_msg.data = [linear_velocity, angular_velocity]
        self.control_output_pub.publish(output_msg)

        self.get_logger().info(
            f"ONNX모델: 위치 Unity: ({current_pos_unity[0]:.1f}, {current_pos_unity[1]:.1f}) | "
            f"웨이포인트 Unity: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) | "
            f"거리: {distance:.1f}m | "
            f"제어값: Linear={linear_velocity:.3f}, Angular={angular_velocity:.3f} | "
            f"스러스터: L={self.left_thrust:.1f}, R={self.right_thrust:.1f}"
        )

    def update_waypoint_positions(self):
        """웨이포인트 위치 업데이트 (Unity와 동일한 로직)"""
        if len(self.waypoints) == 0:
            # 웨이포인트가 없는 경우
            self.current_target_position = np.zeros(2)
            self.previous_target_position = np.zeros(2)
            self.next_target_position = np.zeros(2)
            return
        
        # 현재 웨이포인트 위치
        if self.current_target_index < len(self.waypoints):
            self.current_target_position = np.array(self.waypoints[self.current_target_index])
        else:
            self.current_target_position = np.zeros(2)
        
        # 이전 웨이포인트 위치
        if self.current_target_index > 0:
            self.previous_target_position = np.array(self.waypoints[self.current_target_index - 1])
        else:
            self.previous_target_position = np.zeros(2)  # 첫 번째 웨이포인트일 경우 (0,0)
        
        # 다음 웨이포인트 위치
        if self.current_target_index + 1 < len(self.waypoints):
            self.next_target_position = np.array(self.waypoints[self.current_target_index + 1])
        else:
            # 마지막 웨이포인트일 경우 현재 웨이포인트와 동일한 값
            self.next_target_position = self.current_target_position.copy()

    def calculate_thruster_commands(self, linear_velocity, angular_velocity):
        """선형/각속도를 스러스터 명령으로 변환"""
        # TurtleBot의 linear/angular velocity를 VRX 스러스터로 변환
        forward_thrust = linear_velocity * 1000.0  # 전진 명령
        turn_thrust = angular_velocity * 1000.0    # 선회 명령
        
        # 좌우 스러스터 계산
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        
        # 범위 제한
        left_thrust = np.clip(left_thrust, -1000.0, 1000.0)
        right_thrust = np.clip(right_thrust, -1000.0, 1000.0)
        
        return left_thrust, right_thrust

    def timer_callback(self):
        """주기적으로 스러스터 명령 발행"""
        left_msg = Float64()
        left_msg.data = self.left_thrust
        self.left_thrust_pub.publish(left_msg)
        
        right_msg = Float64()
        right_msg.data = self.right_thrust
        self.right_thrust_pub.publish(right_msg)

    def destroy_node(self):
        """노드 종료 시 정리"""
        # 스러스터 정지
        left_msg = Float64()
        left_msg.data = 0.0
        self.left_thrust_pub.publish(left_msg)
        
        right_msg = Float64()
        right_msg.data = 0.0
        self.right_thrust_pub.publish(right_msg)
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VRXONNXControllerV6()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
