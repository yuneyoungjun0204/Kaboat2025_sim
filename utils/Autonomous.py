#!/usr/bin/env python3
"""
Autonomous Navigation Module for VRX
- 좌표계 변환 (Unity NED ↔ Gazebo ENU ↔ Body-fixed)
- ONNX 모델 기반 제어
- 직접 경로 모드
- 장애물 회피
"""

import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point
import time
from collections import deque

class CoordinateTransformer:
    """좌표계 변환 클래스 - 회전 방향 및 X축 기준 차이 고려"""
    
    @staticmethod
    def unity_ned_to_gazebo_enu(unity_pos):
        """
        Unity NED → Gazebo ENU 좌표계 변환
        Unity NED: X=동서(Easting), Y=남북(Northing), Z=아래(Depth)
        Gazebo ENU: X=동(East), Y=북(North), Z=위(Up)
        
        좌표계 차이:
        - Unity NED: X축 기준이 서쪽(0도), 시계방향 양수
        - Gazebo ENU: X축 기준이 동쪽(0도), 반시계방향 양수
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
        
        좌표계 차이:
        - Unity NED: 0도=서쪽, 90도=북쪽, 180도=동쪽, 270도=남쪽 (시계방향 양수)
        - Gazebo ENU: 0도=동쪽, 90도=북쪽, 180도=서쪽, 270도=남쪽 (반시계방향 양수)
        
        X축 기준 차이: Unity 서쪽(0도) ↔ Gazebo 동쪽(0도)
        회전 방향 차이: Unity 시계방향(+) ↔ Gazebo 반시계방향(+)
        """
        # Unity → Gazebo: 180도 회전 + 좌표계 차이
        gazebo_heading = unity_heading_deg + 180.0
        return CoordinateTransformer.normalize_angle_0_360(gazebo_heading)
    
    @staticmethod
    def gazebo_heading_to_unity_heading(gazebo_heading_deg):
        """
        Gazebo 헤딩 → Unity 헤딩 변환
        """
        # Gazebo → Unity: -180도 회전 + 좌표계 차이
        unity_heading = gazebo_heading_deg - 180.0
        return CoordinateTransformer.normalize_angle_0_360(unity_heading)
    
    @staticmethod
    def unity_angular_velocity_to_gazebo_angular_velocity(unity_angular_vel):
        """
        Unity 각속도 → Gazebo 각속도 변환
        
        회전 방향 차이:
        - Unity NED: 시계방향 회전이 양수 (+)
        - Gazebo ENU: 반시계방향 회전이 양수 (+)
        
        X축 기준은 동일하지만 회전 방향이 반대이므로 부호 반전
        """
        # 회전 방향이 반대이므로 부호 반전
        return -unity_angular_vel
    
    @staticmethod
    def gazebo_angular_velocity_to_unity_angular_velocity(gazebo_angular_vel):
        """
        Gazebo 각속도 → Unity 각속도 변환
        """
        # 회전 방향이 반대이므로 부호 반전
        return -gazebo_angular_vel
    
    @staticmethod
    def body_fixed_lidar_to_unity_ned(lidar_dist, lidar_angle_deg, robot_heading_deg, robot_pos):
        """
        Body-fixed LiDAR → Unity NED 좌표계 변환
        
        Body-fixed: 로봇 기준 상대 좌표, 시계방향 양수
        Unity NED: 절대 좌표계, 시계방향 양수
        
        X축 기준:
        - Body-fixed: 전방이 0도
        - Unity NED: 서쪽이 0도
        """
        # 1. Body-fixed LiDAR를 절대 각도로 변환
        # Body-fixed는 로봇 기준이므로 로봇 헤딩을 더함
        absolute_angle_deg = lidar_angle_deg + robot_heading_deg
        
        # 2. Unity NED 좌표계로 변환
        # 로봇 헤딩이 Gazebo ENU이므로 Unity NED로 변환
        unity_heading = CoordinateTransformer.gazebo_heading_to_unity_heading(robot_heading_deg)
        
        # 3. Unity NED 좌표계에서의 위치 계산
        # Unity NED: X=동서, Y=남북, 0도=서쪽
        unity_angle_rad = np.radians(absolute_angle_deg)
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

class ObstacleDetector:
    """장애물 감지 클래스"""
    
    def __init__(self, coord_transformer):
        self.coord_transformer = coord_transformer
    
    def check_obstacles_in_forward_path(self, lidar_distances, range_degrees=40.0, obstacle_threshold=25.0):
        """전방 범위에 장애물이 있는지 확인"""
        # 전방 범위 (중앙 ±range_degrees/2도)에 해당하는 LiDAR 인덱스 계산
        # LiDAR: -100도 ~ 100도, 201개 (중앙이 인덱스 100)
        center_index = 100  # 0도 (전방) 인덱스
        half_range = range_degrees / 2.0  # ±20도
        
        # 범위에 해당하는 인덱스 범위 계산
        # 각도당 인덱스: 200도 / 200인덱스 = 1도당 1인덱스
        half_range_indices = int(half_range)  # ±20도 → ±20인덱스
        start_index = max(0, center_index - half_range_indices)
        end_index = min(201, center_index + half_range_indices + 1)
        
        has_obstacles = False
        
        for i in range(start_index, end_index):
            if lidar_distances[i] < obstacle_threshold:
                has_obstacles = True
                break
        
        return has_obstacles
    
    def goal_check(self, goal_distance, goal_psi, agent_position_unity, lidar_distances_360, agent_heading, boat_width=5.0):
        """
        목적지까지 경로에 장애물이 있는지 판단하는 함수 (Unity NED 좌표계 기준)
        """
        l = goal_distance
        
        # Unity NED 좌표계에서 목표 방향 벡터 계산
        # Unity NED: 0도=서쪽, 90도=북쪽, 시계방향 양수
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
            [agent_position_unity[0] + l * dx - half_width * perp_x, 
             agent_position_unity[1] + l * dy - half_width * perp_y],
            # 오른쪽 앞 모서리 (목표 방향 + 오른쪽)
            [agent_position_unity[0] + l * dx + half_width * perp_x,
             agent_position_unity[1] + l * dy + half_width * perp_y],
            # 오른쪽 뒤 모서리 (현재 위치 + 오른쪽)
            [agent_position_unity[0] + half_width * perp_x,
             agent_position_unity[1] + half_width * perp_y],
            # 왼쪽 뒤 모서리 (현재 위치 + 왼쪽)
            [agent_position_unity[0] - half_width * perp_x,
             agent_position_unity[1] - half_width * perp_y]
        ]
        
        # 직사각형 영역 정보 저장 [type, x1, y1, x2, y2, x3, y3, x4, y4]
        area_info = [4.0]  # type 4 = 직사각형
        for corner in corners:
            area_info.extend([float(corner[0]), float(corner[1])])
        
        # 직사각형 경로 내의 LiDAR 포인트들 체크 (360도 전체 사용)
        isAble = True
        for i, lidar_dist in enumerate(lidar_distances_360):
            if lidar_dist >= 50.0 or lidar_dist <= 0.0:  # 유효하지 않은 데이터 스킵
                continue
                
            # LiDAR 각도 계산 (360도 전체: 0도 ~ 360도)
            # Body-fixed LiDAR → Unity NED 좌표계 변환
            lidar_angle = i  # 0도부터 359도까지 (Body-fixed 기준)
            
            # Body-fixed LiDAR를 Unity NED 절대 좌표로 변환
            # Body-fixed: 0도=전방, 시계방향 양수
            # Unity NED: 0도=서쪽, 시계방향 양수
            lidar_angle_rad = np.radians(lidar_angle + agent_heading)  # Unity 헤딩 기준
            
            # LiDAR 포인트의 실제 위치 (Unity NED 좌표계)
            lidar_x = agent_position_unity[0] + lidar_dist * np.cos(lidar_angle_rad)
            lidar_y = agent_position_unity[1] + lidar_dist * np.sin(lidar_angle_rad)
            
            # 직사각형 내부에 있는지 체크 (점-다각형 포함 테스트)
            if self.point_in_polygon(lidar_x, lidar_y, corners):
                isAble = False
                break
        
        return isAble, area_info
    
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

class DirectNavigationController:
    """직접 경로 제어 클래스"""
    
    def __init__(self, coord_transformer):
        self.coord_transformer = coord_transformer
    
    def calculate_direct_heading(self, current_pos, target_pos, agent_heading):
        """atan을 이용한 직접 헤딩 계산 (Unity NED 좌표계, 회전 방향 고려)"""
        # Unity NED 좌표계: X=동서(Easting), Y=남북(Northing)
        dx = target_pos[0] - current_pos[0]  # X 방향 차이 (동서)
        dy = target_pos[1] - current_pos[1]  # Y 방향 차이 (남북)
        
        # Unity 좌표계에서 목표 방향 계산 (라디안)
        # Unity NED: 0도=서쪽, 90도=북쪽, 180도=동쪽, 270도=남쪽, 시계방향 양수
        target_heading_rad = np.arctan2(dx, dy)  # Unity 좌표계에 맞게
        
        # 현재 헤딩과의 차이 계산 (라디안)
        current_heading_rad = np.radians(agent_heading)
        heading_diff_rad = target_heading_rad - current_heading_rad
        
        # -π ~ π 범위로 정규화 (회전 방향 고려)
        heading_diff_rad = self.coord_transformer.normalize_angle_pi_pi(heading_diff_rad)
            
        return heading_diff_rad

class ONNXModelController:
    """ONNX 모델 기반 제어 클래스"""
    
    def __init__(self, model_path, coord_transformer):
        self.coord_transformer = coord_transformer
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # 스케일링 (TurtleBot과 동일)
        self.v_scale = 0.4   # linear velocity scale
        self.w_scale = -0.6  # angular velocity scale
    
    def predict_control(self, observation_values):
        """ONNX 모델로 제어 명령 예측"""
        # 관찰값을 numpy 배열로 변환 (Unity와 동일한 크기: 211개)
        observation_array = np.array(observation_values, dtype=np.float32)
        
        # Stacked 입력 생성 (2번의 211개 데이터 = 422개)
        stacked_input = np.concatenate([
            observation_array,  # 첫 번째 211개 데이터
            observation_array   # 두 번째 211개 데이터 (단순화)
        ]).reshape(1, 426)

        # ONNX 모델 추론
        outputs = self.session.run(None, {self.input_name: stacked_input})
        
        # TurtleBot 스타일 출력 처리
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), 0.02)
            angular_velocity = max(min(outputs[2][0][0] * self.w_scale, 1.0), -1.0)
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        return linear_velocity, angular_velocity

class ThrusterController:
    """스러스터 제어 클래스"""
    
    @staticmethod
    def calculate_thruster_commands(linear_velocity, angular_velocity):
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

class WaypointManager:
    """웨이포인트 관리 클래스"""
    
    def __init__(self, coord_transformer):
        self.coord_transformer = coord_transformer
        self.waypoints = []  # 웨이포인트 리스트 (Unity NED 좌표계)
        self.current_target_index = 0
        self.target_position = None
        self.waypoint_reached = False
    
    def add_waypoint(self, gazebo_waypoint):
        """웨이포인트 추가 (Gazebo ENU → Unity NED 변환)"""
        unity_waypoint = self.coord_transformer.gazebo_enu_to_unity_ned(gazebo_waypoint)
        
        self.waypoints.append(unity_waypoint.tolist())
        self.current_target_index = len(self.waypoints) - 1
        self.target_position = unity_waypoint
        self.waypoint_reached = False
        
        return unity_waypoint
    
    def check_waypoint_reached(self, current_pos, threshold=15.0):
        """웨이포인트 도달 확인"""
        if self.target_position is None:
            return False, None
        
        distance = np.sqrt((current_pos[0] - self.target_position[0])**2 + (current_pos[1] - self.target_position[1])**2)
        
        if distance < threshold:
            if not self.waypoint_reached:
                self.waypoint_reached = True
                # 다음 웨이포인트로 이동
                self.current_target_index += 1
                if self.current_target_index < len(self.waypoints):
                    # 다음 웨이포인트가 있으면 이동
                    next_waypoint = self.waypoints[self.current_target_index]
                    self.target_position = np.array(next_waypoint, dtype=np.float32)
                    self.waypoint_reached = False
                    return True, next_waypoint
                else:
                    # 모든 웨이포인트 완료
                    self.target_position = None
                    return True, None
            return True, None
        
        return False, None
    
    def update_waypoint_positions(self):
        """웨이포인트 위치 업데이트 (Unity와 동일한 로직)"""
        if len(self.waypoints) == 0:
            # 웨이포인트가 없는 경우
            current_target_position = np.zeros(2)
            previous_target_position = np.zeros(2)
            next_target_position = np.zeros(2)
            return current_target_position, previous_target_position, next_target_position
        
        # 현재 웨이포인트 위치
        if self.current_target_index < len(self.waypoints):
            current_target_position = np.array(self.waypoints[self.current_target_index])
        else:
            current_target_position = np.zeros(2)
        
        # 이전 웨이포인트 위치
        if self.current_target_index > 0:
            previous_target_position = np.array(self.waypoints[self.current_target_index - 1])
        else:
            previous_target_position = np.zeros(2)  # 첫 번째 웨이포인트일 경우 (0,0)
        
        # 다음 웨이포인트 위치
        if self.current_target_index + 1 < len(self.waypoints):
            next_target_position = np.array(self.waypoints[self.current_target_index + 1])
        else:
            # 마지막 웨이포인트일 경우 현재 웨이포인트와 동일한 값
            next_target_position = current_target_position.copy()
        
        return current_target_position, previous_target_position, next_target_position

class LiDARProcessor:
    """LiDAR 데이터 처리 클래스"""
    
    @staticmethod
    def process_lidar_data(msg, sensor_manager):
        """LiDAR 데이터 처리 및 좌표계 변환"""
        lidar_data = sensor_manager.process_lidar_data(msg)
        
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
            lidar_distances = np.full(201, 50.0, dtype=np.float32)
            return lidar_distances, np.full(360, 50.0, dtype=np.float32)
        
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
        
        lidar_distances = resampled_ranges.astype(np.float32)
        
        # 360도 LiDAR 데이터도 저장 (goal_check용)
        if len(msg.ranges) >= 360:
            # 360도 데이터가 충분하면 그대로 사용
            lidar_distances_360 = np.array(msg.ranges[:360], dtype=np.float32)
        else:
            # 360도 데이터가 부족하면 기본값으로 채움
            lidar_distances_360 = np.full(360, 50.0, dtype=np.float32)
        
        return lidar_distances, lidar_distances_360

class SensorDataProcessor:
    """센서 데이터 처리 클래스"""
    
    def __init__(self, coord_transformer, sensor_manager):
        self.coord_transformer = coord_transformer
        self.sensor_manager = sensor_manager
        
        # Unity 관찰값 구조에 맞는 변수들 (Unity NED 좌표계 기준)
        self.lidar_distances = np.zeros(201, dtype=np.float32)  # LiDAR 거리 (201개)
        self.lidar_distances_360 = np.zeros(360, dtype=np.float32)  # LiDAR 거리 (360도 전체)
        self.agent_heading = 0.0                                # 에이전트 헤딩 (Unity 좌표계)
        self.angular_velocity_y = 0.0                           # IMU 각속도 (Unity 좌표계)
        
        # 위치 관련 변수들 (Unity NED 좌표계)
        self.agent_position_unity = np.zeros(2, dtype=np.float32)     # Unity NED 좌표계 위치
        self.agent_position_gazebo = np.zeros(2, dtype=np.float32)    # Gazebo ENU 좌표계 위치
        
        # 이전 명령 저장
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        # 각속도 추적용 변수 (Unity와 동일)
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.current_angular_acceleration = np.zeros(3)
        
        # 기준점 설정 상태
        self.reference_point_set = False
    
    def process_gps_data(self, msg):
        """GPS 데이터 처리 (좌표계 변환 포함)"""
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
                return True  # 기준점 설정 완료 신호
        
        return False
    
    def process_imu_data(self, msg):
        """IMU 데이터 처리 (회전 방향 변환 포함)"""
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
        
        # Z축 각속도 (헤딩 방향 - Yaw) - Gazebo ENU → Unity NED 변환
        # 회전 방향 차이 고려: Gazebo 반시계방향(+) ↔ Unity 시계방향(+)
        gazebo_angular_velocity_z = current_angular_velocity[2]  # Gazebo ENU 각속도
        self.angular_velocity_y = self.coord_transformer.gazebo_angular_velocity_to_unity_angular_velocity(gazebo_angular_velocity_z)
    
    def process_lidar_data(self, msg):
        """LiDAR 데이터 처리"""
        self.lidar_distances, self.lidar_distances_360 = LiDARProcessor.process_lidar_data(msg, self.sensor_manager)
    
    def create_observation_vector(self, current_target_position, previous_target_position, next_target_position):
        """Unity 관찰값 구조에 맞게 입력 벡터 생성 (Unity NED 좌표계 기준)"""
        observation_values = []
        
        # 1. LiDAR 거리 (201개)
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))
        
        # 2. 에이전트 헤딩 (1개) - 무한대 값 방지
        if np.isinf(self.agent_heading) or np.isnan(self.agent_heading):
            self.agent_heading = 0.0
        observation_values.append(float(self.agent_heading))
        
        # 3. IMU 각속도 Y축 (1개) - 무한대 값 방지 (회전 방향 변환 적용됨)
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
            val = float(current_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 6. 이전 웨이포인트 위치 (2개: X, Y) - Unity NED 좌표계 - 무한대 값 방지
        for i in range(2):
            val = float(previous_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 7. 다음 웨이포인트 위치 (2개: X, Y) - Unity NED 좌표계 - 무한대 값 방지
        for i in range(2):
            val = float(next_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 8. 이전 명령 (2개: moment_input, force_input)
        observation_values.append(float(self.previous_moment_input))
        observation_values.append(float(self.previous_force_input))
        
        return observation_values
    
    def update_previous_commands(self, linear_velocity, angular_velocity):
        """이전 명령 업데이트"""
        self.previous_moment_input = angular_velocity
        self.previous_force_input = linear_velocity
