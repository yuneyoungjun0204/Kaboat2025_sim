#!/usr/bin/env python3
"""
VRX ONNX 모델 기반 선박 제어 시스템 v9
- v2의 단순한 구조를 v5의 Unity 관찰값 구조로 업그레이드
- LiDAR 거리 (201개) + 헤딩 + 각속도 + 위치 + 웨이포인트들 + 이전 명령
- v5의 모델 입력 구조 (426개)를 사용하되 v2의 단순한 로직 유지
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
import math

class VRXONNXControllerV9(Node):
    """VRX ONNX 모델 기반 제어 노드 v9 - v2 구조 + v5 모델 입력"""
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v9')
        
        # ONNX 모델 로드 (v5와 동일한 모델 사용)
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-7315183.onnx'
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
        
        # 웨이포인트 서브스크라이버 (trajectory_viz에서 받음)
        self.waypoint_sub = self.create_subscription(
            Point, 
            '/vrx/waypoint', 
            self.waypoint_callback, 
            10
        )
        
        # ROS2 퍼블리셔 (스러스터 제어)
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        # v5의 Unity 관찰값 구조에 맞는 변수들
        self.lidar_distances = np.zeros(201, dtype=np.float32)      # LiDAR 거리 (201개)
        self.lidar_distances_360 = np.zeros(360, dtype=np.float32)  # LiDAR 거리 (360도 전체)
        self.agent_heading = 0.0                                    # 에이전트 헤딩 (Y rotation)
        self.angular_velocity_y = 0.0                               # IMU 각속도 (Y축)
        
        # 위치 관련 변수들 (Unity 좌표계: X=동서, Z=남북)
        self.agent_position = np.zeros(2, dtype=np.float32)         # 현재 에이전트 위치 (X, Z)
        self.current_target_position = np.zeros(2, dtype=np.float32)# 현재 웨이포인트 (X, Z)
        self.previous_target_position = np.zeros(2, dtype=np.float32)# 이전 웨이포인트 (X, Z)
        self.next_target_position = np.zeros(2, dtype=np.float32)   # 다음 웨이포인트 (X, Z)
        
        # 이전 명령 저장
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        # v2의 단순한 웨이포인트 관리 (고정 웨이포인트 + 동적 추가)
        self.waypoints = [
            [200.0, 0.0],    # 첫 번째 웨이포인트 (동쪽 50m)
            [100.0, 0.0],   # 두 번째 웨이포인트 (동쪽 100m)
            [100.0, 150.0], # 세 번째 웨이포인트 (동쪽 100m, 북쪽 150m)
        ]
        self.current_target_index = 0
        self.target_position = None
        
        # 각속도 추적용 변수 (Unity와 동일)
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.current_angular_acceleration = np.zeros(3)
        
        # v2의 스케일링 (TurtleBot과 동일)
        self.v_scale = 1    # linear velocity scale
        self.w_scale = -1   # angular velocity scale
        
        # LiDAR 데이터 스케일링 변수
        self.lidar_scale_factor = 8.0  # LiDAR 거리값을 나누는 값 (4.0 = 1/4 크기)
        self.lidar_max_distance = 50*self.lidar_scale_factor  # LiDAR 최대 거리 (미터)

        # 최근 스러스터 명령 저장용 변수
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        # 기준점 설정 상태
        self.reference_point_set = False
        
        # 웨이포인트 도달 상태
        self.waypoint_reached = False
        
        # v9 모드 정보 퍼블리셔
        self.mode_pub = self.create_publisher(String, '/vrx/current_mode', 10)
        
        # goal_check 영역 정보 퍼블리셔
        self.goal_check_pub = self.create_publisher(Float64MultiArray, '/vrx/goal_check_areas', 10)
        # 관찰값(211개) 디버깅용 퍼블리셔
        self.observation_pub = self.create_publisher(Float64MultiArray, '/vrx/observations', 10)

        # 10Hz 주기로 스러스터 제어
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('🚢 VRX ONNX Controller v9 시작! (v2 구조 + v5 모델 입력)')
        self.get_logger().info('📍 기본 웨이포인트 설정됨, trajectory_viz에서 추가 웨이포인트 설정 가능')

    def waypoint_callback(self, msg):
        """웨이포인트 콜백 - trajectory_viz에서 클릭한 점을 받음"""
        new_waypoint = [msg.x, msg.y]
        self.waypoints.append(new_waypoint)
        self.current_target_index = len(self.waypoints) - 1
        self.target_position = np.array(new_waypoint, dtype=np.float32)
        self.waypoint_reached = False
        self.get_logger().info(f'🎯 새 웨이포인트 추가: ({msg.y:.1f}, {msg.x:.1f}) - 총 {len(self.waypoints)}개')

    def gps_callback(self, msg):
        """GPS 데이터 콜백 - 로봇 위치 업데이트 (Unity 좌표계로 변환)"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            # UTM 좌표를 Unity 좌표계로 변환 (X=동서, Z=남북)
            self.agent_position = np.array([
                gps_data['utm_y'],  # Unity X (동서)
                gps_data['utm_x']   # Unity Z (남북)
            ], dtype=np.float32)
            
            # 첫 번째 GPS 데이터 기준점 설정 확인
            if not self.reference_point_set:
                self.reference_point_set = True
                self.get_logger().info(f'📍 기준점 설정 완료: (0, 0) - 첫 번째 GPS 위치')

    def imu_callback(self, msg):
        """IMU 데이터 콜백 - 헤딩과 각속도 업데이트"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        
        # Unity 헤딩 (Y rotation) - 0~360도
        self.agent_heading = imu_data['yaw_degrees']
        if self.agent_heading < 0:
            self.agent_heading += 360.0
            
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

    def goal_check(self, goal_distance, goal_psi):
        """
        목적지까지 경로에 장애물이 있는지 판단하는 함수 (v5에서 가져옴)
        """
        l = goal_distance
        boat_width = 10  # 배 폭 (미터)
        theta = int(np.degrees(np.arctan2(boat_width/2, l)))+np.pi/2
        
        check_ld = [0] * 360
        isAble = True
        
        # 영역 정보를 저장할 리스트
        area_info = []
        
        # 직사각형 경로 영역 체크
        dx = np.cos(np.radians(goal_psi))
        dy = np.sin(np.radians(goal_psi))
        
        # 직사각형의 4개 모서리 점들 계산
        half_width = boat_width / 3.0
        
        # 수직 방향 벡터 (목표 방향에 수직)
        perp_x = -dy  # 수직 방향
        perp_y = dx   # 수직 방향
        
        # 직사각형의 4개 모서리 (시계방향)
        corners = [
            # 왼쪽 앞 모서리 (목표 방향 + 왼쪽)
            [self.agent_position[1] + l * dx - half_width * perp_x, 
             self.agent_position[0] + l * dy - half_width * perp_y],
            # 오른쪽 앞 모서리 (목표 방향 + 오른쪽)
            [self.agent_position[1] + l * dx + half_width * perp_x,
             self.agent_position[0] + l * dy + half_width * perp_y],
            # 오른쪽 뒤 모서리 (현재 위치 + 오른쪽)
            [self.agent_position[1] + half_width * perp_x,
             self.agent_position[0] + half_width * perp_y],
            # 왼쪽 뒤 모서리 (현재 위치 + 왼쪽)
            [self.agent_position[1] - half_width * perp_x,
             self.agent_position[0] - half_width * perp_y]
        ]
        
        # 직사각형 영역 정보 저장 [type, x1, y1, x2, y2, x3, y3, x4, y4]
        area_info.extend([4.0])  # type 4 = 직사각형
        for corner in corners:
            area_info.extend([float(corner[0]), float(corner[1])])
        
        # 직사각형 경로 내의 LiDAR 포인트들 체크 (360도 전체 사용)
        for i, lidar_dist in enumerate(self.lidar_distances_360):
            if lidar_dist >= self.lidar_max_distance or lidar_dist <= 0.0:  # 유효하지 않은 데이터 스킵
                continue
                
            # LiDAR 각도 계산 (360도 전체: 0도 ~ 360도)
            lidar_angle = i  # 0도부터 359도까지
            lidar_angle_rad = np.radians(lidar_angle + self.agent_heading)  # 로봇 헤딩 기준
            
            # LiDAR 포인트의 실제 위치
            lidar_x = self.agent_position[1] + lidar_dist * np.cos(lidar_angle_rad)
            lidar_y = self.agent_position[0] + lidar_dist * np.sin(lidar_angle_rad)
            
            # 직사각형 내부에 있는지 체크 (점-다각형 포함 테스트)
            if self.point_in_polygon(lidar_x, lidar_y, corners):
                isAble = False
                self.get_logger().debug(f'🚧 직사각형 경로 내 장애물: 거리={lidar_dist:.1f}m, 각도={lidar_angle:.1f}°')
                break
        
        # 영역 정보를 ROS 메시지로 발행
        if len(area_info) > 0:
            area_msg = Float64MultiArray()
            area_msg.data = area_info
            self.goal_check_pub.publish(area_msg)
        
        return isAble
    
    def normalize_angle(self, angle):
        """각도를 0-359도 범위로 정규화"""
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
    
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

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 - Unity와 동일한 201개 거리 데이터"""
        lidar_data = self.sensor_manager.process_lidar_data(msg)
        
        # LiDAR 데이터를 -100도~100도 범위로 필터링 (raw data 사용)
        lidar_ranges = lidar_data['ranges']
        lidar_angles = lidar_data['angles']
        
        # -100도 ~ 100도 범위의 LiDAR 데이터 필터링
        angle_mask = (lidar_angles >= np.radians(-100)) & (lidar_angles <= np.radians(100))
        filtered_ranges = lidar_ranges[angle_mask]
        filtered_angles = lidar_angles[angle_mask]  # 필터링된 각도도 함께 추출
        
        # 필터링된 데이터가 비어있는지 확인
        if len(filtered_ranges) == 0:
            # 필터링된 데이터가 없으면 기본값으로 채움
            self.lidar_distances = np.full(201, self.lidar_max_distance, dtype=np.float32)
            self.get_logger().warn('LiDAR 필터링 데이터 없음 - 기본값 사용')
            return
        
        # raw data를 -100도~100도 범위로 직접 사용 (리샘플링 없음)
        # LiDAR는 -100도부터 100도까지 0도를 포함해서 총 201개 포인트
        raw_ranges = np.full(201, self.lidar_max_distance, dtype=np.float32)  # 기본값으로 초기화
        
        # 필터링된 데이터가 있으면 해당 위치에 배치
        if len(filtered_ranges) > 0:
            # 각도를 인덱스로 변환하여 올바른 위치에 배치
            for i, angle in enumerate(filtered_angles):
                # 각도를 인덱스로 변환 (-100도 = 0, 0도 = 100, 100도 = 200)
                angle_deg = np.degrees(angle)
                if -100 <= angle_deg <= 100:
                    idx = int(angle_deg + 100)  # -100도를 0번 인덱스로 변환
                    idx = max(0, min(200, idx))  # 범위 제한
                    raw_ranges[idx] = filtered_ranges[i]
        
        # 무한대 값 방지 및 거리값 스케일링
        for i in range(len(raw_ranges)):
            if np.isinf(raw_ranges[i]) or np.isnan(raw_ranges[i]) or raw_ranges[i]>=self.lidar_max_distance:
                raw_ranges[i] = self.lidar_max_distance  # 최대 거리로 설정
            else:
                raw_ranges[i] = raw_ranges[i] / self.lidar_scale_factor  # 거리값 스케일링
        
        self.lidar_distances = raw_ranges.astype(np.float32)
        
        # 360도 LiDAR 데이터도 저장 (goal_check용) - 거리값 스케일링
        if len(msg.ranges) >= 360:
            # 360도 데이터가 충분하면 거리값을 스케일링해서 사용
            raw_360_ranges = np.array(msg.ranges[:360], dtype=np.float32)
            for i in range(len(raw_360_ranges)):
                if np.isinf(raw_360_ranges[i]) or np.isnan(raw_360_ranges[i]) or raw_360_ranges[i]>=self.lidar_max_distance:
                    raw_360_ranges[i] = self.lidar_max_distance  # 최대 거리로 설정
                else:
                    raw_360_ranges[i] = raw_360_ranges[i] / self.lidar_scale_factor  # 거리값 스케일링
            self.lidar_distances_360 = raw_360_ranges
        else:
            # 360도 데이터가 부족하면 기본값으로 채움
            self.lidar_distances_360 = np.full(360, self.lidar_max_distance, dtype=np.float32)
        
        # 제어 실행
        self.control_vrx()

    def control_vrx(self):
        """v2의 단순한 제어 로직 + v5의 모델 입력 구조"""
        # 현재 웨이포인트 설정 (v2의 단순한 방식)
        if self.current_target_index < len(self.waypoints):
            self.target_position = np.array(self.waypoints[self.current_target_index], dtype=np.float32)
        else:
            self.target_position = np.array(self.waypoints[-1], dtype=np.float32)  # 마지막 웨이포인트 유지

        # 웨이포인트 도달 확인 (v2의 단순한 방식)
        current_pos = self.agent_position
        distance = np.sqrt((current_pos[0] - self.target_position[0])**2 + (current_pos[1] - self.target_position[1])**2)
        if distance < 10.0:  # 10m 이내 도달
            self.current_target_index += 1
            self.get_logger().info(f'🎯 웨이포인트 {self.current_target_index-1} 도달! 다음: {self.get_next_waypoint()}')

        # v5의 모델 입력 구조 사용
        # 웨이포인트 위치 업데이트
        self.update_waypoint_positions()

        # Unity 관찰값 구조에 맞게 입력 벡터 생성 (213개) - v2 방식 사용
        input_vector = np.zeros(213, dtype=np.float32)
        
        # 1. LiDAR 거리 (201개) - v2와 동일
        for i in range(len(self.lidar_distances)):
            distance = float(self.lidar_distances[i])
            # 무한대 값 방지
            if np.isinf(distance) or np.isnan(distance):
                distance = self.lidar_max_distance
            input_vector[i] = distance
        
        # 2. 에이전트 헤딩 (1개) - v2와 동일
        rotation_y = float(self.agent_heading)
        if np.isinf(rotation_y) or np.isnan(rotation_y):
            rotation_y = 0.0
        input_vector[201] = rotation_y
        
        # 3. IMU 각속도 Y축 (1개) - v2와 동일
        heading_angular_velocity = float(self.angular_velocity_y)
        if np.isinf(heading_angular_velocity) or np.isnan(heading_angular_velocity):
            heading_angular_velocity = 0.0
        input_vector[202] = heading_angular_velocity
        
        # 4. 현재 에이전트 위치 (2개: X, Z) - v2와 동일
        agent_pos_x = float(self.agent_position[0])
        agent_pos_z = float(self.agent_position[1])
        if np.isinf(agent_pos_x) or np.isnan(agent_pos_x):
            agent_pos_x = 0.0
        if np.isinf(agent_pos_z) or np.isnan(agent_pos_z):
            agent_pos_z = 0.0
        input_vector[203:205] = [agent_pos_x, agent_pos_z]
        
        # 5. 현재 웨이포인트 위치 (2개: X, Z) - v2와 동일
        current_target_pos_x = float(self.current_target_position[0])
        current_target_pos_z = float(self.current_target_position[1])
        if np.isinf(current_target_pos_x) or np.isnan(current_target_pos_x):
            current_target_pos_x = 0.0
        if np.isinf(current_target_pos_z) or np.isnan(current_target_pos_z):
            current_target_pos_z = 0.0
        input_vector[205:207] = [current_target_pos_x, current_target_pos_z]
        
        # 6. 이전 웨이포인트 위치 (2개: X, Z) - v2와 동일
        previous_target_pos_x = float(self.previous_target_position[0])
        previous_target_pos_z = float(self.previous_target_position[1])
        if np.isinf(previous_target_pos_x) or np.isnan(previous_target_pos_x):
            previous_target_pos_x = 0.0
        if np.isinf(previous_target_pos_z) or np.isnan(previous_target_pos_z):
            previous_target_pos_z = 0.0
        input_vector[207:209] = [previous_target_pos_x, previous_target_pos_z]
        
        # 7. 다음 웨이포인트 위치 (2개: X, Z) - v2와 동일
        next_target_pos_x = float(self.next_target_position[0])
        next_target_pos_z = float(self.next_target_position[1])
        if np.isinf(next_target_pos_x) or np.isnan(next_target_pos_x):
            next_target_pos_x = 0.0
        if np.isinf(next_target_pos_z) or np.isnan(next_target_pos_z):
            next_target_pos_z = 0.0
        input_vector[209:211] = [next_target_pos_x, next_target_pos_z]
        
        # 8. 이전 명령 (2개: moment_input, force_input) - v2와 동일
        input_vector[211:213] = [float(self.previous_moment_input), float(self.previous_force_input)]

        # 관찰값을 numpy 배열로 변환 (Unity와 동일한 크기: 213개)
        observation_array = input_vector

        # ──────────────────────────────────────────────────────────────
        # DEBUG: 관찰값 퍼블리시 및 inf/nan 검사
        obs_msg = Float64MultiArray()
        obs_msg.data = observation_array.astype(float).tolist()
        self.observation_pub.publish(obs_msg)

        if np.isinf(observation_array).any() or np.isnan(observation_array).any():
            self.get_logger().warn(
                f"⚠️ 관찰값에 inf 또는 nan 포함: inf={np.isinf(observation_array).sum()} nan={np.isnan(observation_array).sum()}"
            )
        # ──────────────────────────────────────────────────────────────
        
        # ONNX 모델이 426개 입력을 기대하므로 stacked input 구조 사용
        # 213개 관찰값을 2번 반복해서 426개로 만들기
        stacked_input = np.concatenate([
            observation_array,  # 첫 번째 213개 데이터
            observation_array   # 두 번째 213개 데이터 (동일한 데이터 반복)
        ]).reshape(1, 426)
        
        model_input = stacked_input

        # ONNX 모델 추론
        outputs = self.session.run(None, {self.input_name: model_input})
        
        # ONNX 모델 출력 구조 확인 및 처리
        # outputs는 리스트이며, 각 항목은 NumPy 배열입니다
        # 일반적으로 deterministic_continuous_actions가 3번째 출력 (인덱스 2)
        # continuous_actions가 4번째 출력 (인덱스 3)에 위치합니다
        
        linear_velocity = 0.0
        angular_velocity = 0.0
        
        if len(outputs) >= 4:
            # deterministic_continuous_actions (3번째 출력, 인덱스 2)
            if outputs[4].shape == (1, 2):  # shape: [1, 2]
                linear_velocity = float(outputs[4][0][0]) * self.v_scale
                angular_velocity = float(outputs[4][0][1]) * self.w_scale
            # continuous_actions (4번째 출력, 인덱스 3) - 백업 옵션
            elif outputs[2].shape == (1, 2):  # shape: [1, 2]
                print(outputs[3].shape)
                linear_velocity = float(outputs[2][0][0]) * self.v_scale
                angular_velocity = float(outputs[2][0][1]) * self.w_scale
            else:
                self.get_logger().warn(f"예상치 못한 출력 형태: outputs[2]={outputs[2].shape}, outputs[3]={outputs[3].shape}")
        elif len(outputs) >= 2:
            # 이전 방식 (호환성 유지)
            linear_velocity = float(outputs[0][0]) * self.v_scale
            angular_velocity = float(outputs[1][0]) * self.w_scale
        else:
            self.get_logger().warn(f"출력 개수 부족: {len(outputs)}개 (최소 2개 필요)")
        
        # 값 범위 제한 (Unity와 동일)
        linear_velocity = np.clip(linear_velocity, 0.1, 0.5)
        angular_velocity = -np.clip(angular_velocity, -0.5, 0.5)

        # 이전 명령 업데이트 (Unity와 동일)
        self.previous_moment_input = angular_velocity
        self.previous_force_input = linear_velocity

        # v2의 스러스터 명령 변환 방식 사용
        self.left_thrust, self.right_thrust = self.calculate_thruster_commands(linear_velocity, angular_velocity)

        # trajectory_viz.py로 출력값 전송을 위한 퍼블리셔
        if not hasattr(self, 'control_output_pub'):
            self.control_output_pub = self.create_publisher(Float64MultiArray, '/vrx/control_output', 10)
        
        # 제어 출력값 발행 [linear_velocity, angular_velocity]
        output_msg = Float64MultiArray()
        output_msg.data = [linear_velocity, angular_velocity]
        self.control_output_pub.publish(output_msg)

        # 모드 정보 발행
        mode_msg = String()
        mode_msg.data = "ONNX_V9"
        self.mode_pub.publish(mode_msg)

        # 출력 정보 로깅 (디버깅용)
        output_info = f"출력개수:{len(outputs)}"
        if len(outputs) >= 4:
            output_info += f" | det_actions:{outputs[2].shape} | cont_actions:{outputs[3].shape}"
        
        self.get_logger().info(
            f"V9모델(Unity+Stacked): 위치: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | "
            f"웨이포인트: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) | "
            f"거리: {distance:.1f}m | "
            f"제어값: Linear={linear_velocity:.3f}, Angular={angular_velocity:.3f} | "
            f"스러스터: L={self.left_thrust:.1f}, R={self.right_thrust:.1f} | "
            f"관찰값: {len(observation_array)}→{len(model_input.flatten())} | {output_info}"
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
        """v2의 스러스터 명령 변환 방식 사용"""
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

    def get_next_waypoint(self):
        """다음 웨이포인트 반환 (v2에서 가져옴)"""
        if self.current_target_index < len(self.waypoints):
            return self.waypoints[self.current_target_index]
        else:
            return self.waypoints[-1]

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
        node = VRXONNXControllerV9()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
