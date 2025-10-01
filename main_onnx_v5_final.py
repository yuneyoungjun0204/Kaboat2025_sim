#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point
import time
from utils import SensorDataManager
import math

class VRXONNXControllerV5(Node):
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v5')
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-3076146.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-10721680.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-12499862.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-14248543.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-6953161.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-3999831.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-8933853.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/correct_IMU/Ray-5999760.onnx'
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/Ray-19946289.onnx'
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.sensor_manager = SensorDataManager()
        
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan', self.lidar_callback, 10)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        self.waypoint_sub = self.create_subscription(Point, '/vrx/waypoint', self.waypoint_callback, 10)
        
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.model_input_pub = self.create_publisher(Float64MultiArray, '/vrx/model_input', 10)
        self.lidar_pub = self.create_publisher(Float64MultiArray, '/vrx/lidar_data', 10)
        self.heading_pub = self.create_publisher(Float64, '/vrx/agent_heading', 10)
        self.angular_vel_pub = self.create_publisher(Float64, '/vrx/angular_velocity', 10)
        self.position_pub = self.create_publisher(Float64MultiArray, '/vrx/agent_position', 10)
        self.current_waypoint_pub = self.create_publisher(Float64MultiArray, '/vrx/current_waypoint', 10)
        self.previous_waypoint_pub = self.create_publisher(Float64MultiArray, '/vrx/previous_waypoint', 10)
        self.next_waypoint_pub = self.create_publisher(Float64MultiArray, '/vrx/next_waypoint', 10)
        self.previous_moment_pub = self.create_publisher(Float64, '/vrx/previous_moment', 10)
        self.previous_force_pub = self.create_publisher(Float64, '/vrx/previous_force', 10)
        self.control_output_pub = self.create_publisher(Float64MultiArray, '/vrx/control_output', 10)
        self.control_mode_pub = self.create_publisher(String, '/vrx/control_mode', 10)
        self.obstacle_check_area_pub = self.create_publisher(Float64MultiArray, '/vrx/obstacle_check_area', 10)
        self.los_target_pub = self.create_publisher(Float64MultiArray, '/vrx/los_target', 10)

        self.lidar_distances = np.zeros(201, dtype=np.float32)
        self.max_lidar_distance=100.0
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.current_target_position = np.zeros(2, dtype=np.float32)
        self.previous_target_position = np.zeros(2, dtype=np.float32)
        self.next_target_position = np.zeros(2, dtype=np.float32)
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        self.waypoints = []
        self.current_target_index = 0
        self.target_position = None
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.v_scale = 1.0
        self.w_scale = -1.0
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.reference_point_set = False
        self.waypoint_reached = False
        self.thrust_scale=800
        self.angular_velocity_y_scale=1
        
        # LiDAR 스케일링 변수
        # self.lidar_scale_factor = 1.5  # LiDAR 거리값을 나누는 스케일 팩터
        self.lidar_scale_factor = 1.0  # LiDAR 거리값을 나누는 스케일 팩터
        self.lidar_max_distance = self.max_lidar_distance  # LiDAR 최대 거리값
        
        # 직접제어 모드 관련 변수
        self.boat_width = 2.2  # 배 폭 D (미터)
        self.boat_height = 50.0  # 배 높이 L (미터)
        
        self.use_direct_control = False  # 직접제어 모드 플래그
        
        # LOS guidance 관련 변수
        self.los_delta =10.0  # 고정 delta 값 (미터) - 수직 오프셋
        self.los_lookahead_min = 30.0  # 최소 look-ahead 거리 (미터)
        self.los_lookahead_max = 80.0  # 최대 look-ahead 거리 (미터)
        self.los_lookahead_factor = 1.0  # look-ahead 거리 계산 계수 (조정 가능)
        
        # 1차 저주파 필터 변수
        self.filter_alpha = 0.35  # 필터 계수 (0~1, 낮을수록 더 부드러움)
        self.filtered_linear_velocity = 0.0  # 필터링된 선형 속도
        self.filtered_angular_velocity = 0.0  # 필터링된 각속도
        self.filtered_left_thrust = 0.0  # 필터링된 좌측 스러스터
        self.filtered_right_thrust = 0.0  # 필터링된 우측 스러스터
        
        self.timer = self.create_timer(0.01, self.timer_callback)







    def waypoint_callback(self, msg):
        new_waypoint = [msg.y, msg.x]
        self.waypoints.append(new_waypoint)
        self.current_target_index = len(self.waypoints) - 1
        self.target_position = np.array(new_waypoint, dtype=np.float32)
        self.waypoint_reached = False








    def gps_callback(self, msg):
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            self.agent_position = np.array([gps_data['utm_y'], gps_data['utm_x']], dtype=np.float32)
            if not self.reference_point_set:
                self.reference_point_set = True









    def imu_callback(self, msg):
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.agent_heading = imu_data['yaw_degrees']
        if self.agent_heading < 0:
            self.agent_heading += 360.0
        current_time = time.time()
        current_angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        if self.agent_heading  > 0:
            delta_time = current_time - self.last_angular_velocity_update_time
            if delta_time > 0:
                self.current_angular_acceleration = (current_angular_velocity - self.previous_angular_velocity) / delta_time
        self.previous_angular_velocity = current_angular_velocity
        self.last_angular_velocity_update_time = current_time
        self.angular_velocity_y = min(max(current_angular_velocity[2]*self.angular_velocity_y_scale,-180),180)











    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        raw_ranges = np.full(201, self.lidar_max_distance, dtype=np.float32)
        
        for i in range(len(ranges)):
            angle_rad = angle_min + i * angle_increment
            angle_deg = np.degrees(angle_rad)
            
            if -100 <= angle_deg <= 100:
                distance = ranges[i]
                if np.isinf(distance) or np.isnan(distance) or distance >= self.lidar_max_distance:
                    distance = self.lidar_max_distance
                else:
                    # 스케일 팩터로 나누어 스케일링
                    distance = distance / self.lidar_scale_factor
                
                idx = int(angle_deg + 100)
                idx = max(0, min(200, idx))
                raw_ranges[idx] = distance
        
        self.lidar_distances = raw_ranges.astype(np.float32)
        self.control_vrx()

    def calculate_goal_psi(self, current_pos, target_pos):
        """목적지와 현재 위치의 차이로부터 goal_psi 계산"""
        dx = target_pos[0] - current_pos[0]  # X 방향 차이
        dy = target_pos[1] - current_pos[1]  # Y 방향 차이
        goal_psi = np.arctan2(dy, dx)  # atan2(dy/dx)
        return goal_psi

    def calculate_distance_L(self, current_pos, target_pos):
        """목적지와 현재 위치 간의 거리 L 계산"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        L = min(self.boat_height,np.sqrt(dx**2 + dy**2))
        return L

    def calculate_range_theta(self, L):
        """D만큼 탐색할 각도 범위 계산"""
        range_theta = np.arctan2(self.boat_width, L)  # atan(D/L)
        return range_theta

    def calculate_theta_lidar(self, goal_psi, current_psi):
        """LiDAR 탐색 기준 각도 계산"""
        theta_lidar = goal_psi - current_psi
        return theta_lidar

    def normalize_angle(self, angle):
        """각도를 -π ~ π 범위로 정규화"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def calculate_crosstrack_error(self, current_pos, line_start, line_end):
        """현재 위치에서 직선까지의 crosstrack error 계산"""
        # 직선의 방향 벡터
        line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            return 0.0
        
        # 현재 위치에서 직선 시작점까지의 벡터
        point_vec = np.array([current_pos[0] - line_start[0], current_pos[1] - line_start[1]])
        
        # 직선 방향 벡터의 단위 벡터
        line_unit = line_vec / line_length
        
        # crosstrack error 계산 (직선에 수직인 거리)
        crosstrack_error = np.linalg.norm(point_vec - np.dot(point_vec, line_unit) * line_unit)
        
        # 부호 결정 (직선의 왼쪽/오른쪽)
        cross_product = np.cross(line_unit, point_vec)
        if cross_product < 0:
            crosstrack_error = -crosstrack_error
        
        return crosstrack_error

    def calculate_adaptive_lookahead_distance(self, crosstrack_error):
        """crosstrack error와 반비례하는 adaptive look-ahead distance 계산"""
        # crosstrack error의 절댓값 사용
        abs_crosstrack_error = abs(crosstrack_error)
        
        # look-ahead distance = los_lookahead_factor / (1 + abs_crosstrack_error)
        # crosstrack error가 클수록 look-ahead distance가 작아짐 (더 정밀한 경로 추종)
        adaptive_lookahead = self.los_lookahead_factor / (1.0 + abs_crosstrack_error * 0.1)
        
        # 범위 제한
        adaptive_lookahead = np.clip(adaptive_lookahead, self.los_lookahead_min, self.los_lookahead_max)
        
        return adaptive_lookahead

    def calculate_los_guidance_point(self, current_pos, waypoint_start, waypoint_end):
        """LOS guidance를 사용한 look-ahead point 계산"""
        # 현재 위치에서 직선까지의 crosstrack error 계산
        crosstrack_error = self.calculate_crosstrack_error(current_pos, waypoint_start, waypoint_end)
        
        # adaptive look-ahead distance 계산
        adaptive_lookahead = self.calculate_adaptive_lookahead_distance(crosstrack_error)
        
        # 직선의 방향 벡터
        line_vec = np.array([waypoint_end[0] - waypoint_start[0], waypoint_end[1] - waypoint_start[1]])
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            return waypoint_end
        
        # 직선 방향 벡터의 단위 벡터
        line_unit = line_vec / line_length
        
        # LOS look-ahead point 계산
        # 현재 위치에서 직선까지의 수직 거리를 계산하여 직선 위의 정확한 지점을 찾는 방식
        
        # 현재 위치에서 직선까지의 투영점 계산
        point_vec = np.array([current_pos[0] - waypoint_start[0], current_pos[1] - waypoint_start[1]])
        projection_length = np.dot(point_vec, line_unit)
        projection_point = np.array(waypoint_start) + projection_length * line_unit
        
        # 투영점에서 line_unit 방향으로 adaptive look-ahead distance만큼 이동
        los_point = projection_point + adaptive_lookahead * line_unit
        
        # crosstrack error를 사용하여 수직 오프셋 계산 (점과 직선 사이의 거리 기반)
        # crosstrack error의 부호와 크기를 그대로 사용
        perpendicular_unit = np.array([line_unit[1], -line_unit[0]])  # 수직 단위 벡터
        
        # crosstrack error의 부호에 따라 수직 방향 결정
        if crosstrack_error < 0:
            perpendicular_unit = -perpendicular_unit
        
        # crosstrack error를 기반으로 한 수직 오프셋 (점과 직선 사이의 실제 거리 사용)
        perpendicular_offset = abs(crosstrack_error) * self.los_delta / 10.0  # delta를 비례 계수로 사용
        
        # 최종 LOS point 계산
        los_point += perpendicular_offset * perpendicular_unit
        
        return los_point

    def check_obstacles_in_rectangle_path(self, current_pos, target_pos):
        """LOS guidance를 사용한 직사각형 경로 내 장애물 존재 여부 확인 및 체크 영역 반환"""
        # 1. LOS guidance point 계산
        # 현재 웨이포인트와 다음 웨이포인트를 사용하여 LOS point 계산
        if len(self.waypoints) > 1 and self.current_target_index < len(self.waypoints) - 1:
            waypoint_start = self.waypoints[self.current_target_index]
            waypoint_end = self.waypoints[self.current_target_index + 1]
            los_target = self.calculate_los_guidance_point(current_pos, waypoint_start, waypoint_end)
        else:
            # 웨이포인트가 하나뿐이거나 마지막 웨이포인트인 경우
            if len(self.waypoints) > 0:
                # 현재 웨이포인트를 사용하되, 이전 웨이포인트가 없으면 (0,0)을 사용
                if self.current_target_index > 0:
                    waypoint_start = self.waypoints[self.current_target_index - 1]
                else:
                    waypoint_start = [0.0, 0.0]  # 이전 웨이포인트가 없으면 (0,0) 가정
                waypoint_end = self.waypoints[self.current_target_index]
                los_target = self.calculate_los_guidance_point(current_pos, waypoint_start, waypoint_end)
            else:
                # 웨이포인트가 없으면 직접 목적지 사용
                los_target = target_pos
        
        # 2. LOS target과 현재 위치 간의 거리 L 계산
        L = self.calculate_distance_L(current_pos, los_target)
        
        # 3. goal_psi 계산 (LOS target 기준)
        goal_psi = self.calculate_goal_psi(current_pos, los_target)
        
        # 3. 현재 자신의 각도 (NED 좌표계 기준)
        current_psi = np.radians((self.agent_heading+180)%360-180)
        
        # 4. LiDAR 탐색 기준 각도 계산
        theta_lidar = self.calculate_theta_lidar(goal_psi, current_psi)
        theta_lidar = -self.normalize_angle(theta_lidar)
        
        # 5. D만큼 탐색할 각도 범위 계산
        range_theta = self.calculate_range_theta(L)
        
        # 6. 체크 영역 점들 생성 (시각화용)
        check_area_points = []
        
        # LiDAR 데이터로 장애물 검사
        obstacle_found = False
        
        # 1. 직사각형 경로 내 장애물 검사 (기존 로직)
        for i in range(-90, 90):  # 180도 탐색
            # 현재 탐색 각도
            theta = theta_lidar + np.radians(i)
            theta = self.normalize_angle(theta-np.pi/2)
            squar_theta = np.pi/2-theta
            if abs(i) <= range_theta*180/np.pi:
                # D만큼 탐색하는 구간
                search_distance = L 
            else:
                search_distance=self.boat_width/np.sin(abs(i)*np.pi/180)
                # search_distance = math.sqrt((L*(90-abs(i))/(90-range_theta/2*180/np.pi))**2+self.boat_width**2)
            
            # 체크 영역 점 추가 (시각화용)
            check_y = current_pos[0] + search_distance * np.cos(theta-theta_lidar+goal_psi+np.pi/2)
            check_x = current_pos[1] + search_distance * np.sin(theta-theta_lidar+goal_psi+np.pi/2)
            check_area_points.extend([check_x, check_y])
            
            # LiDAR에서 해당 각도의 거리 데이터 가져오기
            lidar_distance = self.get_lidar_distance_at_angle_degrees(np.degrees(theta))
            if lidar_distance > self.max_lidar_distance or lidar_distance < 0.0 or np.isinf(lidar_distance):
                lidar_distance = self.max_lidar_distance
            
            # 장애물 검사
            if lidar_distance < search_distance:
                obstacle_found = True
        

        
        L_front = 20.0
        range_theta = self.calculate_range_theta(L_front)
        for i in range(-90, 90):  # 180도 탐색
            # 현재 탐색 각도
            theta = np.radians(i)
            theta = self.normalize_angle(theta)
            if abs(i) <= range_theta*180/np.pi:
                # D만큼 탐색하는 구간
                search_distance = L_front
            else:
                search_distance=self.boat_width/np.sin(abs(i)*np.pi/180)
                # search_distance = math.sqrt((L*(90-abs(i))/(90-range_theta/2*180/np.pi))**2+self.boat_width**2)
            
            # 체크 영역 점 추가 (시각화용)
            check_y = current_pos[0] + search_distance * np.cos(theta)
            check_x = current_pos[1] + search_distance * np.sin(theta)
            check_area_points.extend([check_x, check_y])
            
            # LiDAR에서 해당 각도의 거리 데이터 가져오기
            lidar_distance = self.get_lidar_distance_at_angle_degrees(np.degrees(theta))
            if lidar_distance > self.max_lidar_distance or lidar_distance < 0.0 or np.isinf(lidar_distance):
                lidar_distance = self.max_lidar_distance
            
            # 정면 90도 내에서 20m 이내에 장애물이 있으면 장애물로 판단
            if lidar_distance < search_distance:
                obstacle_found = True

        
        # 체크 영역 토픽 발행
        area_msg = Float64MultiArray()
        area_msg.data = check_area_points
        self.obstacle_check_area_pub.publish(area_msg)
        
        # LOS target 정보 발행 (trajectory_viz.py에서 시각화용)
        los_target_msg = Float64MultiArray()
        los_target_msg.data = [current_pos[1]+los_target[1], +current_pos[0]+los_target[0]]  # [X, Y]
        self.los_target_pub.publish(los_target_msg)
        
        return obstacle_found, los_target

    def get_lidar_distance_at_angle_degrees(self, angle_deg):
        """주어진 각도(도 단위)에서 LiDAR 거리 데이터 가져오기"""
        # 각도를 -100~100도 범위로 정규화
        while angle_deg > 100:
            angle_deg -= 360
        while angle_deg < -100:
            angle_deg += 360
        
        # -100~100도 범위 내에서만 처리
        if -180 <= angle_deg <= 180:
            # 각도를 인덱스로 변환
            idx = int(angle_deg + 100)
            idx = max(0, min(200, idx))
            return self.lidar_distances[idx]
        else:
            return self.lidar_max_distance  # 범위 밖이면 최대 거리 반환

    def calculate_direct_heading_to_target(self, current_pos, target_pos):
        """목적지로 향하는 직접 헤딩 계산"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # 목표 방향 계산 (NED 좌표계)
        target_heading_rad = np.arctan2(dx, dy)
        
        # 현재 헤딩과의 차이 계산
        current_heading_rad = np.radians(self.agent_heading)
        heading_diff_rad = target_heading_rad - current_heading_rad
        
        # -π ~ π 범위로 정규화
        heading_diff_rad = -np.arctan2(np.sin(heading_diff_rad), np.cos(heading_diff_rad))
        
        return heading_diff_rad

    def apply_direct_control(self, current_pos, los_target):
        """LOS target을 향한 직접제어 모드 적용"""
        # LOS target으로 향하는 헤딩 계산
        heading_diff_rad = self.calculate_direct_heading_to_target(current_pos, los_target)
        
        # LOS target까지의 거리 계산
        distance_to_los = np.sqrt((los_target[0] - current_pos[0])**2 + (los_target[1] - current_pos[1])**2)
        
        # 직접제어 명령 계산 (LOS guidance 방식)
        # 헤딩 차이가 클수록 더 강한 각속도 적용
        angular_velocity = np.clip(heading_diff_rad / np.pi, -0.7, 0.7)
        
        # 거리에 따른 선형 속도 조절 (가까울수록 느리게)
        if distance_to_los > 20.0:
            linear_velocity = 1.0  # 원거리에서 빠른 속도
        elif distance_to_los > 10.0:
            linear_velocity = 0.6  # 중거리에서 중간 속도
        else:
            linear_velocity = 0.4  # 근거리에서 느린 속도
        
        # 각속도에 따른 선형 속도 보정 (회전할 때는 속도 감소)
        linear_velocity = linear_velocity * (1.0 - abs(angular_velocity) * 0.3)
        linear_velocity = np.clip(linear_velocity, 0.1, 1.0)  # 최소/최대 속도 제한
        
        return linear_velocity, angular_velocity

    def apply_low_pass_filter(self, new_value, current_filtered_value, alpha=None):
        """1차 저주파 필터 적용
        new_value: 새로운 입력값
        current_filtered_value: 현재 필터링된 값
        alpha: 필터 계수 (기본값: self.filter_alpha)
        """
        if alpha is None:
            alpha = self.filter_alpha
        
        # 1차 저주파 필터 공식: y[n] = α * x[n] + (1-α) * y[n-1]
        filtered_value = alpha * new_value + (1 - alpha) * current_filtered_value
        return filtered_value

    def filter_control_commands(self, linear_velocity, angular_velocity):
        """제어 명령에 1차 저주파 필터 적용"""
        # 선형 속도 필터링
        self.filtered_linear_velocity = self.apply_low_pass_filter(
            linear_velocity, self.filtered_linear_velocity
        )
        
        # 각속도 필터링
        self.filtered_angular_velocity = self.apply_low_pass_filter(
            angular_velocity, self.filtered_angular_velocity
        )
        
        return self.filtered_linear_velocity, self.filtered_angular_velocity

    def filter_thruster_commands(self, left_thrust, right_thrust):
        """스러스터 명령에 1차 저주파 필터 적용"""
        # 좌측 스러스터 필터링
        self.filtered_left_thrust = self.apply_low_pass_filter(
            left_thrust, self.filtered_left_thrust
        )
        
        # 우측 스러스터 필터링
        self.filtered_right_thrust = self.apply_low_pass_filter(
            right_thrust, self.filtered_right_thrust
        )
        
        return self.filtered_left_thrust, self.filtered_right_thrust











    def control_vrx(self):
        if self.target_position is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return
        
        current_pos = self.agent_position
        distance = np.sqrt((current_pos[0] - self.target_position[0])**2 + (current_pos[1] - self.target_position[1])**2)
        
        if distance < 15.0:
            if not self.waypoint_reached:
                self.waypoint_reached = True
                self.current_target_index += 1
                if self.current_target_index < len(self.waypoints):
                    next_waypoint = self.waypoints[self.current_target_index]
                    self.target_position = np.array(next_waypoint, dtype=np.float32)
                    self.waypoint_reached = False
                else:
                    self.target_position = None
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return
        
        self.update_waypoint_positions()
        
        # 장애물 검사 및 제어 모드 결정
        has_obstacles, los_target = self.check_obstacles_in_rectangle_path(current_pos, self.target_position)
        
        if has_obstacles:
            # 장애물이 있으면 ONNX 모델 사용
            self.use_direct_control = False
            linear_velocity, angular_velocity = self.get_onnx_control()
        else:
            # 장애물이 없으면 직접제어 모드 사용 (LOS target을 따라감)
            self.use_direct_control = True
            linear_velocity, angular_velocity = self.apply_direct_control(current_pos, los_target)
        
        # 1차 저주파 필터 적용하여 부드러운 명령 생성
        filtered_linear_velocity, filtered_angular_velocity = self.filter_control_commands(linear_velocity, angular_velocity)
        
        # 이전 명령 업데이트 (필터링된 값 사용)
        self.previous_moment_input = filtered_angular_velocity
        self.previous_force_input = filtered_linear_velocity
        
        # 스러스터 명령 계산 (필터링된 값 사용)
        raw_left_thrust, raw_right_thrust = self.calculate_thruster_commands(filtered_linear_velocity, filtered_angular_velocity)
        
        # 스러스터 명령에도 필터 적용
        self.left_thrust, self.right_thrust = self.filter_thruster_commands(raw_left_thrust, raw_right_thrust)
        
        # 제어 출력값 발행 (필터링된 값 사용, trajectory_viz.py에서 사용)
        control_output_msg = Float64MultiArray()
        control_output_msg.data = [filtered_linear_velocity, filtered_angular_velocity]
        self.control_output_pub.publish(control_output_msg)
        
        # 제어 모드 발행 (trajectory_viz.py에서 사용)
        mode_msg = String()
        if self.use_direct_control:
            mode_msg.data = "DIRECT_CONTROL"
        else:
            mode_msg.data = "ONNX_MODEL"
        self.control_mode_pub.publish(mode_msg)

    def get_onnx_control(self):
        """ONNX 모델을 사용한 제어"""
        observation_values = []
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))
        if np.isinf(self.agent_heading) or np.isnan(self.agent_heading): self.agent_heading = 0.0
        observation_values.append(float(self.agent_heading))
        if np.isinf(self.angular_velocity_y) or np.isnan(self.angular_velocity_y): self.angular_velocity_y = 0.0
        observation_values.append(float(self.angular_velocity_y))
        for i in range(2):
            val = float(self.agent_position[i])
            if np.isinf(val) or np.isnan(val): val = 0.0
            observation_values.append(val)
        for i in range(2):
            val = float(self.current_target_position[i])
            if np.isinf(val) or np.isnan(val): val = 0.0
            observation_values.append(val)
        for i in range(2):
            val = float(self.previous_target_position[i])
            if np.isinf(val) or np.isnan(val): val = 0.0
            observation_values.append(val)
        for i in range(2):
            val = float(self.next_target_position[i])
            if np.isinf(val) or np.isnan(val): val = 0.0
            observation_values.append(val)
        observation_values.append(float(self.previous_moment_input))
        observation_values.append(float(self.previous_force_input))
        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([observation_array, observation_array]).reshape(1, 426)
        
        # 전체 모델 입력 발행
        model_input_msg = Float64MultiArray()
        model_input_msg.data = stacked_input.flatten().astype(float).tolist()
        self.model_input_pub.publish(model_input_msg)
        
        # 각 요소별 개별 토픽 발행
        lidar_msg = Float64MultiArray()
        lidar_msg.data = self.lidar_distances.astype(float).tolist()
        self.lidar_pub.publish(lidar_msg)
        
        heading_msg = Float64()
        heading_msg.data = float(self.agent_heading)
        self.heading_pub.publish(heading_msg)
        
        angular_vel_msg = Float64()
        angular_vel_msg.data = -float(self.angular_velocity_y)
        self.angular_vel_pub.publish(angular_vel_msg)
        
        position_msg = Float64MultiArray()
        position_msg.data = self.agent_position.astype(float).tolist()
        self.position_pub.publish(position_msg)
        
        current_waypoint_msg = Float64MultiArray()
        current_waypoint_msg.data = self.current_target_position.astype(float).tolist()
        self.current_waypoint_pub.publish(current_waypoint_msg)
        
        previous_waypoint_msg = Float64MultiArray()
        previous_waypoint_msg.data = self.previous_target_position.astype(float).tolist()
        self.previous_waypoint_pub.publish(previous_waypoint_msg)
        
        next_waypoint_msg = Float64MultiArray()
        next_waypoint_msg.data = self.next_target_position.astype(float).tolist()
        self.next_waypoint_pub.publish(next_waypoint_msg)
        
        previous_moment_msg = Float64()
        previous_moment_msg.data = float(self.previous_moment_input)
        self.previous_moment_pub.publish(previous_moment_msg)
        
        previous_force_msg = Float64()
        previous_force_msg.data = float(self.previous_force_input)
        self.previous_force_pub.publish(previous_force_msg)
        
        outputs = self.session.run(None, {self.input_name: stacked_input})
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), 0.12)
            angular_velocity = max(min(outputs[4][0][0] * self.w_scale, 1.0), -1.0)
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0
        
        return linear_velocity, angular_velocity












    def update_waypoint_positions(self):
        if len(self.waypoints) == 0:
            self.current_target_position = np.zeros(2)
            self.previous_target_position = np.zeros(2)
            self.next_target_position = np.zeros(2)
            return
        if self.current_target_index < len(self.waypoints):
            self.current_target_position = np.array(self.waypoints[self.current_target_index])
        else:
            self.current_target_position = np.zeros(2)
        if self.current_target_index > 0:
            self.previous_target_position = np.array(self.waypoints[self.current_target_index - 1])
        else:
            self.previous_target_position = np.zeros(2)
        if self.current_target_index + 1 < len(self.waypoints):
            self.next_target_position = np.array(self.waypoints[self.current_target_index + 1])
        else:
            self.next_target_position = self.current_target_position.copy()









    def calculate_thruster_commands(self, linear_velocity, angular_velocity):
        forward_thrust = linear_velocity * self.thrust_scale
        turn_thrust = angular_velocity * self.thrust_scale
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        return left_thrust, right_thrust










    def timer_callback(self):
        left_msg = Float64()
        left_msg.data = self.left_thrust
        self.left_thrust_pub.publish(left_msg)
        right_msg = Float64()
        right_msg.data = self.right_thrust
        self.right_thrust_pub.publish(right_msg)









    def destroy_node(self):
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
        node = VRXONNXControllerV5()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
