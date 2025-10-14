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

class VRXSpeedController(Node):

    def __init__(self):
        super().__init__('vrx_speed_controller')
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/Ray-40547322.onnx'
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
        self.motor_value_pub = self.create_publisher(Float64MultiArray, '/vrx/motor_values', 10)

        self.lidar_distances = np.zeros(201, dtype=np.float32)
        self.max_lidar_distance = 100.0
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.current_target_position = np.zeros(2, dtype=np.float32)
        self.previous_target_position = np.zeros(2, dtype=np.float32)
        self.next_target_position = np.zeros(2, dtype=np.float32)
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

        # 이전 observation 저장 (모델 입력용 - 시간적 변화 감지)
        self.previous_observation = np.zeros(213, dtype=np.float32)

        self.waypoints = []
        self.current_target_index = 0
        self.target_position = None
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0

        # Unity 학습 환경과 동일한 스케일링 파라미터
        self.thrust_scale = 800.0  # 최대 추력
        self.angular_velocity_y_scale = 1.0

        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.reference_point_set = False
        self.waypoint_reached = False

        # LiDAR 스케일링 변수
        self.lidar_scale_factor = 1.0
        self.lidar_max_distance = self.max_lidar_distance

        # 직접제어 모드 관련 변수
        self.boat_width = 2.2
        self.boat_height = 50.0
        self.use_direct_control = False

        # LOS guidance 관련 변수
        self.los_delta = 10.0
        self.los_lookahead_min = 30.0
        self.los_lookahead_max = 80.0
        self.los_lookahead_factor = 1.0

        # 스무딩 변수 (Unity의 MoveTowards와 유사)
        self.throttle_smooth_speed = 5.0  # Unity의 currentThrottleSmoothSpeed에 해당
        self.current_throttle1 = 0.0  # Unity의 ship_controller.input.Throttle
        self.current_throttle2 = 0.0  # Unity의 ship_controller.input.Throttle2
        self.dt = 0.01  # 타이머 주기 (Time.deltaTime에 해당)

        # 1차 저주파 필터 변수
        self.filter_alpha = 0.35
        self.filtered_left_thrust = 0.0
        self.filtered_right_thrust = 0.0

        self.timer = self.create_timer(self.dt, self.timer_callback)

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
        if self.agent_heading > 0:
            delta_time = current_time - self.last_angular_velocity_update_time
            if delta_time > 0:
                self.current_angular_acceleration = (current_angular_velocity - self.previous_angular_velocity) / delta_time
        self.previous_angular_velocity = current_angular_velocity
        self.last_angular_velocity_update_time = current_time
        self.angular_velocity_y = min(max(current_angular_velocity[2] * self.angular_velocity_y_scale, -180), 180)

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
                    distance = distance / self.lidar_scale_factor

                idx = int(angle_deg + 100)
                idx = max(0, min(200, idx))
                raw_ranges[idx] = distance

        self.lidar_distances = raw_ranges.astype(np.float32)
        self.control_vrx()

    def calculate_goal_psi(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        goal_psi = np.arctan2(dy, dx)
        return goal_psi

    def calculate_distance_L(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        L = min(self.boat_height, np.sqrt(dx**2 + dy**2))
        return L

    def calculate_range_theta(self, L):
        range_theta = np.arctan2(self.boat_width, L)
        return range_theta

    def calculate_theta_lidar(self, goal_psi, current_psi):
        theta_lidar = goal_psi - current_psi
        return theta_lidar

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def calculate_crosstrack_error(self, current_pos, line_start, line_end):
        line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            return 0.0

        point_vec = np.array([current_pos[0] - line_start[0], current_pos[1] - line_start[1]])
        line_unit = line_vec / line_length
        crosstrack_error = np.linalg.norm(point_vec - np.dot(point_vec, line_unit) * line_unit)

        cross_product = np.cross(line_unit, point_vec)
        if cross_product < 0:
            crosstrack_error = -crosstrack_error

        return crosstrack_error

    def calculate_adaptive_lookahead_distance(self, crosstrack_error):
        abs_crosstrack_error = abs(crosstrack_error)
        adaptive_lookahead = self.los_lookahead_factor / (1.0 + abs_crosstrack_error * 0.1)
        adaptive_lookahead = np.clip(adaptive_lookahead, self.los_lookahead_min, self.los_lookahead_max)
        return adaptive_lookahead

    def calculate_los_guidance_point(self, current_pos, waypoint_start, waypoint_end):
        crosstrack_error = self.calculate_crosstrack_error(current_pos, waypoint_start, waypoint_end)
        adaptive_lookahead = self.calculate_adaptive_lookahead_distance(crosstrack_error)

        line_vec = np.array([waypoint_end[0] - waypoint_start[0], waypoint_end[1] - waypoint_start[1]])
        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            return waypoint_end

        line_unit = line_vec / line_length
        point_vec = np.array([current_pos[0] - waypoint_start[0], current_pos[1] - waypoint_start[1]])
        projection_length = np.dot(point_vec, line_unit)
        projection_point = np.array(waypoint_start) + projection_length * line_unit
        los_point = projection_point + adaptive_lookahead * line_unit

        perpendicular_unit = np.array([line_unit[1], -line_unit[0]])

        if crosstrack_error < 0:
            perpendicular_unit = -perpendicular_unit

        perpendicular_offset = abs(crosstrack_error) * self.los_delta / 10.0
        los_point += perpendicular_offset * perpendicular_unit

        return los_point

    def check_obstacles_in_rectangle_path(self, current_pos, target_pos):
        if len(self.waypoints) > 1 and self.current_target_index < len(self.waypoints) - 1:
            waypoint_start = self.waypoints[self.current_target_index]
            waypoint_end = self.waypoints[self.current_target_index + 1]
            los_target = self.calculate_los_guidance_point(current_pos, waypoint_start, waypoint_end)
        else:
            if len(self.waypoints) > 0:
                if self.current_target_index > 0:
                    waypoint_start = self.waypoints[self.current_target_index - 1]
                else:
                    waypoint_start = [0.0, 0.0]
                waypoint_end = self.waypoints[self.current_target_index]
                los_target = self.calculate_los_guidance_point(current_pos, waypoint_start, waypoint_end)
            else:
                los_target = target_pos

        L = self.calculate_distance_L(current_pos, los_target)
        goal_psi = self.calculate_goal_psi(current_pos, los_target)
        current_psi = np.radians((self.agent_heading + 180) % 360 - 180)
        theta_lidar = self.calculate_theta_lidar(goal_psi, current_psi)
        theta_lidar = -self.normalize_angle(theta_lidar)
        range_theta = self.calculate_range_theta(L)

        check_area_points = []
        obstacle_found = False

        for i in range(-90, 90):
            theta = theta_lidar + np.radians(i)
            theta = self.normalize_angle(theta - np.pi / 2)
            squar_theta = np.pi / 2 - theta
            if abs(i) <= range_theta * 180 / np.pi:
                search_distance = L
            else:
                search_distance = self.boat_width / np.sin(abs(i) * np.pi / 180)

            check_y = current_pos[0] + search_distance * np.cos(theta - theta_lidar + goal_psi + np.pi / 2)
            check_x = current_pos[1] + search_distance * np.sin(theta - theta_lidar + goal_psi + np.pi / 2)
            check_area_points.extend([check_x, check_y])

            lidar_distance = self.get_lidar_distance_at_angle_degrees(np.degrees(theta))
            if lidar_distance > self.max_lidar_distance or lidar_distance < 0.0 or np.isinf(lidar_distance):
                lidar_distance = self.max_lidar_distance

            if lidar_distance < search_distance:
                obstacle_found = True

        L_front = 20.0
        range_theta = self.calculate_range_theta(L_front)
        for i in range(-90, 90):
            theta = np.radians(i)
            theta = self.normalize_angle(theta)
            if abs(i) <= range_theta * 180 / np.pi:
                search_distance = L_front
            else:
                search_distance = self.boat_width / np.sin(abs(i) * np.pi / 180)

            check_y = current_pos[0] + search_distance * np.cos(theta)
            check_x = current_pos[1] + search_distance * np.sin(theta)
            check_area_points.extend([check_x, check_y])

            lidar_distance = self.get_lidar_distance_at_angle_degrees(np.degrees(theta))
            if lidar_distance > self.max_lidar_distance or lidar_distance < 0.0 or np.isinf(lidar_distance):
                lidar_distance = self.max_lidar_distance

            if lidar_distance < search_distance:
                obstacle_found = True

        area_msg = Float64MultiArray()
        area_msg.data = check_area_points
        self.obstacle_check_area_pub.publish(area_msg)

        los_target_msg = Float64MultiArray()
        los_target_msg.data = [current_pos[1] + los_target[1], +current_pos[0] + los_target[0]]
        self.los_target_pub.publish(los_target_msg)

        return obstacle_found, los_target

    def get_lidar_distance_at_angle_degrees(self, angle_deg):
        while angle_deg > 100:
            angle_deg -= 360
        while angle_deg < -100:
            angle_deg += 360

        if -180 <= angle_deg <= 180:
            idx = int(angle_deg + 100)
            idx = max(0, min(200, idx))
            return self.lidar_distances[idx]
        else:
            return self.lidar_max_distance

    def calculate_direct_heading_to_target(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_heading_rad = np.arctan2(dx, dy)
        current_heading_rad = np.radians(self.agent_heading)
        heading_diff_rad = target_heading_rad - current_heading_rad
        heading_diff_rad = -np.arctan2(np.sin(heading_diff_rad), np.cos(heading_diff_rad))
        return heading_diff_rad

    def apply_direct_control(self, current_pos, los_target):
        heading_diff_rad = self.calculate_direct_heading_to_target(current_pos, los_target)
        distance_to_los = np.sqrt((los_target[0] - current_pos[0])**2 + (los_target[1] - current_pos[1])**2)

        moment_input = np.clip(heading_diff_rad / np.pi, -0.7, 0.7)

        if distance_to_los > 20.0:
            force_input = 1.0
        elif distance_to_los > 10.0:
            force_input = 0.6
        else:
            force_input = 0.4

        force_input = force_input * (1.0 - abs(moment_input) * 0.3)
        force_input = np.clip(force_input, 0.1, 1.0)

        return moment_input, force_input

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

        has_obstacles, los_target = self.check_obstacles_in_rectangle_path(current_pos, self.target_position)
        moment_input, force_input = self.get_onnx_control()
        # if has_obstacles:
        #     self.use_direct_control = False
        #     moment_input, force_input = self.get_onnx_control()
        # else:
        #     self.use_direct_control = True
        #     moment_input, force_input = self.apply_direct_control(current_pos,los_target)

        # Unity 학습 환경과 동일한 로직으로 스러스터 명령 계산
        self.left_thrust, self.right_thrust = self.calculate_thruster_commands_unity_style(moment_input, force_input)

        control_output_msg = Float64MultiArray()
        control_output_msg.data = [force_input, moment_input]
        self.control_output_pub.publish(control_output_msg)

        mode_msg = String()
        if self.use_direct_control:
            mode_msg.data = "DIRECT_CONTROL"
        else:
            mode_msg.data = "ONNX_MODEL"
        self.control_mode_pub.publish(mode_msg)

    def get_onnx_control(self):
        observation_values = []
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))
        if np.isinf(self.agent_heading) or np.isnan(self.agent_heading):
            self.agent_heading = 0.0
        observation_values.append(float(self.agent_heading))
        if np.isinf(self.angular_velocity_y) or np.isnan(self.angular_velocity_y):
            self.angular_velocity_y = 0.0
        observation_values.append(float(self.angular_velocity_y))
        for i in range(2):
            val = float(self.agent_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        for i in range(2):
            val = float(self.current_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        for i in range(2):
            val = float(self.previous_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        for i in range(2):
            val = float(self.next_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        observation_values.append(float(self.previous_moment_input))
        observation_values.append(float(self.previous_force_input))
        observation_array = np.array(observation_values, dtype=np.float32)

        # 이전 observation과 현재 observation을 연결 (시간적 변화 감지)
        stacked_input = np.concatenate([self.previous_observation, observation_array]).reshape(1, 426)

        # 다음 스텝을 위해 현재 observation 저장
        self.previous_observation = observation_array.copy()

        model_input_msg = Float64MultiArray()
        model_input_msg.data = stacked_input.flatten().astype(float).tolist()
        self.model_input_pub.publish(model_input_msg)

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
            # Unity C# 코드와 동일: actionBuffers.ContinuousActions[0] = moment, [1] = force
            moment_input = np.clip(outputs[4][0][0], -1.0, 1.0)  # moment_input
            force_input = np.clip(outputs[4][0][1], -0.1, 1.0)   # force_input
        else:
            moment_input = 0.0
            force_input = 0.0

        return moment_input, force_input

    def calculate_thruster_commands_unity_style(self, moment_input, force_input):
        """
        Unity C# 학습 환경과 동일한 로직으로 스러스터 명령 계산

        Args:
            moment_input: 모멘트 입력 (-1 ~ 1)
            force_input: 힘 입력 (-0.1 ~ 1)

        Returns:
            left_thrust, right_thrust: 좌우 스러스터 값
        """
        # 이전 입력값 저장 (다음 observation에 사용)
        self.previous_moment_input = moment_input
        self.previous_force_input = force_input

        # moment_input을 제한하여 targetThrottle 값들이 -1~1 범위를 넘지 않도록 수정
        limited_moment_input = moment_input

        # targetThrottle1 = moment_input + force_input이 -1~1 범위를 넘는 경우 moment_input 조정
        if moment_input + force_input > 1.0:
            limited_moment_input = 1.0 - force_input
        elif moment_input + force_input < -1.0:
            limited_moment_input = -1.0 - force_input

        # targetThrottle2 = -moment_input + force_input이 -1~1 범위를 넘는 경우 moment_input 조정
        if -limited_moment_input + force_input > 1.0:
            limited_moment_input = force_input - 1.0
        elif -limited_moment_input + force_input < -1.0:
            limited_moment_input = force_input + 1.0

        # 최종 moment_input을 -1~1 범위로 클램핑
        limited_moment_input = np.clip(limited_moment_input, -1.0, 1.0)

        # 목표 스로틀 값 계산
        target_throttle1 = limited_moment_input + force_input
        target_throttle2 = -limited_moment_input + force_input

        # Unity의 MoveTowards와 유사하게 부드럽게 변화 (스무딩)
        self.current_throttle1 = self.move_towards(
            self.current_throttle1,
            target_throttle1,
            self.throttle_smooth_speed * self.dt
        )
        self.current_throttle2 = self.move_towards(
            self.current_throttle2,
            target_throttle2,
            self.throttle_smooth_speed * self.dt
        )

        # 스로틀 값을 실제 추력으로 변환
        left_thrust = self.current_throttle1 * self.thrust_scale
        right_thrust = self.current_throttle2 * self.thrust_scale

        # 추력 범위 제한
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)

        # 모터 값 발행 (ROS 토픽)
        motor_msg = Float64MultiArray()
        motor_msg.data = [float(limited_moment_input), float(force_input)]
        self.motor_value_pub.publish(motor_msg)

        return left_thrust, right_thrust

    def move_towards(self, current, target, max_delta):
        """Unity의 Mathf.MoveTowards와 동일한 함수"""
        if abs(target - current) <= max_delta:
            return target
        return current + np.sign(target - current) * max_delta

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
        node = VRXSpeedController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
