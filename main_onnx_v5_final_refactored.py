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
from utils.avoid_control import AvoidanceController


class VRXONNXControllerV5Refactored(Node):
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v5_refactored')
        
        # ONNX 모델 로드
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/Ray-19946289.onnx'
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # 센서 관리자
        self.sensor_manager = SensorDataManager()
        
        # ROS2 서브스크라이버
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan', 
                                self.lidar_callback, 10)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', 
                                self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', 
                                self.imu_callback, 10)
        self.waypoint_sub = self.create_subscription(Point, '/vrx/waypoint', 
                                                     self.waypoint_callback, 10)
        
        # ROS2 퍼블리셔
        self.setup_publishers()
        
        # 센서 데이터
        self.lidar_distances = np.zeros(201, dtype=np.float32)
        self.max_lidar_distance = 100.0
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.agent_position = np.zeros(2, dtype=np.float32)
        
        # 웨이포인트 관리
        self.waypoints = []
        self.current_target_index = 0
        self.target_position = None
        self.waypoint_reached = False
        
        # 제어 파라미터
        self.v_scale = 1.0
        self.w_scale = -1.0
        self.thrust_scale = 800
        self.angular_velocity_y_scale = 1
        self.lidar_scale_factor = 1.0
        
        # 장애물 회피 컨트롤러
        self.avoidance_controller = AvoidanceController(
            boat_width=2.2,
            boat_height=50.0,
            max_lidar_distance=100.0,
            los_delta=10.0,
            los_lookahead_min=30.0,
            los_lookahead_max=80.0,
            filter_alpha=0.35
        )
        
        # 제어 상태
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.use_direct_control = False
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        # IMU 관련
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.reference_point_set = False
        
        # 타이머
        self.timer = self.create_timer(0.01, self.timer_callback)
    
    def setup_publishers(self):
        """ROS2 퍼블리셔 설정"""
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
    
    def waypoint_callback(self, msg):
        """웨이포인트 콜백"""
        new_waypoint = [msg.y, msg.x]
        self.waypoints.append(new_waypoint)
        self.current_target_index = len(self.waypoints) - 1
        self.target_position = np.array(new_waypoint, dtype=np.float32)
        self.waypoint_reached = False
    
    def gps_callback(self, msg):
        """GPS 콜백"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            self.agent_position = np.array([gps_data['utm_y'], gps_data['utm_x']], dtype=np.float32)
            if not self.reference_point_set:
                self.reference_point_set = True
    
    def imu_callback(self, msg):
        """IMU 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.agent_heading = imu_data['yaw_degrees']
        if self.agent_heading < 0:
            self.agent_heading += 360.0
        
        current_time = time.time()
        current_angular_velocity = np.array([msg.angular_velocity.x, 
                                            msg.angular_velocity.y, 
                                            msg.angular_velocity.z])
        
        if self.agent_heading > 0:
            delta_time = current_time - self.last_angular_velocity_update_time
            if delta_time > 0:
                self.current_angular_acceleration = (current_angular_velocity - 
                                                    self.previous_angular_velocity) / delta_time
        
        self.previous_angular_velocity = current_angular_velocity
        self.last_angular_velocity_update_time = current_time
        self.angular_velocity_y = min(max(current_angular_velocity[2] * 
                                         self.angular_velocity_y_scale, -180), 180)
    
    def lidar_callback(self, msg):
        """LiDAR 콜백"""
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        raw_ranges = np.full(201, self.max_lidar_distance, dtype=np.float32)
        
        for i in range(len(ranges)):
            angle_rad = angle_min + i * angle_increment
            angle_deg = np.degrees(angle_rad)
            
            if -100 <= angle_deg <= 100:
                distance = ranges[i]
                if np.isinf(distance) or np.isnan(distance) or distance >= self.max_lidar_distance:
                    distance = self.max_lidar_distance
                else:
                    distance = distance / self.lidar_scale_factor
                
                idx = int(angle_deg + 100)
                idx = max(0, min(200, idx))
                raw_ranges[idx] = distance
        
        self.lidar_distances = raw_ranges.astype(np.float32)
        self.control_vrx()
    
    def get_lidar_distance_at_angle_degrees(self, angle_deg):
        """주어진 각도에서 LiDAR 거리 가져오기"""
        while angle_deg > 100:
            angle_deg -= 360
        while angle_deg < -100:
            angle_deg += 360
        
        if -180 <= angle_deg <= 180:
            idx = int(angle_deg + 100)
            idx = max(0, min(200, idx))
            return self.lidar_distances[idx]
        else:
            return self.max_lidar_distance
    
    def control_vrx(self):
        """VRX 제어 메인 로직"""
        if self.target_position is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return
        
        # 목표까지 거리 확인
        current_pos = self.agent_position
        distance = np.sqrt((current_pos[0] - self.target_position[0])**2 + 
                          (current_pos[1] - self.target_position[1])**2)
        
        # 웨이포인트 도달 확인
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
        
        # LOS target 계산
        los_target = self.avoidance_controller.get_los_target(
            current_pos, self.waypoints, self.current_target_index
        )
        
        # 장애물 확인 및 제어
        self.use_direct_control, linear_velocity, angular_velocity, check_area_points = \
            self.avoidance_controller.check_obstacles_and_get_control(
                current_pos, los_target, self.agent_heading, self.lidar_distances,
                self.get_lidar_distance_at_angle_degrees, self.get_onnx_control
            )
        
        # 필터 적용
        filtered_linear, filtered_angular = self.avoidance_controller.apply_filters(
            linear_velocity, angular_velocity
        )
        
        # 이전 명령 업데이트
        self.previous_moment_input = filtered_angular
        self.previous_force_input = filtered_linear
        
        # 스러스터 계산
        raw_left_thrust, raw_right_thrust = self.calculate_thruster_commands(
            filtered_linear, filtered_angular
        )
        
        # 스러스터 필터 적용
        self.left_thrust, self.right_thrust = self.avoidance_controller.apply_thrust_filters(
            raw_left_thrust, raw_right_thrust
        )
        
        # 시각화 데이터 발행
        self.publish_visualization_data(check_area_points, los_target, filtered_linear, filtered_angular)
    
    def get_onnx_control(self):
        """ONNX 모델 제어"""
        current_target, previous_target, next_target = self.get_waypoint_positions()
        
        observation_values = []
        
        # LiDAR 데이터
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))
        
        # 헤딩
        if np.isinf(self.agent_heading) or np.isnan(self.agent_heading):
            self.agent_heading = 0.0
        observation_values.append(float(self.agent_heading))
        
        # 각속도
        if np.isinf(self.angular_velocity_y) or np.isnan(self.angular_velocity_y):
            self.angular_velocity_y = 0.0
        observation_values.append(float(self.angular_velocity_y))
        
        # 위치 및 웨이포인트
        for val in [self.agent_position, current_target, previous_target, next_target]:
            for i in range(2):
                v = float(val[i])
                if np.isinf(v) or np.isnan(v):
                    v = 0.0
                observation_values.append(v)
        
        # 이전 입력
        observation_values.append(float(self.previous_moment_input))
        observation_values.append(float(self.previous_force_input))
        
        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([observation_array, observation_array]).reshape(1, 426)
        
        # 모델 입력 발행
        self.publish_model_inputs(stacked_input, current_target, previous_target, next_target)
        
        # ONNX 추론
        outputs = self.session.run(None, {self.input_name: stacked_input})
        
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), 0.12)
            angular_velocity = max(min(outputs[4][0][0] * self.w_scale, 1.0), -1.0)
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0
        
        return linear_velocity, angular_velocity
    
    def get_waypoint_positions(self):
        """웨이포인트 위치 반환"""
        if len(self.waypoints) == 0:
            zeros = np.zeros(2, dtype=np.float32)
            return zeros, zeros, zeros
        
        if self.current_target_index < len(self.waypoints):
            current_target = np.array(self.waypoints[self.current_target_index], dtype=np.float32)
        else:
            current_target = np.zeros(2, dtype=np.float32)
        
        if self.current_target_index > 0:
            previous_target = np.array(self.waypoints[self.current_target_index - 1], dtype=np.float32)
        else:
            previous_target = np.zeros(2, dtype=np.float32)
        
        if self.current_target_index + 1 < len(self.waypoints):
            next_target = np.array(self.waypoints[self.current_target_index + 1], dtype=np.float32)
        else:
            next_target = current_target.copy()
        
        return current_target, previous_target, next_target
    
    def calculate_thruster_commands(self, linear_velocity, angular_velocity):
        """스러스터 명령 계산"""
        forward_thrust = linear_velocity * self.thrust_scale
        turn_thrust = angular_velocity * self.thrust_scale
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        return left_thrust, right_thrust
    
    def publish_visualization_data(self, check_area_points, los_target, linear_vel, angular_vel):
        """시각화 데이터 발행"""
        # 체크 영역
        area_msg = Float64MultiArray()
        area_msg.data = check_area_points
        self.obstacle_check_area_pub.publish(area_msg)
        
        # LOS target
        los_target_msg = Float64MultiArray()
        los_target_msg.data = [self.agent_position[1] + los_target[1], 
                              self.agent_position[0] + los_target[0]]
        self.los_target_pub.publish(los_target_msg)
        
        # 제어 출력
        control_output_msg = Float64MultiArray()
        control_output_msg.data = [linear_vel, angular_vel]
        self.control_output_pub.publish(control_output_msg)
        
        # 제어 모드
        mode_msg = String()
        mode_msg.data = "DIRECT_CONTROL" if self.use_direct_control else "ONNX_MODEL"
        self.control_mode_pub.publish(mode_msg)
    
    def publish_model_inputs(self, stacked_input, current_target, previous_target, next_target):
        """모델 입력 데이터 발행"""
        # 전체 모델 입력
        model_input_msg = Float64MultiArray()
        model_input_msg.data = stacked_input.flatten().astype(float).tolist()
        self.model_input_pub.publish(model_input_msg)
        
        # 개별 요소
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
        current_waypoint_msg.data = current_target.astype(float).tolist()
        self.current_waypoint_pub.publish(current_waypoint_msg)
        
        previous_waypoint_msg = Float64MultiArray()
        previous_waypoint_msg.data = previous_target.astype(float).tolist()
        self.previous_waypoint_pub.publish(previous_waypoint_msg)
        
        next_waypoint_msg = Float64MultiArray()
        next_waypoint_msg.data = next_target.astype(float).tolist()
        self.next_waypoint_pub.publish(next_waypoint_msg)
        
        previous_moment_msg = Float64()
        previous_moment_msg.data = float(self.previous_moment_input)
        self.previous_moment_pub.publish(previous_moment_msg)
        
        previous_force_msg = Float64()
        previous_force_msg.data = float(self.previous_force_input)
        self.previous_force_pub.publish(previous_force_msg)
    
    def timer_callback(self):
        """타이머 콜백 - 스러스터 명령 발행"""
        left_msg = Float64()
        left_msg.data = self.left_thrust
        self.left_thrust_pub.publish(left_msg)
        
        right_msg = Float64()
        right_msg.data = self.right_thrust
        self.right_thrust_pub.publish(right_msg)
    
    def destroy_node(self):
        """노드 종료"""
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
        node = VRXONNXControllerV5Refactored()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

