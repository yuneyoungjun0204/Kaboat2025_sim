#!/usr/bin/env python3
"""
VRX 미션 통합 시스템
- 여러 미션을 순차적으로 실행
- 웨이포인트 기반 미션 전환
- Gate → Circle → Avoid 순서로 진행
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point
import time
from utils import SensorDataManager
from utils.mission_gate import GateMission
from utils.mission_circle import CircleMission
from utils.mission_avoid import AvoidMission
from utils.base_mission import MissionStatus


class VRXMissionController(Node):
    """VRX 미션 통합 제어 노드"""
    
    def __init__(self):
        super().__init__('vrx_mission_controller')
        
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
        
        # 제어 파라미터
        self.v_scale = 1.0
        self.w_scale = -1.0
        self.thrust_scale = 800
        self.angular_velocity_y_scale = 1
        self.lidar_scale_factor = 1.0
        
        # 웨이포인트 수집 (trajectory_viz.py에서 클릭으로 추가)
        self.collected_waypoints = []
        self.waypoint_collection_mode = True  # 웨이포인트 수집 모드
        
        # 미션 리스트
        self.missions = []
        self.current_mission_index = 0
        self.current_mission = None
        
        # 미션 설정 대기
        self.missions_configured = False
        
        # 제어 상태
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        # IMU 관련
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.reference_point_set = False
        
        # 타이머
        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.get_logger().info('🚀 VRX 미션 통합 시스템 시작!')
        self.get_logger().info('📍 웨이포인트를 클릭하여 미션을 설정하세요.')
        self.get_logger().info('   - 처음 2개: Gate Mission')
        self.get_logger().info('   - 다음 2개: Circle Mission')
        self.get_logger().info('   - 그 다음: Avoid Mission')
    
    def setup_publishers(self):
        """ROS2 퍼블리셔 설정"""
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.model_input_pub = self.create_publisher(Float64MultiArray, '/vrx/model_input', 10)
        self.lidar_pub = self.create_publisher(Float64MultiArray, '/vrx/lidar_data', 10)
        self.heading_pub = self.create_publisher(Float64, '/vrx/agent_heading', 10)
        self.angular_vel_pub = self.create_publisher(Float64, '/vrx/angular_velocity', 10)
        self.position_pub = self.create_publisher(Float64MultiArray, '/vrx/agent_position', 10)
        self.control_output_pub = self.create_publisher(Float64MultiArray, '/vrx/control_output', 10)
        self.control_mode_pub = self.create_publisher(String, '/vrx/control_mode', 10)
        self.obstacle_check_area_pub = self.create_publisher(Float64MultiArray, '/vrx/obstacle_check_area', 10)
        self.los_target_pub = self.create_publisher(Float64MultiArray, '/vrx/los_target', 10)
        self.mission_status_pub = self.create_publisher(String, '/vrx/mission_status', 10)
        self.current_waypoint_pub = self.create_publisher(Float64MultiArray, '/vrx/current_waypoint', 10)
        self.previous_waypoint_pub = self.create_publisher(Float64MultiArray, '/vrx/previous_waypoint', 10)
        self.next_waypoint_pub = self.create_publisher(Float64MultiArray, '/vrx/next_waypoint', 10)
        self.previous_moment_pub = self.create_publisher(Float64, '/vrx/previous_moment', 10)
        self.previous_force_pub = self.create_publisher(Float64, '/vrx/previous_force', 10)
    
    def waypoint_callback(self, msg):
        """웨이포인트 콜백 - 수집 모드에서 미션 설정"""
        new_waypoint = [msg.y, msg.x]
        self.collected_waypoints.append(new_waypoint)
        
        waypoint_count = len(self.collected_waypoints)
        self.get_logger().info(f'📍 웨이포인트 {waypoint_count} 추가: ({msg.y:.1f}, {msg.x:.1f})')
        
        # 웨이포인트가 충분히 모이면 미션 구성
        if not self.missions_configured:
            self.try_configure_missions()
    
    def try_configure_missions(self):
        """수집된 웨이포인트로 미션 구성 시도"""
        waypoint_count = len(self.collected_waypoints)
        
        # 최소 6개 웨이포인트 필요 (Gate 2개 + Circle 2개 + Avoid 2개)
        if waypoint_count >= 6:
            self.get_logger().info('🎯 미션 구성 중...')
            
            # 1. Gate Mission (처음 2개 웨이포인트)
            gate_waypoints = self.collected_waypoints[0:2]
            gate_mission = GateMission(
                waypoints=gate_waypoints,
                thrust_scale=self.thrust_scale,
                completion_threshold=15.0
            )
            self.missions.append(gate_mission)
            self.get_logger().info(f'✅ Gate Mission 구성: {len(gate_waypoints)}개 웨이포인트')
            
            # 2. Circle Mission (다음 2개 웨이포인트)
            circle_waypoints = self.collected_waypoints[2:4]
            circle_mission = CircleMission(
                waypoints=circle_waypoints,
                circle_radius=10.0,
                circle_direction='clockwise',
                thrust_scale=self.thrust_scale,
                completion_threshold=15.0
            )
            self.missions.append(circle_mission)
            self.get_logger().info(f'✅ Circle Mission 구성: {len(circle_waypoints)}개 웨이포인트')
            
            # 3. Avoid Mission (나머지 웨이포인트)
            avoid_waypoints = self.collected_waypoints[4:]
            avoid_mission = AvoidMission(
                waypoints=avoid_waypoints,
                onnx_control_func=self.get_onnx_control,
                get_lidar_distance_func=self.get_lidar_distance_at_angle_degrees,
                thrust_scale=self.thrust_scale,
                completion_threshold=15.0
            )
            self.missions.append(avoid_mission)
            self.get_logger().info(f'✅ Avoid Mission 구성: {len(avoid_waypoints)}개 웨이포인트')
            
            # 미션 구성 완료
            self.missions_configured = True
            self.waypoint_collection_mode = False
            
            # 첫 번째 미션 시작
            if len(self.missions) > 0:
                self.current_mission = self.missions[0]
                self.current_mission.start()
                self.get_logger().info(f'🚀 [{self.current_mission.mission_name}] 미션 시작!')
    
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
        self.control_missions()
    
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
    
    def control_missions(self):
        """미션 제어 메인 로직"""
        # 미션이 구성되지 않았으면 대기
        if not self.missions_configured:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return
        
        # 현재 미션이 없으면 대기
        if self.current_mission is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return
        
        # 현재 미션 업데이트
        if isinstance(self.current_mission, AvoidMission):
            # Avoid 미션은 LiDAR 데이터 필요
            self.left_thrust, self.right_thrust = self.current_mission.update(
                self.agent_position, self.agent_heading, self.lidar_distances
            )
            
            # 시각화 데이터 발행
            viz_data = self.current_mission.get_visualization_data()
            self.publish_visualization_data(viz_data)
        elif isinstance(self.current_mission, (GateMission, CircleMission)):
            # Gate/Circle 미션은 위치와 헤딩만 필요
            self.left_thrust, self.right_thrust = self.current_mission.update(
                self.agent_position, self.agent_heading
            )
        
        # 미션 완료 확인
        if self.current_mission.is_completed():
            self.get_logger().info(f'🎉 [{self.current_mission.mission_name}] 미션 완료!')
            
            # 다음 미션으로 전환
            self.current_mission_index += 1
            
            if self.current_mission_index < len(self.missions):
                self.current_mission = self.missions[self.current_mission_index]
                self.current_mission.start()
                self.get_logger().info(f'🚀 [{self.current_mission.mission_name}] 미션 시작!')
            else:
                self.get_logger().info('🏁 모든 미션 완료!')
                self.current_mission = None
                self.left_thrust = 0.0
                self.right_thrust = 0.0
        
        # 미션 상태 발행
        self.publish_mission_status()
    
    def get_onnx_control(self):
        """ONNX 모델 제어 (Avoid 미션용)"""
        if self.current_mission is None or not isinstance(self.current_mission, AvoidMission):
            return 0.0, 0.0
        
        current_target, previous_target, next_target = self.current_mission.get_waypoint_positions()
        
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
        observation_values.append(float(self.current_mission.previous_moment_input))
        observation_values.append(float(self.current_mission.previous_force_input))
        
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
    
    def publish_visualization_data(self, viz_data):
        """시각화 데이터 발행"""
        # 체크 영역
        if 'check_area_points' in viz_data:
            area_msg = Float64MultiArray()
            area_msg.data = viz_data['check_area_points']
            self.obstacle_check_area_pub.publish(area_msg)
        
        # LOS target
        if 'los_target' in viz_data and viz_data['los_target'] is not None:
            los_target_msg = Float64MultiArray()
            los_target = viz_data['los_target']
            los_target_msg.data = [self.agent_position[1] + los_target[1], 
                                  self.agent_position[0] + los_target[0]]
            self.los_target_pub.publish(los_target_msg)
        
        # 제어 출력
        if 'linear_velocity' in viz_data and 'angular_velocity' in viz_data:
            control_output_msg = Float64MultiArray()
            control_output_msg.data = [viz_data['linear_velocity'], viz_data['angular_velocity']]
            self.control_output_pub.publish(control_output_msg)
        
        # 제어 모드
        if 'control_mode' in viz_data:
            mode_msg = String()
            mode_msg.data = viz_data['control_mode']
            self.control_mode_pub.publish(mode_msg)
    
    def publish_mission_status(self):
        """미션 상태 발행"""
        if self.current_mission is not None:
            status_msg = String()
            status_msg.data = f"{self.current_mission.mission_name} ({self.current_mission_index + 1}/{len(self.missions)})"
            self.mission_status_pub.publish(status_msg)
    
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
        
        if self.current_mission is not None and isinstance(self.current_mission, AvoidMission):
            previous_moment_msg = Float64()
            previous_moment_msg.data = float(self.current_mission.previous_moment_input)
            self.previous_moment_pub.publish(previous_moment_msg)
            
            previous_force_msg = Float64()
            previous_force_msg.data = float(self.current_mission.previous_force_input)
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
        node = VRXMissionController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

