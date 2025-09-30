#!/usr/bin/env python3
"""
VRX ONNX 모델 기반 선박 제어 시스템 v4
- Unity CollectObservations와 동일한 관찰값 구조
- LiDAR 거리 (201개) + 헤딩 + 각속도 + 위치 + 웨이포인트들 + 이전 명령
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Point
import time
from collections import deque
from utils import SensorDataManager

class VRXONNXControllerV4(Node):
    """VRX ONNX 모델 기반 제어 노드 v4 - Unity 관찰값 구조"""
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v4')
        
        # ONNX 모델 로드
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-3076146.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-10721680.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-12499862.onnx'
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-14248543.onnx'
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-676612.onnx'
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

        # Unity 관찰값 구조에 맞는 변수들
        self.lidar_distances = np.zeros(201, dtype=np.float32)  # LiDAR 거리 (201개)
        self.agent_heading = 0.0                                # 에이전트 헤딩 (Y rotation)
        self.angular_velocity_y = 0.0                           # IMU 각속도 (Y축)
        
        # 위치 관련 변수들 (Unity 좌표계: X=동서, Z=남북)
        self.agent_position = np.zeros(2, dtype=np.float32)     # 현재 에이전트 위치 (X, Z)
        self.current_target_position = np.zeros(2, dtype=np.float32)  # 현재 웨이포인트 (X, Z)
        self.previous_target_position = np.zeros(2, dtype=np.float32) # 이전 웨이포인트 (X, Z)
        self.next_target_position = np.zeros(2, dtype=np.float32)     # 다음 웨이포인트 (X, Z)
        
        # 이전 명령 저장
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0
        
        # 웨이포인트 관리
        self.waypoints = []  # 웨이포인트 리스트
        self.current_target_index = 0
        self.target_position = None
        
        # 각속도 추적용 변수 (Unity와 동일)
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0
        self.current_angular_acceleration = np.zeros(3)
        
        # 스케일링 (TurtleBot과 동일)
        self.v_scale = 0.4   # linear velocity scale
        self.w_scale = -0.6   # angular velocity scale

        # 최근 스러스터 명령 저장용 변수
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        # 기준점 설정 상태
        self.reference_point_set = False
        
        # 웨이포인트 도달 상태
        self.waypoint_reached = False

        # 10Hz 주기로 스러스터 제어
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('🚢 VRX ONNX Controller v4 시작! (Unity 관찰값 구조)')
        self.get_logger().info('📍 웨이포인트를 robot_visualizer에서 클릭하여 설정하세요')

    def waypoint_callback(self, msg):
        """웨이포인트 콜백 - robot_visualizer에서 클릭한 점을 받음"""
        new_waypoint = [msg.y, msg.x]
        self.waypoints.append(new_waypoint)
        self.current_target_index = len(self.waypoints) - 1
        self.target_position = np.array(new_waypoint, dtype=np.float32)  # 좌표 일치시키기
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

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 - Unity와 동일한 201개 거리 데이터"""
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
        
        # 제어 실행
        self.control_vrx()

    def control_vrx(self):
        """Unity 관찰값 구조 기반 제어 및 ONNX 모델 실행"""
        # 웨이포인트가 없으면 모터 정지
        if self.target_position is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            self.get_logger().debug('웨이포인트 없음 - 모터 정지')
            return

        # 웨이포인트 도달 확인
        current_pos = self.agent_position
        distance = np.sqrt((current_pos[0] - self.target_position[0])**2 + (current_pos[1] - self.target_position[1])**2)
        
        # 디버깅 로그 추가
        self.get_logger().debug(f'현재위치: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | 목표: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) | 거리: {distance:.1f}m')
        
        if distance < 15.0:  # 5m 이내 도달
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
                    self.get_logger().info(f'🎯 다음 웨이포인트로 이동: ({next_waypoint[0]:.1f}, {next_waypoint[1]:.1f})')
                else:
                    # 모든 웨이포인트 완료
                    self.target_position = None
                    self.get_logger().info('🏁 모든 웨이포인트 완료! 정지합니다.')
            # 도달했으면 모터 정지
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return

        # 웨이포인트 위치 업데이트
        self.update_waypoint_positions()

        # Unity 관찰값 구조에 맞게 입력 벡터 생성
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
        
        # 4. 현재 에이전트 위치 (2개: X, Z) - 무한대 값 방지
        for i in range(2):
            val = float(self.agent_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 5. 현재 웨이포인트 위치 (2개: X, Z) - 무한대 값 방지
        for i in range(2):
            val = float(self.current_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 6. 이전 웨이포인트 위치 (2개: X, Z) - 무한대 값 방지
        for i in range(2):
            val = float(self.previous_target_position[i])
            if np.isinf(val) or np.isnan(val):
                val = 0.0
            observation_values.append(val)
        
        # 7. 다음 웨이포인트 위치 (2개: X, Z) - 무한대 값 방지
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
        ]).reshape(1, 426)

        # ONNX 모델 추론
        outputs = self.session.run(None, {self.input_name: stacked_input})
        
        # TurtleBot 스타일 출력 처리
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), 0.05)
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
            f"위치: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | "
            f"웨이포인트: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) | "
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
        node = VRXONNXControllerV4()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
