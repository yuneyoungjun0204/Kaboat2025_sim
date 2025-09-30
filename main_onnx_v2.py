#!/usr/bin/env python3
"""
VRX ONNX 모델 기반 선박 제어 시스템 v2
- TurtleBot 코드를 VRX 환경에 맞게 수정
- LiDAR, GPS, IMU 데이터를 입력으로 사용
- ROS2를 통해 스러스터 제어
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import time
from collections import deque
from utils import SensorDataManager

class VRXONNXControllerV2(Node):
    """VRX ONNX 모델 기반 제어 노드 v2"""
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v2')
        
        # ONNX 모델 로드
        # self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/Ray-48130414.onnx'
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
        
        # ROS2 퍼블리셔 (스러스터 제어)
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        # 변수 초기화 (VRX 환경에 맞게)
        self.lidar_data = np.zeros(201, dtype=np.float32)  # -100도~100도, 201개
        self.robot_position = np.zeros(2, dtype=np.float32)  # GPS UTM 좌표
        self.robot_forward = np.zeros(2, dtype=np.float32)   # 헤딩 방향
        self.target_position = np.zeros(2, dtype=np.float32) # 웨이포인트
        self.input_vector = np.zeros(211, dtype=np.float32)  # Unity 관측 구조
        self.history = deque(maxlen=2)  # 2번의 211개 데이터
        
        # 스케일링 (TurtleBot과 동일)
        self.v_scale = 0.5    # linear velocity scale
        self.w_scale = -0.5   # angular velocity scale

        # 최근 스러스터 명령 저장용 변수
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        # 기준점 설정 상태
        self.reference_point_set = False
        
        # 웨이포인트 설정
        self.waypoints = [
            [50.0, 0.0],    # 첫 번째 웨이포인트 (동쪽 50m)
            [100.0, 0.0],   # 두 번째 웨이포인트 (동쪽 100m)
            [100.0, 150.0], # 세 번째 웨이포인트 (동쪽 100m, 북쪽 150m)
        ]
        self.current_waypoint_idx = 0

        # 10Hz 주기로 스러스터 제어
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('🚢 VRX ONNX Controller v2 시작!')

    def gps_callback(self, msg):
        """GPS 데이터 콜백 - 로봇 위치 업데이트"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            self.robot_position = np.array([
                gps_data['utm_x'],
                gps_data['utm_y']
            ], dtype=np.float32)
            
            # 첫 번째 GPS 데이터 기준점 설정 확인
            if not self.reference_point_set:
                self.reference_point_set = True
                self.get_logger().info(f'📍 기준점 설정 완료: (0, 0) - 첫 번째 GPS 위치')

    def imu_callback(self, msg):
        """IMU 데이터 콜백 - 로봇 방향 업데이트"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        
        # 헤딩을 방향 벡터로 변환
        heading_rad = np.radians(imu_data['yaw_degrees'])
        self.robot_forward = np.array([
            np.cos(heading_rad),
            np.sin(heading_rad)
        ], dtype=np.float32)

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 - -100도~100도 범위 201개 샘플 저장"""
        lidar_data = self.sensor_manager.process_lidar_data(msg)
        
        # LiDAR 데이터를 -100도~100도 범위로 필터링하고 201개로 리샘플링
        lidar_ranges = lidar_data['ranges']
        lidar_angles = lidar_data['angles']
        
        # -100도 ~ 100도 범위의 LiDAR 데이터 필터링
        angle_mask = (lidar_angles >= np.radians(-100)) & (lidar_angles <= np.radians(100))
        filtered_ranges = lidar_ranges[angle_mask]
        filtered_angles = lidar_angles[angle_mask]
        
        # 201개로 리샘플링
        target_angles = np.linspace(np.radians(-100), np.radians(100), 201)
        self.lidar_data = np.interp(target_angles, filtered_angles, filtered_ranges).astype(np.float32)
        
        # 제어 실행
        self.control_vrx()

    def control_vrx(self):
        """웨이포인트 기반 제어 및 ONNX 모델 실행"""
        # 현재 웨이포인트 설정
        if self.current_waypoint_idx < len(self.waypoints):
            self.target_position = np.array(self.waypoints[self.current_waypoint_idx], dtype=np.float32)
        else:
            self.target_position = np.array(self.waypoints[-1], dtype=np.float32)  # 마지막 웨이포인트 유지

        # Unity 관측 구조에 맞게 입력 벡터 생성 (211개)
        self.input_vector[:201] = self.lidar_data           # LiDAR 거리 (201개)
        self.input_vector[201] = np.degrees(np.arctan2(self.robot_forward[1], self.robot_forward[0]))  # Heading (1개)
        self.input_vector[202] = 0.0                        # YAW rate (1개) - 단순화
        self.input_vector[203:205] = self.robot_position    # 현재 위치 (2개)
        self.input_vector[205:207] = self.target_position   # 현재 웨이포인트 (2개)
        self.input_vector[207:209] = self.target_position   # 다음 웨이포인트 (2개) - 동일
        self.input_vector[209:211] = [0.0, 0.0]            # 이전 명령 (2개) - 단순화

        self.history.append(self.input_vector.copy())
        
        if len(self.history) < 2:
            return  # 2번의 데이터가 모이지 않았으면 대기

        # Stacked 입력 생성 (2번의 211개 데이터 = 422개)
        model_input = np.concatenate([
            self.history[0],  # 첫 번째 211개 데이터
            self.history[1]   # 두 번째 211개 데이터
        ]).astype(np.float32).reshape(1, 422)

        # ONNX 모델 추론
        outputs = self.session.run(None, {self.input_name: model_input})
        
        # TurtleBot 스타일 출력 처리
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), -1) + 0.05
            angular_velocity = max(min(outputs[2][0][0] * self.w_scale, 1.0), -0.15)
        else:
            # outputs[0] 사용 (현재 모델 구조에 맞게)
            if len(outputs) > 0:
                angular_velocity = float(outputs[0][0]) * self.w_scale
                linear_velocity = 0.1  # 기본 전진 속도
            else:
                linear_velocity = 0.0
                angular_velocity = 0.0

        # 스러스터 명령으로 변환 (Unity 스타일)
        self.left_thrust, self.right_thrust = self.calculate_thruster_commands(linear_velocity, angular_velocity)

        # 웨이포인트 도달 확인
        current_pos = self.robot_position
        distance = np.sqrt((current_pos[0] - self.target_position[0])**2 + (current_pos[1] - self.target_position[1])**2)
        if distance < 10.0:  # 10m 이내 도달
            self.current_waypoint_idx += 1
            self.get_logger().info(f'🎯 웨이포인트 {self.current_waypoint_idx-1} 도달! 다음: {self.get_next_waypoint()}')

        self.get_logger().info(
            f"위치: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | "
            f"웨이포인트: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) | "
            f"거리: {distance:.1f}m | "
            f"스러스터: L={self.left_thrust:.1f}, R={self.right_thrust:.1f}"
        )

    def calculate_thruster_commands(self, linear_velocity, angular_velocity):
        """선형/각속도를 스러스터 명령으로 변환"""
        # TurtleBot의 linear/angular velocity를 VRX 스러스터로 변환
        # 단순화된 변환: linear -> 전진, angular -> 선회
        forward_thrust = linear_velocity * 200.0  # 전진 명령
        turn_thrust = angular_velocity * 200.0    # 선회 명령
        
        # 좌우 스러스터 계산
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        
        # 범위 제한
        left_thrust = np.clip(left_thrust, -200.0, 200.0)
        right_thrust = np.clip(right_thrust, -200.0, 200.0)
        
        return left_thrust, right_thrust

    def get_next_waypoint(self):
        """다음 웨이포인트 반환"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
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
        node = VRXONNXControllerV2()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
