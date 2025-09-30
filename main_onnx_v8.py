#!/usr/bin/env python3
"""
VRX ONNX 모델 기반 선박 제어 시스템 v8
- 모든 기능을 utils/Autonomous.py로 모듈화
- 간단하고 깔끔한 메인 파일
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point
from utils import SensorDataManager
from utils.Autonomous import (
    CoordinateTransformer, ObstacleDetector, DirectNavigationController,
    ONNXModelController, ThrusterController, WaypointManager, SensorDataProcessor
)

class VRXONNXControllerV8(Node):
    """VRX ONNX 모델 기반 제어 노드 v8 - 모듈화된 Autonomous 클래스 사용"""
    
    def __init__(self):
        super().__init__('vrx_onnx_controller_v8')
        
        # 좌표계 변환기 초기화
        self.coord_transformer = CoordinateTransformer()
        
        # 센서 데이터 관리자
        self.sensor_manager = SensorDataManager()
        
        # 모듈화된 클래스들 초기화
        self.obstacle_detector = ObstacleDetector(self.coord_transformer)
        self.direct_navigation = DirectNavigationController(self.coord_transformer)
        self.thruster_controller = ThrusterController()
        self.waypoint_manager = WaypointManager(self.coord_transformer)
        self.sensor_processor = SensorDataProcessor(self.coord_transformer, self.sensor_manager)
        
        # ONNX 모델 컨트롤러
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-12499862.onnx'
        self.onnx_controller = ONNXModelController(self.model_path, self.coord_transformer)
        self.get_logger().info(f"ONNX Model Loaded: {self.model_path}")
        
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
        
        # v5 모드 정보 퍼블리셔
        self.mode_pub = self.create_publisher(String, '/vrx/current_mode', 10)
        
        # goal_check 영역 정보 퍼블리셔
        self.goal_check_pub = self.create_publisher(Float64MultiArray, '/vrx/goal_check_areas', 10)

        # 최근 스러스터 명령 저장용 변수
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        # v5 추가: 직접 경로 모드 플래그
        self.use_direct_navigation = False

        # 10Hz 주기로 스러스터 제어
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('🚢 VRX ONNX Controller v8 시작! (모듈화된 Autonomous 클래스 사용)')
        self.get_logger().info('📍 웨이포인트를 robot_visualizer에서 클릭하여 설정하세요')

    def waypoint_callback(self, msg):
        """웨이포인트 콜백 - robot_visualizer에서 클릭한 점을 받음 (Gazebo ENU 좌표계)"""
        gazebo_waypoint = np.array([msg.x, msg.y])
        unity_waypoint = self.waypoint_manager.add_waypoint(gazebo_waypoint)
        
        self.get_logger().info(f'🎯 새 웨이포인트 추가: Gazebo({msg.x:.1f}, {msg.y:.1f}) → Unity({unity_waypoint[0]:.1f}, {unity_waypoint[1]:.1f}) - 총 {len(self.waypoint_manager.waypoints)}개')

    def gps_callback(self, msg):
        """GPS 데이터 콜백 - 로봇 위치 업데이트 (좌표계 변환 포함)"""
        reference_set = self.sensor_processor.process_gps_data(msg)
        if reference_set:
            self.get_logger().info(f'📍 기준점 설정 완료: Gazebo({self.sensor_processor.agent_position_gazebo[0]:.1f}, {self.sensor_processor.agent_position_gazebo[1]:.1f}) → Unity({self.sensor_processor.agent_position_unity[0]:.1f}, {self.sensor_processor.agent_position_unity[1]:.1f})')

    def imu_callback(self, msg):
        """IMU 데이터 콜백 - 헤딩과 각속도 업데이트 (회전 방향 변환 포함)"""
        self.sensor_processor.process_imu_data(msg)

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 - 좌표계 변환 포함 (Body-fixed → Unity NED)"""
        self.sensor_processor.process_lidar_data(msg)
        
        # 제어 실행
        self.control_vrx()

    def control_vrx(self):
        """Unity 관찰값 구조 기반 제어 및 ONNX 모델 실행 + v5 직접 경로 모드 (회전 방향 및 X축 기준 차이 고려)"""
        # 웨이포인트가 없으면 모터 정지
        if self.waypoint_manager.target_position is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            self.get_logger().debug('웨이포인트 없음 - 모터 정지')
            
            # 모드 정보 발행
            mode_msg = String()
            mode_msg.data = "STOP"
            self.mode_pub.publish(mode_msg)
            return

        # 웨이포인트 도달 확인 (Unity NED 좌표계 기준)
        current_pos_unity = self.sensor_processor.agent_position_unity
        reached, next_waypoint = self.waypoint_manager.check_waypoint_reached(current_pos_unity)
        
        if reached:
            if next_waypoint is not None:
                self.get_logger().info(f'🎯 다음 웨이포인트로 이동: Unity({next_waypoint[0]:.1f}, {next_waypoint[1]:.1f})')
            else:
                self.get_logger().info('🏁 모든 웨이포인트 완료! 정지합니다.')
            
            # 도달했으면 모터 정지
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            
            # 모드 정보 발행
            mode_msg = String()
            mode_msg.data = "REACHED"
            self.mode_pub.publish(mode_msg)
            return

        # 거리 계산
        distance = np.sqrt((current_pos_unity[0] - self.waypoint_manager.target_position[0])**2 + (current_pos_unity[1] - self.waypoint_manager.target_position[1])**2)
        
        # 디버깅 로그 추가
        self.get_logger().debug(f'현재위치 Unity: ({current_pos_unity[0]:.1f}, {current_pos_unity[1]:.1f}) | 목표 Unity: ({self.waypoint_manager.target_position[0]:.1f}, {self.waypoint_manager.target_position[1]:.1f}) | 거리: {distance:.1f}m')

        # v5 추가: 전방 범위에 장애물이 있는지 확인 (기존 방식)
        has_obstacles_forward = self.obstacle_detector.check_obstacles_in_forward_path(
            self.sensor_processor.lidar_distances
        )
        
        # v5 추가: 목적지까지 경로에 장애물이 있는지 확인 (Unity NED 좌표계 기준)
        dx = self.waypoint_manager.target_position[0] - current_pos_unity[0]  # X 방향 차이 (동서)
        dy = self.waypoint_manager.target_position[1] - current_pos_unity[1]  # Y 방향 차이 (남북)
        goal_psi = np.degrees(np.arctan2(dx, dy))  # Unity NED 좌표계에서 목적지 방향 각도
        goal_psi = self.coord_transformer.normalize_angle_0_360(int(goal_psi))
        
        is_able, area_info = self.obstacle_detector.goal_check(
            distance, goal_psi, current_pos_unity, 
            self.sensor_processor.lidar_distances_360, 
            self.sensor_processor.agent_heading
        )
        has_obstacles_path = not is_able
        
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
        
        # v5 추가: 전방 장애물이 없으면 직접 경로 모드 사용
        if not has_obstacles_forward and distance > 20.0:  # 20m 이상 떨어져 있을 때만
            self.use_direct_navigation = True
            self.get_logger().info('🛤️ 직접 경로 모드 활성화 (경로에 장애물 없음)')
            
            # 모드 정보 발행
            mode_msg = String()
            mode_msg.data = "DIRECT_CLEAR"
            self.mode_pub.publish(mode_msg)
            
            # atan을 이용한 직접 헤딩 계산 (Unity NED 좌표계, 회전 방향 고려)
            heading_diff_rad = -self.direct_navigation.calculate_direct_heading(
                current_pos_unity, self.waypoint_manager.target_position, self.sensor_processor.agent_heading
            )
            
            # 선형 속도 최대 고정
            linear_velocity = 0.3  # 최대 속도 고정
            
            # 각속도 (헤딩 차이에 비례)
            angular_velocity = np.clip(heading_diff_rad / np.pi, -0.3, 0.3)
            
            # 스러스터 명령으로 변환
            self.left_thrust, self.right_thrust = self.thruster_controller.calculate_thruster_commands(linear_velocity, angular_velocity)
            
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
                self.get_logger().info('🚧 ONNX 모델 모드 (전방 범위 장애물 감지)')
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
        current_target, previous_target, next_target = self.waypoint_manager.update_waypoint_positions()

        # Unity 관찰값 구조에 맞게 입력 벡터 생성 (Unity NED 좌표계 기준)
        observation_values = self.sensor_processor.create_observation_vector(current_target, previous_target, next_target)

        # ONNX 모델로 제어 명령 예측
        linear_velocity, angular_velocity = self.onnx_controller.predict_control(observation_values)

        # 이전 명령 업데이트
        self.sensor_processor.update_previous_commands(linear_velocity, angular_velocity)

        # 스러스터 명령으로 변환
        self.left_thrust, self.right_thrust = self.thruster_controller.calculate_thruster_commands(linear_velocity, angular_velocity)

        # trajectory_viz.py로 출력값 전송을 위한 퍼블리셔
        if not hasattr(self, 'control_output_pub'):
            self.control_output_pub = self.create_publisher(Float64MultiArray, '/vrx/control_output', 10)
        
        # 제어 출력값 발행 [linear_velocity, angular_velocity]
        output_msg = Float64MultiArray()
        output_msg.data = [linear_velocity, angular_velocity]
        self.control_output_pub.publish(output_msg)

        self.get_logger().info(
            f"ONNX모델: 위치 Unity: ({current_pos_unity[0]:.1f}, {current_pos_unity[1]:.1f}) | "
            f"웨이포인트 Unity: ({self.waypoint_manager.target_position[0]:.1f}, {self.waypoint_manager.target_position[1]:.1f}) | "
            f"거리: {distance:.1f}m | "
            f"제어값: Linear={linear_velocity:.3f}, Angular={angular_velocity:.3f} | "
            f"스러스터: L={self.left_thrust:.1f}, R={self.right_thrust:.1f}"
        )

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
        node = VRXONNXControllerV8()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
