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
from sensor_msgs.msg import LaserScan, NavSatFix, Imu, Image
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import time
import cv2
from utils import (
    SensorDataManager,
    MiDaSHybridDepthEstimator,
    BlobDetector,
    MultiTargetTracker,
    Visualizer,
    TrackbarController,
    get_config
)
from utils.mission_gate import GateMission
from utils.mission_circle import CircleMission
from utils.mission_avoid import AvoidMission
from utils.base_mission import MissionStatus


class VRXMissionController(Node):
    """VRX 미션 통합 제어 노드"""

    def __init__(self):
        super().__init__('vrx_mission_controller')

        # Config 로드
        self.config = get_config()

        # ONNX 모델 로드 (Config에서 경로 가져오기)
        self.model_path = self.config.get_model_path()
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

        # 센서 관리자
        self.sensor_manager = SensorDataManager()
        
        # CV Bridge (이미지 처리용)
        self.bridge = CvBridge()
        self.current_image = None
        
        # 부표 탐지 시스템 (Gate/Circle 미션용)
        self.depth_estimator = MiDaSHybridDepthEstimator()
        self.blob_detector = BlobDetector()
        self.tracker = MultiTargetTracker()
        self.visualizer = Visualizer()
        self.trackbar_controller = TrackbarController()
        
        # 탐지 결과
        self.current_detections = []
        self.current_tracks = []
        self.current_depth_map = None
        self.best_red_track = None
        self.best_green_track = None
        
        # ROS2 서브스크라이버 (Config에서 토픽 가져오기)
        qos = self.config.get_qos('sensor_data')
        self.create_subscription(
            LaserScan,
            self.config.get_sensor_topic('lidar'),
            self.lidar_callback,
            qos
        )
        self.create_subscription(
            NavSatFix,
            self.config.get_sensor_topic('gps'),
            self.gps_callback,
            qos
        )
        self.create_subscription(
            Imu,
            self.config.get_sensor_topic('imu'),
            self.imu_callback,
            qos
        )
        self.create_subscription(
            Image,
            self.config.get_topic('sensors', 'camera', 'front_left'),
            self.image_callback,
            qos
        )
        self.waypoint_sub = self.create_subscription(
            Point,
            self.config.get_vrx_topic('waypoint'),
            self.waypoint_callback,
            qos
        )
        
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
        
        # 시각화 창 관리
        self.visualization_windows_open = False
        self.current_mission_type = None
        
        # 타이머
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)  # 30Hz 시각화
        
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
                
                # 첫 미션 창 열기
                self.open_visualization_windows()
    
    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Gate/Circle 미션일 때만 부표 탐지 실행
            if self.visualization_windows_open and self.current_mission is not None:
                mission_type = type(self.current_mission).__name__
                if mission_type in ["GateMission", "CircleMission"]:
                    self.process_buoy_detection()
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
    
    def process_buoy_detection(self):
        """부표 탐지 처리 (Gate/Circle 미션용)"""
        if self.current_image is None:
            return
        
        # 트랙바 파라미터 가져오기
        params = self.trackbar_controller.get_all_parameters()
        
        # 파라미터 업데이트
        self.blob_detector.update_detection_parameters(**params['blob_detector'])
        self.tracker.update_tracking_parameters(**params['tracking'])
        
        # 깊이 추정
        self.current_depth_map = self.depth_estimator.estimate_depth(self.current_image)
        
        # 부표 탐지
        self.current_detections = self.blob_detector.detect_blobs(
            self.current_image,
            self.current_depth_map,
            params['detection']['min_depth_threshold'],
            params['detection']['max_depth_threshold']
        )
        
        # 추적
        self.current_tracks = self.tracker.update(self.current_detections, self.current_depth_map)
        self.best_red_track, self.best_green_track = self.tracker.get_best_tracks()
    
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
            
            # 현재 미션 창 닫기
            self.close_visualization_windows()
            
            # 다음 미션으로 전환
            self.current_mission_index += 1
            
            if self.current_mission_index < len(self.missions):
                self.current_mission = self.missions[self.current_mission_index]
                self.current_mission.start()
                self.get_logger().info(f'🚀 [{self.current_mission.mission_name}] 미션 시작!')
                
                # 새 미션 창 열기
                self.open_visualization_windows()
            else:
                self.get_logger().info('🏁 모든 미션 완료!')
                self.current_mission = None
                self.left_thrust = 0.0
                self.right_thrust = 0.0
        
        # 미션 상태 발행
        self.publish_mission_status()
    
    def open_visualization_windows(self):
        """현재 미션에 맞는 시각화 창 열기"""
        if self.current_mission is None:
            return
        
        mission_type = type(self.current_mission).__name__
        self.current_mission_type = mission_type
        
        if mission_type == "GateMission":
            # Gate Mission: 카메라 뷰 + 부표 인식 + 트랙바
            cv2.namedWindow('Gate Mission - Camera View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gate Mission - Camera View', 960, 540)
            
            cv2.namedWindow('Gate Mission - Depth Map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gate Mission - Depth Map', 640, 360)
            
            # 트랙바 창 생성
            self.trackbar_controller.setup_trackbars()
            
            self.get_logger().info('📷 Gate Mission 카메라 창 + 트랙바 열림')
            
        elif mission_type == "CircleMission":
            # Circle Mission: 카메라 뷰 + 부표 인식 + 트랙바
            cv2.namedWindow('Circle Mission - Camera View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Circle Mission - Camera View', 960, 540)
            
            cv2.namedWindow('Circle Mission - Depth Map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Circle Mission - Depth Map', 640, 360)
            
            # 트랙바 창 생성
            self.trackbar_controller.setup_trackbars()
            
            self.get_logger().info('🔄 Circle Mission 카메라 창 + 트랙바 열림')
            
        elif mission_type == "AvoidMission":
            # Avoid Mission: LiDAR + 경로 시각화
            cv2.namedWindow('Avoid Mission - LiDAR View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Avoid Mission - LiDAR View', 800, 800)
            self.get_logger().info('🚧 Avoid Mission LiDAR 창 열림')
        
        self.visualization_windows_open = True
    
    def close_visualization_windows(self):
        """시각화 창 닫기"""
        if not self.visualization_windows_open:
            return
        
        cv2.destroyAllWindows()
        self.visualization_windows_open = False
        self.get_logger().info('🔒 시각화 창 닫힘')
    
    def visualization_callback(self):
        """시각화 업데이트 (30Hz)"""
        if not self.visualization_windows_open or self.current_mission is None:
            return
        
        mission_type = type(self.current_mission).__name__
        
        if mission_type == "GateMission":
            self.visualize_gate_mission()
        elif mission_type == "CircleMission":
            self.visualize_circle_mission()
        elif mission_type == "AvoidMission":
            self.visualize_avoid_mission()
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.get_logger().info('ESC 키 눌림 - 시각화 중지')
    
    def visualize_gate_mission(self):
        """Gate Mission 시각화 (부표 탐지 포함)"""
        if self.current_image is None:
            return
        
        # main.py와 동일한 시각화
        params = self.trackbar_controller.get_all_parameters()
        
        # 메인 카메라 뷰 (부표 탐지 결과 포함)
        main_image = self.visualizer.visualize_tracking_results(
            self.current_image.copy(),
            self.current_tracks,
            self.current_detections,
            0,  # frame_count
            "navigation",  # control_mode
            "green"  # target_color
        )
        
        # 미션 정보 오버레이
        cv2.putText(main_image, f"Gate Mission", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(main_image, f"Heading: {self.agent_heading:.1f} deg", (10, main_image.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 목표 지점 표시
        if self.current_mission.target_position is not None:
            target = self.current_mission.target_position
            distance = np.linalg.norm(self.agent_position - target)
            cv2.putText(main_image, f"Target Distance: {distance:.1f}m", 
                       (10, main_image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 부표 정보 표시
        if self.best_red_track and self.best_red_track.confidence > 0.3:
            cv2.putText(main_image, f"Red Buoy: {self.best_red_track.depth:.2f}m",
                       (main_image.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.best_green_track and self.best_green_track.confidence > 0.3:
            cv2.putText(main_image, f"Green Buoy: {self.best_green_track.depth:.2f}m",
                       (main_image.shape[1] - 250, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Gate Mission - Camera View', main_image)
        
        # 깊이 맵 표시
        if params['visualization']['show_depth'] and self.current_depth_map is not None:
            depth_image = self.visualizer.visualize_depth_map(
                self.current_depth_map,
                self.current_tracks,
                self.current_detections
            )
            cv2.imshow('Gate Mission - Depth Map', depth_image)
    
    def visualize_circle_mission(self):
        """Circle Mission 시각화 (부표 탐지 포함)"""
        if self.current_image is None:
            return
        
        # main.py와 동일한 시각화
        params = self.trackbar_controller.get_all_parameters()
        
        # 메인 카메라 뷰 (부표 탐지 결과 포함)
        main_image = self.visualizer.visualize_tracking_results(
            self.current_image.copy(),
            self.current_tracks,
            self.current_detections,
            0,  # frame_count
            "approach",  # control_mode
            "green"  # target_color
        )
        
        # 미션 정보 오버레이
        cv2.putText(main_image, f"Circle Mission", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(main_image, f"Heading: {self.agent_heading:.1f} deg", 
                   (10, main_image.shape[0] - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(main_image, f"Radius: {self.current_mission.circle_radius:.1f}m", 
                   (10, main_image.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 선회 상태 표시
        circling_status = "CIRCLING" if self.current_mission.circling_started else "APPROACHING"
        color = (0, 255, 0) if self.current_mission.circling_started else (0, 165, 255)
        cv2.putText(main_image, f"Status: {circling_status}", 
                   (10, main_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 부표 정보 표시
        if self.best_red_track and self.best_red_track.confidence > 0.3:
            cv2.putText(main_image, f"Red: {self.best_red_track.depth:.2f}m",
                       (main_image.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.best_green_track and self.best_green_track.confidence > 0.3:
            cv2.putText(main_image, f"Green: {self.best_green_track.depth:.2f}m",
                       (main_image.shape[1] - 200, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Circle Mission - Camera View', main_image)
        
        # 깊이 맵 표시
        if params['visualization']['show_depth'] and self.current_depth_map is not None:
            depth_image = self.visualizer.visualize_depth_map(
                self.current_depth_map,
                self.current_tracks,
                self.current_detections
            )
            cv2.imshow('Circle Mission - Depth Map', depth_image)
    
    def visualize_avoid_mission(self):
        """Avoid Mission 시각화 (LiDAR 기반)"""
        # 800x800 LiDAR 뷰 캔버스
        canvas = np.zeros((800, 800, 3), dtype=np.uint8)
        center_x, center_y = 400, 400
        scale = 4.0  # 미터당 픽셀
        
        # 미션 정보
        cv2.putText(canvas, "Avoid Mission - LiDAR", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), 2)
        
        # 로봇 중심
        cv2.circle(canvas, (center_x, center_y), 15, (0, 255, 0), -1)
        
        # LiDAR 데이터 시각화
        for i in range(len(self.lidar_distances)):
            angle_deg = i - 100  # -100 ~ +100도
            distance = self.lidar_distances[i]
            
            if distance < self.max_lidar_distance:
                angle_rad = np.radians(angle_deg)
                x = int(center_x + distance * scale * np.sin(angle_rad))
                y = int(center_y - distance * scale * np.cos(angle_rad))
                
                # 거리에 따른 색상 (가까울수록 빨강)
                if distance < 10:
                    color = (0, 0, 255)  # 빨강
                elif distance < 30:
                    color = (0, 165, 255)  # 주황
                else:
                    color = (0, 255, 255)  # 노랑
                
                if 0 <= x < 800 and 0 <= y < 800:
                    cv2.circle(canvas, (x, y), 2, color, -1)
        
        # LOS target 표시
        if isinstance(self.current_mission, AvoidMission) and self.current_mission.los_target is not None:
            los = self.current_mission.los_target
            los_x = int(center_x + los[1] * scale)
            los_y = int(center_y - los[0] * scale)
            if 0 <= los_x < 800 and 0 <= los_y < 800:
                cv2.circle(canvas, (los_x, los_y), 10, (255, 0, 255), 2)
                cv2.line(canvas, (center_x, center_y), (los_x, los_y), (255, 0, 255), 2)
        
        # 제어 모드 표시
        mode = self.current_mission.get_control_mode() if self.current_mission else "UNKNOWN"
        mode_color = (0, 255, 0) if mode == "DIRECT_CONTROL" else (0, 165, 255)
        cv2.putText(canvas, f"Mode: {mode}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # 헤딩 표시
        cv2.putText(canvas, f"Heading: {self.agent_heading:.1f} deg", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 헤딩 화살표
        heading_rad = np.radians(self.agent_heading)
        arrow_len = 50
        end_x = int(center_x + arrow_len * np.sin(heading_rad))
        end_y = int(center_y - arrow_len * np.cos(heading_rad))
        cv2.arrowedLine(canvas, (center_x, center_y), (end_x, end_y), (255, 255, 0), 3)
        
        cv2.imshow('Avoid Mission - LiDAR View', canvas)
    
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
        # 시각화 창 닫기
        self.close_visualization_windows()
        
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
        node = VRXMissionController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

