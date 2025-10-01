#!/usr/bin/env python3
"""
VRX 순차 미션 시스템
- 사용자가 미션 순서를 지정
- 각 미션을 30초씩 순차 실행
- 간단한 타이머 기반 전환
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
    TrackbarController
)
from utils.mission_gate import GateMission
from utils.mission_circle import CircleMission
from utils.mission_avoid import AvoidMission


class SequentialMissionController(Node):
    """순차 미션 제어 노드 - 타이머 기반"""
    
    def __init__(self):
        super().__init__('sequential_mission_controller')
        
        # ==================== 미션 설정 영역 ====================
        # 여기서 미션 순서와 시간을 지정하세요!
        self.mission_config = [
            {'type': 'gate', 'duration': 30, 'waypoints': [[50, 0], [100, 0]]},
            {'type': 'circle', 'duration': 30, 'waypoints': [[150, 0], [200, 0]], 'radius': 10.0},
            {'type': 'avoid', 'duration': 30, 'waypoints': [[250, 0], [300, 0], [350, 0]]},
            {'type': 'gate', 'duration': 30, 'waypoints': [[400, 0], [450, 0]]},
        ]
        # =======================================================
        
        self.get_logger().info('🚀 순차 미션 시스템 시작!')
        self.get_logger().info(f'총 {len(self.mission_config)}개 미션 구성됨')
        
        # ONNX 모델 로드
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/Ray-19946289.onnx'
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # 센서 관리자
        self.sensor_manager = SensorDataManager()
        
        # CV Bridge
        self.bridge = CvBridge()
        self.current_image = None
        
        # 부표 탐지 시스템
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
        
        # ROS2 서브스크라이버
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan', 
                                self.lidar_callback, 10)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', 
                                self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', 
                                self.imu_callback, 10)
        self.create_subscription(Image, '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
                                self.image_callback, 10)
        
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
        
        # 미션 관리
        self.missions = []
        self.current_mission_index = 0
        self.current_mission = None
        self.mission_start_time = None
        self.current_mission_duration = 30  # 기본 30초
        
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
        
        # 미션 구성
        self.configure_missions_from_config()
        
        # 타이머
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        self.mission_check_timer = self.create_timer(0.1, self.check_mission_timeout)
        
        # 첫 미션 시작
        if len(self.missions) > 0:
            self.start_mission(0)
    
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
    
    def configure_missions_from_config(self):
        """설정에서 미션 생성"""
        for idx, config in enumerate(self.mission_config):
            mission_type = config['type']
            waypoints = config['waypoints']
            duration = config.get('duration', 30)
            
            if mission_type == 'gate':
                mission = GateMission(
                    waypoints=waypoints,
                    thrust_scale=self.thrust_scale,
                    completion_threshold=15.0
                )
                self.missions.append({
                    'mission': mission,
                    'duration': duration,
                    'type': 'gate'
                })
                self.get_logger().info(f'✅ 미션 {idx+1}: Gate ({duration}초)')
                
            elif mission_type == 'circle':
                radius = config.get('radius', 10.0)
                direction = config.get('direction', 'clockwise')
                mission = CircleMission(
                    waypoints=waypoints,
                    circle_radius=radius,
                    circle_direction=direction,
                    thrust_scale=self.thrust_scale,
                    completion_threshold=15.0
                )
                self.missions.append({
                    'mission': mission,
                    'duration': duration,
                    'type': 'circle'
                })
                self.get_logger().info(f'✅ 미션 {idx+1}: Circle ({duration}초, 반경 {radius}m)')
                
            elif mission_type == 'avoid':
                mission = AvoidMission(
                    waypoints=waypoints,
                    onnx_control_func=self.get_onnx_control,
                    get_lidar_distance_func=self.get_lidar_distance_at_angle_degrees,
                    thrust_scale=self.thrust_scale,
                    completion_threshold=15.0
                )
                self.missions.append({
                    'mission': mission,
                    'duration': duration,
                    'type': 'avoid'
                })
                self.get_logger().info(f'✅ 미션 {idx+1}: Avoid ({duration}초)')
    
    def start_mission(self, mission_index):
        """미션 시작"""
        if mission_index >= len(self.missions):
            self.get_logger().info('🏁 모든 미션 완료!')
            self.current_mission = None
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            self.close_visualization_windows()
            return
        
        mission_info = self.missions[mission_index]
        self.current_mission = mission_info['mission']
        self.current_mission_duration = mission_info['duration']
        self.current_mission_index = mission_index
        
        # 미션 시작
        self.current_mission.start()
        self.mission_start_time = time.time()
        
        # 로그
        elapsed_str = f"{mission_index+1}/{len(self.missions)}"
        self.get_logger().info(f'🚀 [{self.current_mission.mission_name}] 미션 시작! ({elapsed_str}) - {self.current_mission_duration}초')
        
        # 시각화 창 열기
        self.open_visualization_windows()
    
    def check_mission_timeout(self):
        """미션 타임아웃 체크 (30초 경과 시 다음 미션으로)"""
        if self.current_mission is None or self.mission_start_time is None:
            return
        
        elapsed_time = time.time() - self.mission_start_time
        remaining_time = self.current_mission_duration - elapsed_time
        
        # 매 초마다 남은 시간 출력
        if int(elapsed_time) != int(elapsed_time - 0.1):
            self.get_logger().info(f'⏱️  [{self.current_mission.mission_name}] 남은 시간: {remaining_time:.1f}초')
        
        # 시간 초과 시 다음 미션으로
        if elapsed_time >= self.current_mission_duration:
            self.get_logger().info(f'⏰ [{self.current_mission.mission_name}] 시간 종료!')
            
            # 창 닫기
            self.close_visualization_windows()
            
            # 다음 미션으로
            self.start_mission(self.current_mission_index + 1)
    
    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Gate/Circle 미션일 때만 부표 탐지
            if self.visualization_windows_open and self.current_mission is not None:
                mission_type = type(self.current_mission).__name__
                if mission_type in ["GateMission", "CircleMission"]:
                    self.process_buoy_detection()
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
    
    def process_buoy_detection(self):
        """부표 탐지 처리"""
        if self.current_image is None:
            return
        
        params = self.trackbar_controller.get_all_parameters()
        self.blob_detector.update_detection_parameters(**params['blob_detector'])
        self.tracker.update_tracking_parameters(**params['tracking'])
        
        self.current_depth_map = self.depth_estimator.estimate_depth(self.current_image)
        self.current_detections = self.blob_detector.detect_blobs(
            self.current_image, self.current_depth_map,
            params['detection']['min_depth_threshold'],
            params['detection']['max_depth_threshold']
        )
        
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
        """LiDAR 거리 조회"""
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
        """미션 제어"""
        if self.current_mission is None:
            self.left_thrust = 0.0
            self.right_thrust = 0.0
            return
        
        # 현재 미션 업데이트
        if isinstance(self.current_mission, AvoidMission):
            self.left_thrust, self.right_thrust = self.current_mission.update(
                self.agent_position, self.agent_heading, self.lidar_distances
            )
            viz_data = self.current_mission.get_visualization_data()
            self.publish_visualization_data(viz_data)
        elif isinstance(self.current_mission, (GateMission, CircleMission)):
            self.left_thrust, self.right_thrust = self.current_mission.update(
                self.agent_position, self.agent_heading
            )
        
        # 미션 상태 발행
        self.publish_mission_status()
    
    def open_visualization_windows(self):
        """시각화 창 열기"""
        if self.current_mission is None:
            return
        
        mission_type = type(self.current_mission).__name__
        self.current_mission_type = mission_type
        
        if mission_type == "GateMission":
            cv2.namedWindow('Gate Mission - Camera View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gate Mission - Camera View', 960, 540)
            cv2.namedWindow('Gate Mission - Depth Map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gate Mission - Depth Map', 640, 360)
            self.trackbar_controller.setup_trackbars()
            self.get_logger().info('📷 Gate Mission 창 열림')
            
        elif mission_type == "CircleMission":
            cv2.namedWindow('Circle Mission - Camera View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Circle Mission - Camera View', 960, 540)
            cv2.namedWindow('Circle Mission - Depth Map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Circle Mission - Depth Map', 640, 360)
            self.trackbar_controller.setup_trackbars()
            self.get_logger().info('🔄 Circle Mission 창 열림')
            
        elif mission_type == "AvoidMission":
            cv2.namedWindow('Avoid Mission - LiDAR View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Avoid Mission - LiDAR View', 800, 800)
            self.get_logger().info('🚧 Avoid Mission 창 열림')
        
        self.visualization_windows_open = True
    
    def close_visualization_windows(self):
        """시각화 창 닫기"""
        if not self.visualization_windows_open:
            return
        
        cv2.destroyAllWindows()
        self.visualization_windows_open = False
        self.get_logger().info('🔒 시각화 창 닫힘')
    
    def visualization_callback(self):
        """시각화 업데이트"""
        if not self.visualization_windows_open or self.current_mission is None:
            return
        
        mission_type = type(self.current_mission).__name__
        
        if mission_type == "GateMission":
            self.visualize_gate_mission()
        elif mission_type == "CircleMission":
            self.visualize_circle_mission()
        elif mission_type == "AvoidMission":
            self.visualize_avoid_mission()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.get_logger().info('ESC - 시각화 중지')
    
    def visualize_gate_mission(self):
        """Gate Mission 시각화"""
        if self.current_image is None:
            return
        
        params = self.trackbar_controller.get_all_parameters()
        main_image = self.visualizer.visualize_tracking_results(
            self.current_image.copy(), self.current_tracks, self.current_detections,
            0, "navigation", "green"
        )
        
        # 타이머 표시
        if self.mission_start_time:
            elapsed = time.time() - self.mission_start_time
            remaining = self.current_mission_duration - elapsed
            cv2.putText(main_image, f"Time: {remaining:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        cv2.putText(main_image, f"Gate Mission", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Gate Mission - Camera View', main_image)
        
        if params['visualization']['show_depth'] and self.current_depth_map is not None:
            depth_image = self.visualizer.visualize_depth_map(
                self.current_depth_map, self.current_tracks, self.current_detections
            )
            cv2.imshow('Gate Mission - Depth Map', depth_image)
    
    def visualize_circle_mission(self):
        """Circle Mission 시각화"""
        if self.current_image is None:
            return
        
        params = self.trackbar_controller.get_all_parameters()
        main_image = self.visualizer.visualize_tracking_results(
            self.current_image.copy(), self.current_tracks, self.current_detections,
            0, "approach", "green"
        )
        
        # 타이머 표시
        if self.mission_start_time:
            elapsed = time.time() - self.mission_start_time
            remaining = self.current_mission_duration - elapsed
            cv2.putText(main_image, f"Time: {remaining:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        cv2.putText(main_image, f"Circle Mission", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Circle Mission - Camera View', main_image)
        
        if params['visualization']['show_depth'] and self.current_depth_map is not None:
            depth_image = self.visualizer.visualize_depth_map(
                self.current_depth_map, self.current_tracks, self.current_detections
            )
            cv2.imshow('Circle Mission - Depth Map', depth_image)
    
    def visualize_avoid_mission(self):
        """Avoid Mission 시각화"""
        canvas = np.zeros((800, 800, 3), dtype=np.uint8)
        center_x, center_y = 400, 400
        scale = 4.0
        
        # 타이머 표시
        if self.mission_start_time:
            elapsed = time.time() - self.mission_start_time
            remaining = self.current_mission_duration - elapsed
            cv2.putText(canvas, f"Time: {remaining:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        cv2.putText(canvas, "Avoid Mission", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        
        cv2.circle(canvas, (center_x, center_y), 15, (0, 255, 0), -1)
        
        # LiDAR 데이터
        for i in range(len(self.lidar_distances)):
            angle_deg = i - 100
            distance = self.lidar_distances[i]
            
            if distance < self.max_lidar_distance:
                angle_rad = np.radians(angle_deg)
                x = int(center_x + distance * scale * np.sin(angle_rad))
                y = int(center_y - distance * scale * np.cos(angle_rad))
                
                if distance < 10:
                    color = (0, 0, 255)
                elif distance < 30:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 255)
                
                if 0 <= x < 800 and 0 <= y < 800:
                    cv2.circle(canvas, (x, y), 2, color, -1)
        
        cv2.imshow('Avoid Mission - LiDAR View', canvas)
    
    def get_onnx_control(self):
        """ONNX 모델 제어"""
        if self.current_mission is None or not isinstance(self.current_mission, AvoidMission):
            return 0.0, 0.0
        
        current_target, previous_target, next_target = self.current_mission.get_waypoint_positions()
        
        observation_values = []
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))
        
        if np.isinf(self.agent_heading) or np.isnan(self.agent_heading):
            self.agent_heading = 0.0
        observation_values.append(float(self.agent_heading))
        
        if np.isinf(self.angular_velocity_y) or np.isnan(self.angular_velocity_y):
            self.angular_velocity_y = 0.0
        observation_values.append(float(self.angular_velocity_y))
        
        for val in [self.agent_position, current_target, previous_target, next_target]:
            for i in range(2):
                v = float(val[i])
                if np.isinf(v) or np.isnan(v):
                    v = 0.0
                observation_values.append(v)
        
        observation_values.append(float(self.current_mission.previous_moment_input))
        observation_values.append(float(self.current_mission.previous_force_input))
        
        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([observation_array, observation_array]).reshape(1, 426)
        
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
        if 'check_area_points' in viz_data:
            area_msg = Float64MultiArray()
            area_msg.data = viz_data['check_area_points']
            self.obstacle_check_area_pub.publish(area_msg)
        
        if 'control_mode' in viz_data:
            mode_msg = String()
            mode_msg.data = viz_data['control_mode']
            self.control_mode_pub.publish(mode_msg)
    
    def publish_mission_status(self):
        """미션 상태 발행"""
        if self.current_mission is not None:
            status_msg = String()
            elapsed = time.time() - self.mission_start_time if self.mission_start_time else 0
            remaining = self.current_mission_duration - elapsed
            status_msg.data = f"{self.current_mission.mission_name} ({self.current_mission_index + 1}/{len(self.missions)}) - {remaining:.0f}s"
            self.mission_status_pub.publish(status_msg)
    
    def timer_callback(self):
        """타이머 콜백"""
        left_msg = Float64()
        left_msg.data = self.left_thrust
        self.left_thrust_pub.publish(left_msg)
        
        right_msg = Float64()
        right_msg.data = self.right_thrust
        self.right_thrust_pub.publish(right_msg)
    
    def destroy_node(self):
        """노드 종료"""
        self.close_visualization_windows()
        
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
        node = SequentialMissionController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

