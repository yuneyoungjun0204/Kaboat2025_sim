#!/usr/bin/env python3
"""
VRX 로봇 제어 시스템 메인 파일 - 위치 기반 모드 전환 버전
지정한 UTM 좌표에 도달하면 Control Mode를 approach로 자동 전환
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
import time
import numpy as np
import math

# 모듈화된 컴포넌트들 import
from utils import (
    MiDaSHybridDepthEstimator,
    BlobDetector,
    MultiTargetTracker,
    NavigationController,
    ThrusterController,
    Visualizer,
    TrackbarController
)

class VRXRobotController(Node):
    """VRX 로봇 제어 메인 노드 - 위치 기반 모드 전환 버전"""
    
    def __init__(self):
        super().__init__('vrx_robot_controller_mode_switch')
        
        # ==================== 목표 위치 설정 ====================
        # 이 위치에 도달하면 approach 모드로 전환 (UTM 좌표)
        self.target_utm_position = np.array([70.0, 40.0])  # [North, East] UTM 좌표
        self.mode_switch_distance = 10.0  # 목표 위치로부터 이 거리 이내면 모드 전환 (미터)
        self.mode_switched = False  # 모드 전환 완료 플래그
        # =======================================================
        
        # 현재 UTM 위치 (시작점을 0,0으로 설정)
        self.current_utm_position = np.array([0.0, 0.0])
        
        # UTM 변환을 위한 기준점 (시작 GPS 위치를 기준으로 설정)
        self.utm_origin_lat = None  # 시작 시 GPS로 설정
        self.utm_origin_lon = None  # 시작 시 GPS로 설정
        self.gps_initialized = False  # GPS 초기화 완료 플래그
        
        # 모드 전환 상태
        self.force_approach_mode = False  # 강제 approach 모드 활성화 플래그
        
        # 마지막 명령 저장 (approach 모드에서 인식 끊김 시 사용)
        self.last_left_cmd = 0.0
        self.last_right_cmd = 0.0
        self.last_approach_status = "대기 중"
        
        # Approach 모드 전용 트랙바 설정
        self.setup_approach_trackbars()
        
        # 초기화
        self.bridge = CvBridge()
        
        # 모듈화된 컴포넌트들 초기화
        self.depth_estimator = MiDaSHybridDepthEstimator()
        self.blob_detector = BlobDetector()
        self.tracker = MultiTargetTracker()
        self.navigation_controller = NavigationController()
        self.thruster_controller = ThrusterController(self)
        self.visualizer = Visualizer()
        self.trackbar_controller = TrackbarController()
        
        # ROS2 서브스크라이버
        self.image_sub = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
            self.image_callback,
            10)
        
        # GPS 서브스크라이버 (실제 UTM 좌표 받기)
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/wamv/sensors/gps/gps/fix',
            self.gps_callback,
            10)
        
        # ROS2 퍼블리셔 (추적 데이터만 직접 관리)
        self.tracking_pub = self.create_publisher(Float32MultiArray, '/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions', 10)
        
        # 위치 시뮬레이션용 타이머 (GPS 데이터가 없을 때만 사용)
        self.create_timer(0.1, self.update_position_simulation)
        
        # 상태 변수
        self.frame_count = 0
        self.start_time = time.time()
        
        self.get_logger().info('🚀 VRX 로봇 제어 시스템 시작! (위치 기반 모드 전환 버전)')
        self.get_logger().info(f'🎯 목표 UTM 위치: ({self.target_utm_position[0]:.1f}, {self.target_utm_position[1]:.1f})')
        self.get_logger().info(f'📏 모드 전환 거리: {self.mode_switch_distance}m')
        self.get_logger().info('📍 시작 GPS 위치를 UTM (0, 0) 기준점으로 설정합니다.')
    
    def gps_callback(self, msg):
        """GPS 콜백 - 실제 UTM 좌표 업데이트"""
        lat = msg.latitude
        lon = msg.longitude
        
        # 첫 번째 GPS 데이터로 기준점 설정 (UTM 0,0 기준)
        if not self.gps_initialized:
            self.utm_origin_lat = lat
            self.utm_origin_lon = lon
            self.gps_initialized = True
            self.get_logger().info(f'📍 GPS 기준점 설정: ({lat:.6f}, {lon:.6f}) → UTM (0, 0)')
        
        # 기준점을 0,0으로 하는 UTM 변환
        utm_x = (lon - self.utm_origin_lon) * 111320 * math.cos(math.radians(lat))
        utm_y = (lat - self.utm_origin_lat) * 110540
        
        self.current_utm_position = np.array([utm_y, utm_x])  # [North, East]
        
        # 목표 위치까지 거리 체크
        self.check_mode_switch_condition()
    
    def update_position_simulation(self):
        """위치 시뮬레이션 (GPS 데이터가 없을 때 사용)"""
        if not self.mode_switched:
            # 목표 방향으로 조금씩 이동 (초당 2m)
            direction = self.target_utm_position - self.current_utm_position
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                unit_direction = direction / distance
                # 0.1초에 0.2m 이동
                self.current_utm_position += unit_direction * 0.2
            
            # 목표 위치까지 거리 체크
            self.check_mode_switch_condition()
    
    def check_mode_switch_condition(self):
        """모드 전환 조건 확인"""
        distance_to_target = np.linalg.norm(self.target_utm_position - self.current_utm_position)
        
        if distance_to_target <= self.mode_switch_distance and not self.mode_switched:
            self.mode_switched = True
            self.force_approach_mode = True  # 강제 approach 모드 활성화
            self.get_logger().info(f'🔄 목표 위치 도달! Approach 모드를 강제 활성화합니다. (거리: {distance_to_target:.2f}m)')
            
            # 상태 메시지
            self.thruster_controller.publish_status("🔄 목표 위치 도달 - Approach 모드 강제 활성화")
    
    def image_callback(self, msg):
        """이미지 콜백 함수 - 핵심 처리 로직만 유지"""
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        # 트랙바에서 모든 파라미터 읽기
        params = self.trackbar_controller.get_all_parameters()
        
        # 강제 approach 모드가 활성화되면 control_mode를 approach로 덮어쓰기
        if self.force_approach_mode:
            params['control']['control_mode'] = 'approach'
        
        # 파라미터 업데이트
        self.update_parameters(params)
        
        # 핵심 처리 파이프라인
        depth_map = self.depth_estimator.estimate_depth(cv_image)
        
        # 일반 부표 감지 (빨강, 초록)
        detections = self.blob_detector.detect_blobs(
            cv_image, 
            depth_map, 
            params['detection']['min_depth_threshold'], 
            params['detection']['max_depth_threshold']
        )
        
        # 검정색 부표 감지 (approach 모드에서만)
        black_detections = []
        if self.force_approach_mode:
            approach_params = self.get_approach_parameters()
            max_depth = approach_params['max_depth']
            black_detections = self.detect_black_buoys(cv_image, depth_map, max_depth)
        
        # 모든 감지 결과 합치기
        all_detections = detections + black_detections
        
        tracks = self.tracker.update(all_detections, depth_map)
        best_red, best_green = self.tracker.get_best_tracks()
        
        # 검정색 부표 추적 결과 가져오기
        best_black = self.get_best_black_track(tracks)
        
        # 제어 처리 - 항상 실행 (모드 전환은 트랙바가 자동 처리)
        self.process_control(best_red, best_green, best_black, params)
        
        # 추적 데이터 퍼블리시
        self.publish_tracking_data(best_red, best_green)
        
        # 시각화 (위치 정보 추가)
        self.visualize_results_with_position(cv_image, tracks, detections, depth_map, params)
        
        # 성능 모니터링
        self.monitor_performance()
    
    def update_parameters(self, params):
        """파라미터 업데이트"""
        # Blob Detector 파라미터
        self.blob_detector.update_detection_parameters(**params['blob_detector'])
        
        # Tracker 파라미터
        self.tracker.update_tracking_parameters(**params['tracking'])
        
        # Navigation Controller 파라미터
        nav_params = params['navigation'].copy()
        nav_params.update({
            'steering_kp': params['pid']['steering_kp'],
            'approach_kp': params['pid']['approach_kp']
        })
        self.navigation_controller.update_control_parameters(**nav_params)
    
    def process_control(self, best_red, best_green, best_black, params):
        """제어 처리 - 모듈화된 로직"""
        control_mode = params['control']['control_mode']
        target_color = params['control']['target_color']
        rotation_direction = params['control']['rotation_direction']
        
        if control_mode == "navigation":
            # 부표 간 네비게이션
            if best_red and best_green and best_red.confidence > 0.3 and best_green.confidence > 0.3:
                left_cmd, right_cmd, error, steering, forward, turn = self.navigation_controller.navigation_control(
                    best_red.center[0], best_green.center[0],
                    best_red.center[1], best_green.center[1]
                )
                status = f"🧭 네비게이션: 중점({(best_red.center[0] + best_green.center[0])/2:.1f}), 오차: {error:.1f}"
            else:
                left_cmd = right_cmd = 150.0
                status = "⚠️ 부표 미탐지: 천천히 직진"
        
        elif control_mode == "approach":
            # Approach 모드 전용 파라미터 사용
            approach_params = self.get_approach_parameters()
            target_color = approach_params['target_color']
            rotation_direction = approach_params['rotation_direction']
            
            # 객체 접근 제어
            if target_color == "green":
                target_track = best_green
            elif target_color == "red":
                target_track = best_red
            else:  # target_color == "black"
                target_track = best_black
            
            if target_track and target_track.confidence > 0.3:
                # NavigationController의 파라미터를 Approach 전용 값으로 업데이트
                self.navigation_controller.base_speed = approach_params['base_speed']
                self.navigation_controller.min_speed = approach_params['min_speed']
                self.navigation_controller.max_turn_thrust = approach_params['max_turn_thrust']
                self.navigation_controller.pid_kp = approach_params['pid_kp']
                
                # Approach 제어 실행
                left_cmd, right_cmd, error, steering, forward, turn, mode, target_x = self.navigation_controller.approach_control(
                    target_track.center[0], target_track.center[1], target_track.depth or 0.0,
                    approach_distance=approach_params['approach_distance'],
                    slow_distance=approach_params['slow_distance'],
                    stop_distance=approach_params['stop_distance'],
                    rotation_direction=rotation_direction
                )
                
                # 마지막 명령 저장
                self.last_left_cmd = left_cmd
                self.last_right_cmd = right_cmd
                
                # 목표 X값 퍼블리시
                self.thruster_controller.publish_target_x(target_x)
                
                direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
                status = f"🎯 접근: {target_color.upper()}({target_track.center[0]:.1f}), 깊이: {target_track.depth:.3f}m, 방향: {direction_name}"
                self.last_approach_status = status
            else:
                # 인식이 끊겼을 때 마지막 명령 유지
                left_cmd = self.last_left_cmd
                right_cmd = self.last_right_cmd
                status = f"🔄 {target_color} 객체 미탐지: 마지막 명령 유지 ({self.last_approach_status})"
        
        # 스러스터 명령 퍼블리시
        self.thruster_controller.publish_thrust_commands(left_cmd, right_cmd)
        self.thruster_controller.publish_status(status)
        
        # 로그 출력 (1초마다)
        if self.frame_count % 10 == 0:
            self.get_logger().info(status)
    
    def setup_approach_trackbars(self):
        """Approach 모드 전용 트랙바 설정"""
        # 트랙바 창 생성
        cv2.namedWindow("Object Approach Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Approach Control", 400, 300)
        
        # 트랙바 생성
        cv2.createTrackbar("Target_Color", "Object Approach Control", 3, 3, self.nothing)  # 1: 초록, 2: 빨강, 3: 검정
        cv2.createTrackbar("Rotation_Direction", "Object Approach Control", 1, 2, self.nothing)  # 1: 시계방향, 2: 반시계방향
        cv2.createTrackbar("Base_Speed", "Object Approach Control", 150, 300, self.nothing)
        cv2.createTrackbar("Min_Speed", "Object Approach Control", 50, 200, self.nothing)
        cv2.createTrackbar("Max_Turn_Thrust", "Object Approach Control", 150, 250, self.nothing)
        cv2.createTrackbar("Approach_Distance", "Object Approach Control", 5, 15, self.nothing)  # 0.01-0.15m
        cv2.createTrackbar("Slow_Distance", "Object Approach Control", 3, 15, self.nothing)  # 0.01-0.15m
        cv2.createTrackbar("Stop_Distance", "Object Approach Control", 2, 15, self.nothing)  # 0.01-0.15m
        cv2.createTrackbar("PID_Kp", "Object Approach Control", 8, 50, self.nothing)  # 0.8-5.0
        cv2.createTrackbar("Max_Depth", "Object Approach Control", 100, 1500, self.nothing)  # 최대 깊이 (cm)
        
        # 초기값 설정 (잘 되었던 값들)
        cv2.setTrackbarPos("Target_Color", "Object Approach Control", 3)  # 1: 초록색, 2: 빨간색, 3: 검정색
        cv2.setTrackbarPos("Rotation_Direction", "Object Approach Control", 1)  # 1: 시계방향, 2: 반시계방향
        cv2.setTrackbarPos("Base_Speed", "Object Approach Control", 150)  # 기본 속도 150
        cv2.setTrackbarPos("Min_Speed", "Object Approach Control", 50)  # 최소 속도 50
        cv2.setTrackbarPos("Max_Turn_Thrust", "Object Approach Control", 150)
        cv2.setTrackbarPos("Approach_Distance", "Object Approach Control", 3)  # 0.05m
        cv2.setTrackbarPos("Slow_Distance", "Object Approach Control", 4)  # 0.03m
        cv2.setTrackbarPos("Stop_Distance", "Object Approach Control", 7)  # 0.02m
        cv2.setTrackbarPos("PID_Kp", "Object Approach Control", 8)  # Kp = 0.8
        cv2.setTrackbarPos("Max_Depth", "Object Approach Control", 100)  # 최대 깊이 100cm
        
        self.get_logger().info('✅ Approach 모드 전용 트랙바 설정 완료')
    
    def nothing(self, x):
        """트랙바 콜백 함수 (빈 함수)"""
        pass
    
    def get_approach_parameters(self):
        """Approach 모드 전용 파라미터 읽기"""
        try:
            # 트랙바에서 값 읽기
            target_color_idx = cv2.getTrackbarPos("Target_Color", "Object Approach Control")
            rotation_dir = cv2.getTrackbarPos("Rotation_Direction", "Object Approach Control")
            base_speed = cv2.getTrackbarPos("Base_Speed", "Object Approach Control")
            min_speed = cv2.getTrackbarPos("Min_Speed", "Object Approach Control")
            max_turn_thrust = cv2.getTrackbarPos("Max_Turn_Thrust", "Object Approach Control")
            approach_dist = cv2.getTrackbarPos("Approach_Distance", "Object Approach Control") / 100.0  # 0.01-0.15m
            slow_dist = cv2.getTrackbarPos("Slow_Distance", "Object Approach Control") / 100.0  # 0.01-0.15m
            stop_dist = cv2.getTrackbarPos("Stop_Distance", "Object Approach Control") / 100.0  # 0.01-0.15m
            pid_kp = cv2.getTrackbarPos("PID_Kp", "Object Approach Control") / 10.0  # 0.8-5.0
            max_depth = cv2.getTrackbarPos("Max_Depth", "Object Approach Control") / 100.0  # 0.5-2.5m
            
            # 색상 변환
            if target_color_idx == 1:
                target_color = "green"
            elif target_color_idx == 2:
                target_color = "red"
            else:  # target_color_idx == 3
                target_color = "black"
            
            # 회전 방향 변환
            rotation_direction = 1 if rotation_dir == 1 else -1
            
            return {
                'target_color': target_color,
                'rotation_direction': rotation_direction,
                'base_speed': base_speed,
                'min_speed': min_speed,
                'max_turn_thrust': max_turn_thrust,
                'approach_distance': approach_dist,
                'slow_distance': slow_dist,
                'stop_distance': stop_dist,
                'pid_kp': pid_kp,
                'max_depth': max_depth
            }
        except:
            # 트랙바가 없으면 기본값 반환
            return {
                'target_color': 'black',
                'rotation_direction': 1,
                'base_speed': 150,
                'min_speed': 50,
                'max_turn_thrust': 150,
                'approach_distance': 0.03,
                'slow_distance': 0.04,
                'stop_distance': 0.07,
                'pid_kp': 0.8,
                'max_depth': 1.0
            }
    
    def detect_black_buoys(self, image, depth_map, max_depth=1.0):
        """검정색 부표 감지 (깊이 필터링 포함)"""
        detections = []
        
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 검정색 범위 정의 (HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])  # V값이 낮은 것이 검정색
        
        # 검정색 마스크 생성
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 최소 면적 필터링
                # 바운딩 박스 계산
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 깊이 정보 가져오기
                depth_value = self._get_depth_at_point(depth_map, center_x, center_y)
                
                # 깊이 필터링: 최대 깊이 이하만 허용
                if depth_value > 0 and depth_value <= max_depth:
                    # Detection 객체 생성
                    detection = type('Detection', (), {
                        'center': (center_x, center_y),
                        'color': 'black',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'depth': depth_value,
                        'confidence': min(area / 1000.0, 1.0)  # 면적 기반 신뢰도
                    })()
                    
                    detections.append(detection)
        
        return detections
    
    def _get_depth_at_point(self, depth_map, x, y):
        """특정 점에서의 깊이 값 가져오기"""
        if depth_map is not None and 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return depth_map[y, x]
        return 0.0
    
    def get_best_black_track(self, tracks):
        """검정색 부표 중 가장 좋은 추적 결과 반환"""
        black_tracks = [track for track in tracks if hasattr(track, 'color') and track.color == 'black']
        
        if not black_tracks:
            return None
        
        # 신뢰도가 가장 높은 추적 결과 반환
        best_track = max(black_tracks, key=lambda t: t.confidence)
        return best_track if best_track.confidence > 0.3 else None
    
    def publish_tracking_data(self, best_red, best_green):
        """추적 데이터 퍼블리시"""
        tracking_msg = Float32MultiArray()
        
        # 데이터 형식: [red_x, red_y, red_depth, green_x, green_y, green_depth, timestamp]
        red_x = float(best_red.center[0]) if best_red and best_red.confidence > 0.3 else 0.0
        red_y = float(best_red.center[1]) if best_red and best_red.confidence > 0.3 else 0.0
        red_depth = float(best_red.depth) if best_red and best_red.depth is not None else 0.0
        
        green_x = float(best_green.center[0]) if best_green and best_green.confidence > 0.3 else 0.0
        green_y = float(best_green.center[1]) if best_green and best_green.confidence > 0.3 else 0.0
        green_depth = float(best_green.depth) if best_green and best_green.depth is not None else 0.0
        
        tracking_msg.data = [red_x, red_y, red_depth, green_x, green_y, green_depth, time.time()]
        self.tracking_pub.publish(tracking_msg)
    
    def visualize_results_with_position(self, image, tracks, detections, depth_map, params):
        """시각화 - 위치 정보 추가"""
        # 메인 추적 결과 시각화
        main_image = self.visualizer.visualize_tracking_results(
            image, tracks, detections, self.frame_count,
            params['control']['control_mode'], params['control']['target_color']
        )
        
        # 위치 정보 오버레이
        h, w = main_image.shape[:2]
        
        # 현재 UTM 위치
        position_text = f"Current UTM: ({self.current_utm_position[0]:.1f}, {self.current_utm_position[1]:.1f})"
        cv2.putText(main_image, position_text, (10, h - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 목표 UTM 위치
        target_text = f"Target UTM: ({self.target_utm_position[0]:.1f}, {self.target_utm_position[1]:.1f})"
        cv2.putText(main_image, target_text, (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 거리
        distance = np.linalg.norm(self.target_utm_position - self.current_utm_position)
        distance_text = f"Distance: {distance:.2f}m (Switch at {self.mode_switch_distance}m)"
        color = (0, 255, 0) if not self.mode_switched else (0, 0, 255)
        if distance <= self.mode_switch_distance and not self.mode_switched:
            color = (0, 255, 255)
        cv2.putText(main_image, distance_text, (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 모드 전환 상태 표시
        if self.mode_switched:
            switch_text = "MODE SWITCHED - APPROACH ACTIVE"
            cv2.putText(main_image, switch_text, (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # 깊이 맵 시각화
        depth_image = None
        if params['visualization']['show_depth']:
            depth_image = self.visualizer.visualize_depth_map(depth_map, tracks, detections)
        
        # 이미지 표시
        self.visualizer.show_images(main_image, depth_image, params['visualization']['show_depth'])
    
    def monitor_performance(self):
        """성능 모니터링"""
        if self.frame_count % 30 == 0:  # 30프레임마다
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            self.get_logger().info(f"FPS: {fps:.2f}, Frame: {self.frame_count}")
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        self.visualizer.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VRXRobotController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
