#!/usr/bin/env python3
"""
VRX 자동 시나리오 전환 시스템
- 지정 좌표까지: Navigation (부표 사이 통과)
- 10m 이내 도달 시: Approach (부표 선회)
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import time
from utils import (
    MiDaSHybridDepthEstimator,
    BlobDetector,
    MultiTargetTracker,
    NavigationController,
    ThrusterController,
    Visualizer,
    TrackbarController
)


class AutoScenarioController(Node):
    """자동 시나리오 전환 제어 노드"""
    
    def __init__(self):
        super().__init__('auto_scenario_controller')
        
        # ==================== 시나리오 설정 영역 ====================
        # 목표 좌표 (UTM 좌표) - 여기서 수정하세요!
        self.target_utm_position = np.array([75.0, 45.0])  # [Y(North), X(East)]
        
        # 시나리오 전환 거리 (미터)
        self.scenario_switch_distance = 15.0
        
        # 선회 방향 (시나리오 2)
        self.rotation_direction = 1  # 1: 시계방향, -1: 반시계방향
        
        # 선회 대상 부표 색상 (검정색으로 고정)
        self.target_color = 'black'  # 검정색 부표만 사용
        # =======================================================
        
        self.get_logger().info('🚀 자동 시나리오 전환 시스템 시작!')
        self.get_logger().info(f'📍 목표 좌표: ({self.target_utm_position[0]:.1f}, {self.target_utm_position[1]:.1f})')
        self.get_logger().info(f'🔄 전환 거리: {self.scenario_switch_distance}m')
        
        # 초기화
        self.bridge = CvBridge()
        self.current_image = None
        
        # 부표 탐지 시스템
        self.depth_estimator = MiDaSHybridDepthEstimator()
        self.blob_detector = BlobDetector()
        self.tracker = MultiTargetTracker()
        self.navigation_controller = NavigationController()
        self.thruster_controller = ThrusterController(self)
        self.visualizer = Visualizer()
        self.trackbar_controller = TrackbarController()
        
        # 탐지 결과
        self.current_detections = []
        self.current_tracks = []
        self.current_depth_map = None
        self.best_red_track = None
        self.best_green_track = None
        self.black_buoy_tracks = []  # 검정색 부표 추적 결과
        
        # GPS 정보 (간단한 시뮬레이션)
        self.current_utm_position = np.array([0.0, 0.0])  # [Y, X]
        
        # 시나리오 상태
        self.current_scenario = 1  # 1: Navigation, 2: Approach
        self.distance_to_target = 0.0
        
        # ROS2 서브스크라이버
        self.image_sub = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
            self.image_callback,
            10
        )
        
        # GPS 시뮬레이션용 (실제로는 NavSatFix 구독)
        # 여기서는 간단히 목표로 조금씩 이동하는 것으로 시뮬레이션
        self.create_timer(0.1, self.update_position_simulation)
        
        # ROS2 퍼블리셔
        self.tracking_pub = self.create_publisher(
            Float32MultiArray, 
            '/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions', 
            10
        )
        
        # 상태 변수
        self.frame_count = 0
        self.start_time = time.time()
        
        # OpenCV 창 및 트랙바 설정
        self.setup_windows_and_trackbars()
        
        self.get_logger().info('✅ 초기화 완료!')
    
    def setup_windows_and_trackbars(self):
        """OpenCV 창 및 트랙바 설정"""
        # 메인 창 생성
        cv2.namedWindow('Auto Scenario - Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Auto Scenario - Camera View', 960, 540)
        
        # 깊이 맵 창 생성
        cv2.namedWindow('Auto Scenario - Depth Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Auto Scenario - Depth Map', 640, 360)
        
        # 시나리오 컨트롤 창 생성
        cv2.namedWindow('Scenario Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Scenario Control', 400, 200)
        
        # 시나리오 관련 트랙바 (먼저 생성)
        cv2.createTrackbar('Switch Distance', 'Scenario Control', 
                          int(self.scenario_switch_distance), 50, self.on_switch_distance_change)
        cv2.createTrackbar('Rotation Dir', 'Scenario Control', 
                          1 if self.rotation_direction > 0 else 0, 1, self.on_rotation_dir_change)
        # 검정색 부표 전용이므로 Target Color 트랙바 제거
        
        # 기본 트랙바 (Control Parameters 창 생성)
        self.trackbar_controller.setup_trackbars()
        
        self.get_logger().info('🎛️  창 및 트랙바 설정 완료!')
    
    def on_switch_distance_change(self, value):
        """전환 거리 변경"""
        self.scenario_switch_distance = float(value)
        self.get_logger().info(f'🔄 전환 거리 변경: {value}m')
    
    def on_rotation_dir_change(self, value):
        """회전 방향 변경"""
        self.rotation_direction = 1 if value > 0 else -1
        direction_name = "시계방향" if self.rotation_direction > 0 else "반시계방향"
        self.get_logger().info(f'🔄 회전 방향 변경: {direction_name}')
    
    def detect_black_buoys(self, image, depth_map, min_depth, max_depth):
        """검정색 부표 탐지"""
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 검정색 범위 (HSV)
        # H: 모든 값, S: 낮음~중간(채도), V: 낮음(밝기)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])  # V값 60 이하면 검정
        
        # 마스크 생성
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 깊이 필터링
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_mask = ((depth_normalized > min_depth) & (depth_normalized < max_depth)).astype(np.uint8) * 255
        
        # 최종 마스크
        final_mask = cv2.bitwise_and(mask, depth_mask)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 면적 필터
            if 200 < area < 10000:
                # 원형도 체크
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity >= 0.3:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Detection 객체 생성
                            detection = type('Detection', (), {
                                'center': (cx, cy),
                                'bbox': (cx-30, cy-30, cx+30, cy+30),  # bbox 추가
                                'area': area,
                                'color': 'black',
                                'depth': depth_normalized[cy, cx] if 0 <= cy < depth_normalized.shape[0] and 0 <= cx < depth_normalized.shape[1] else 0.0,
                                'circularity': circularity
                            })()
                            detections.append(detection)
        
        return detections
    
    def update_position_simulation(self):
        """위치 시뮬레이션 (실제로는 GPS 콜백으로 대체)"""
        # 목표 방향으로 조금씩 이동 (초당 1m)
        direction = self.target_utm_position - self.current_utm_position
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            # 목표 방향 단위 벡터
            unit_direction = direction / distance
            # 0.1초에 0.1m 이동 (초당 1m)
            self.current_utm_position += unit_direction * 0.1
        
        # 목표까지 거리 계산
        self.distance_to_target = np.linalg.norm(
            self.target_utm_position - self.current_utm_position
        )
        
        # 시나리오 전환 확인
        self.check_scenario_switch()
    
    def check_scenario_switch(self):
        """시나리오 전환 체크"""
        old_scenario = self.current_scenario
        
        if self.distance_to_target < self.scenario_switch_distance:
            # 10m 이내: Approach 모드
            self.current_scenario = 2
        else:
            # 10m 이상: Navigation 모드
            self.current_scenario = 1
        
        # 시나리오가 변경되었으면 로그 출력
        if old_scenario != self.current_scenario:
            if self.current_scenario == 1:
                self.get_logger().info('🎯 시나리오 1: Navigation (부표 사이 통과)')
            else:
                self.get_logger().info(f'🔄 시나리오 2: Approach (검정 부표 선회)')
    
    def image_callback(self, msg):
        """이미지 콜백 - 메인 파이프라인"""
        self.frame_count += 1
        
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        # 트랙바 파라미터 읽기
        params = self.trackbar_controller.get_all_parameters()
        
        # 파라미터 업데이트
        self.update_parameters(params)
        
        # 핵심 처리 파이프라인 - 검정색 부표 탐지
        self.current_depth_map = self.depth_estimator.estimate_depth(self.current_image)
        
        # 검정색 부표 탐지
        black_detections = self.detect_black_buoys(
            self.current_image, 
            self.current_depth_map, 
            params['detection']['min_depth_threshold'], 
            params['detection']['max_depth_threshold']
        )
        
        # 기존 빨강/초록 부표 탐지도 유지 (Navigation용)
        self.current_detections = self.blob_detector.detect_blobs(
            self.current_image, 
            self.current_depth_map, 
            params['detection']['min_depth_threshold'], 
            params['detection']['max_depth_threshold']
        )
        
        # 검정색 부표와 기존 부표를 합쳐서 추적
        all_detections = self.current_detections + black_detections
        self.current_tracks = self.tracker.update(all_detections, self.current_depth_map)
        self.best_red_track, self.best_green_track = self.tracker.get_best_tracks()
        
        # 검정색 부표 추적 결과
        self.black_buoy_tracks = [t for t in self.current_tracks 
                                 if hasattr(t, 'color') and t.color == 'black' and t.confidence > 0.3]
        
        # 시나리오별 제어 처리
        self.process_scenario_control(params)
        
        # 추적 데이터 퍼블리시
        self.publish_tracking_data()
        
        # 시각화
        self.visualize_results(params)
        
        # 성능 모니터링
        self.monitor_performance()
    
    def update_parameters(self, params):
        """파라미터 업데이트"""
        self.blob_detector.update_detection_parameters(**params['blob_detector'])
        self.tracker.update_tracking_parameters(**params['tracking'])
        
        nav_params = params['navigation'].copy()
        nav_params.update({
            'steering_kp': params['pid']['steering_kp'],
            'approach_kp': params['pid']['approach_kp']
        })
        self.navigation_controller.update_control_parameters(**nav_params)
    
    def process_scenario_control(self, params):
        """시나리오별 제어 처리"""
        if self.current_scenario == 1:
            # 시나리오 1: Navigation (부표 사이 통과)
            self.process_navigation_scenario()
        else:
            # 시나리오 2: Approach (부표 선회)
            self.process_approach_scenario()
    
    def process_navigation_scenario(self):
        """시나리오 1: Navigation 제어"""
        if self.best_red_track and self.best_green_track and \
           self.best_red_track.confidence > 0.3 and self.best_green_track.confidence > 0.3:
            # 두 부표 모두 보임 - 부표 사이 통과
            left_cmd, right_cmd, error, steering, forward, turn = \
                self.navigation_controller.navigation_control(
                    self.best_red_track.center[0], self.best_green_track.center[0],
                    self.best_red_track.center[1], self.best_green_track.center[1]
                )
            status = f"🎯 Navigation: 중점 추종, 거리: {self.distance_to_target:.1f}m"
        else:
            # 부표 미탐지 - 천천히 직진
            left_cmd = right_cmd = 150.0
            status = f"⚠️ 부표 미탐지: 직진, 거리: {self.distance_to_target:.1f}m"
        
        # 스러스터 명령 발행
        self.thruster_controller.publish_thrust_commands(left_cmd, right_cmd)
        self.thruster_controller.publish_status(status)
        
        # 로그 출력 (1초마다)
        if self.frame_count % 10 == 0:
            self.get_logger().info(status)
    
    def process_approach_scenario(self):
        """시나리오 2: Approach 제어 - 검정 부표 선회"""
        if len(self.black_buoy_tracks) >= 1:
            # 가장 가까운 검정 부표 선택 (Y 좌표가 가장 큰 것 = 화면 아래)
            target_track = max(self.black_buoy_tracks, key=lambda t: t.center[1])
            
            # 접근 및 선회
            left_cmd, right_cmd, error, steering, forward, turn, mode, target_x = \
                self.navigation_controller.approach_control(
                    target_track.center[0], target_track.center[1], 
                    target_track.depth or 0.0,
                    rotation_direction=self.rotation_direction
                )
            
            # 목표 X값 퍼블리시
            self.thruster_controller.publish_target_x(target_x)
            
            direction_name = "시계방향" if self.rotation_direction == 1 else "반시계방향"
            status = f"🔄 Approach: 검정 부표 ID:{target_track.track_id}({target_track.center[0]:.1f}), " \
                    f"깊이: {target_track.depth:.3f}m, {direction_name}, 거리: {self.distance_to_target:.1f}m"
        else:
            # 검정 부표 미탐지 - 정지
            left_cmd = right_cmd = 0.0
            status = f"❌ 검정 부표 미탐지 (0개): 정지, 거리: {self.distance_to_target:.1f}m"
        
        # 스러스터 명령 발행
        self.thruster_controller.publish_thrust_commands(left_cmd, right_cmd)
        self.thruster_controller.publish_status(status)
        
        # 로그 출력 (1초마다)
        if self.frame_count % 10 == 0:
            self.get_logger().info(status)
    
    def publish_tracking_data(self):
        """추적 데이터 퍼블리시"""
        tracking_msg = Float32MultiArray()
        
        red_x = float(self.best_red_track.center[0]) if self.best_red_track and self.best_red_track.confidence > 0.3 else 0.0
        red_y = float(self.best_red_track.center[1]) if self.best_red_track and self.best_red_track.confidence > 0.3 else 0.0
        red_depth = float(self.best_red_track.depth) if self.best_red_track and self.best_red_track.depth is not None else 0.0
        
        green_x = float(self.best_green_track.center[0]) if self.best_green_track and self.best_green_track.confidence > 0.3 else 0.0
        green_y = float(self.best_green_track.center[1]) if self.best_green_track and self.best_green_track.confidence > 0.3 else 0.0
        green_depth = float(self.best_green_track.depth) if self.best_green_track and self.best_green_track.depth is not None else 0.0
        
        tracking_msg.data = [red_x, red_y, red_depth, green_x, green_y, green_depth, time.time()]
        self.tracking_pub.publish(tracking_msg)
    
    def visualize_results(self, params):
        """시각화"""
        if self.current_image is None:
            return
        
        # 시나리오에 따른 control_mode 설정
        if self.current_scenario == 1:
            control_mode = "navigation"
        else:
            control_mode = "approach"
        
        # 메인 추적 결과 시각화
        main_image = self.visualizer.visualize_tracking_results(
            self.current_image.copy(), 
            self.current_tracks, 
            self.current_detections, 
            self.frame_count,
            control_mode, 
            self.target_color
        )
        
        # 시나리오 정보 오버레이
        h, w = main_image.shape[:2]
        
        # 상단: 시나리오 정보
        scenario_text = f"Scenario {self.current_scenario}: "
        if self.current_scenario == 1:
            scenario_text += "NAVIGATION (Gate Passing)"
            color = (0, 255, 0)  # 초록
        else:
            scenario_text += f"APPROACH (검정 부표 선회)"
            color = (0, 255, 255)  # 노랑
        
        cv2.putText(main_image, scenario_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 중단: 거리 정보
        distance_text = f"Distance to Target: {self.distance_to_target:.1f}m"
        distance_color = (0, 0, 255) if self.distance_to_target < self.scenario_switch_distance else (0, 255, 0)
        cv2.putText(main_image, distance_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, distance_color, 2)
        
        # 하단: 현재 위치 및 목표 위치
        position_text = f"Current: ({self.current_utm_position[0]:.1f}, {self.current_utm_position[1]:.1f})"
        cv2.putText(main_image, position_text, (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        target_text = f"Target: ({self.target_utm_position[0]:.1f}, {self.target_utm_position[1]:.1f})"
        cv2.putText(main_image, target_text, (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 전환 거리 표시
        switch_text = f"Switch Distance: {self.scenario_switch_distance:.1f}m"
        cv2.putText(main_image, switch_text, (10, h - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 검정 부표 개수 표시 (우측 상단)
        buoy_count_text = f"Black Buoys: {len(self.black_buoy_tracks)}"
        cv2.putText(main_image, buoy_count_text, (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 검정 부표 정보 표시
        for idx, track in enumerate(self.black_buoy_tracks[:3]):  # 최대 3개
            if hasattr(track, 'depth'):
                buoy_info = f"#{idx+1}: {track.depth:.3f}m"
                cv2.putText(main_image, buoy_info, (w - 250, 70 + idx * 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Auto Scenario - Camera View', main_image)
        
        # 깊이 맵 표시
        if params['visualization']['show_depth'] and self.current_depth_map is not None:
            depth_image = self.visualizer.visualize_depth_map(
                self.current_depth_map, 
                self.current_tracks, 
                self.current_detections
            )
            cv2.imshow('Auto Scenario - Depth Map', depth_image)
        
        # 키 입력 처리
        cv2.waitKey(1)
    
    def monitor_performance(self):
        """성능 모니터링"""
        if self.frame_count % 30 == 0:  # 30프레임마다
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            self.get_logger().info(f"📊 FPS: {fps:.2f}, Frame: {self.frame_count}")
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        self.visualizer.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = AutoScenarioController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

