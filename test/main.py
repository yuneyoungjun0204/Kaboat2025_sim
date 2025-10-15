#!/usr/bin/env python3
"""
VRX 로봇 제어 시스템 메인 파일 - 간소화 버전
모든 기능이 모듈화되어 최대한 간단한 구조
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
import time

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
    """VRX 로봇 제어 메인 노드 - 간소화 버전"""
    
    def __init__(self):
        super().__init__('vrx_robot_controller')
        
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
        
        # ROS2 퍼블리셔 (추적 데이터만 직접 관리)
        self.tracking_pub = self.create_publisher(Float32MultiArray, '/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions', 10)
        
        # 상태 변수
        self.frame_count = 0
        self.start_time = time.time()
        
        self.get_logger().info('🚀 VRX 로봇 제어 시스템 시작! (간소화 버전)')
    
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
        
        # 파라미터 업데이트
        self.update_parameters(params)
        
        # 핵심 처리 파이프라인
        depth_map = self.depth_estimator.estimate_depth(cv_image)
        detections = self.blob_detector.detect_blobs(
            cv_image, 
            depth_map, 
            params['detection']['min_depth_threshold'], 
            params['detection']['max_depth_threshold']
        )
        tracks = self.tracker.update(detections, depth_map)
        best_red, best_green = self.tracker.get_best_tracks()
        
        # 제어 처리
        self.process_control(best_red, best_green, params)
        
        # 추적 데이터 퍼블리시
        self.publish_tracking_data(best_red, best_green)
        
        # 시각화
        self.visualize_results(cv_image, tracks, detections, depth_map, params)
        
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
    
    def process_control(self, best_red, best_green, params):
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
            # 객체 접근 제어
            target_track = best_green if target_color == "green" else best_red
            
            if target_track and target_track.confidence > 0.3:
                left_cmd, right_cmd, error, steering, forward, turn, mode, target_x = self.navigation_controller.approach_control(
                    target_track.center[0], target_track.center[1], target_track.depth or 0.0,
                    rotation_direction=rotation_direction
                )
                
                # 목표 X값 퍼블리시
                self.thruster_controller.publish_target_x(target_x)
                
                direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
                status = f"🎯 접근: {target_color.upper()}({target_track.center[0]:.1f}), 깊이: {target_track.depth:.3f}m, 방향: {direction_name}"
            else:
                left_cmd = right_cmd = 0.0
                status = f"❌ {target_color} 객체 미탐지: 정지"
        
        # 스러스터 명령 퍼블리시
        self.thruster_controller.publish_thrust_commands(left_cmd, right_cmd)
        self.thruster_controller.publish_status(status)
        
        # 로그 출력 (1초마다)
        if self.frame_count % 10 == 0:
            self.get_logger().info(status)
    
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
    
    def visualize_results(self, image, tracks, detections, depth_map, params):
        """시각화 - 모듈화된 시각화 사용"""
        # 메인 추적 결과 시각화
        main_image = self.visualizer.visualize_tracking_results(
            image, tracks, detections, self.frame_count,
            params['control']['control_mode'], params['control']['target_color']
        )
        
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