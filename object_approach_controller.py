#!/usr/bin/env python3
"""
Object Approach Controller
- 탐지된 객체에 다가가다가 일정 거리에서 정지하는 시스템
- PID 제어로 객체를 이미지 중앙(640px)에 위치시키기
- 거리에 따른 적응형 속도 제어
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String
import time
import math
import cv2
import numpy as np

class PIDController:
    """PID 제어기"""
    def __init__(self, kp=2.2, ki=0.001, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error):
        """PID 제어 업데이트"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return 0.0
        
        # 비례 항
        proportional = self.kp * error
        
        # 적분 항
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # 미분 항
        derivative = self.kd * (error - self.previous_error) / dt
        
        # PID 출력
        output = proportional + integral + derivative
        
        # 상태 업데이트
        self.previous_error = error
        self.last_time = current_time
        
        return output

class ObjectApproachController(Node):
    """객체 접근 제어기"""
    
    def __init__(self):
        super().__init__('object_approach_controller')
        
        # PID 제어기 초기화
        self.pid_controller = PIDController(kp=0.8, ki=0.001, kd=0.4)
        
        # ROS2 퍼블리셔 (직접 스러스터 제어)
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/approach/status', 10)
        self.target_x_pub = self.create_publisher(Float64, '/approach/target_x', 10)  # 목표 X값 퍼블리시
        
        # ROS2 서브스크라이버
        self.tracking_sub = self.create_subscription(
            Float32MultiArray,
            '/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions',
            self.tracking_callback,
            10
        )
        
        # 상태 변수
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_depth = 0.0
        self.last_tracking_time = 0.0
        self.tracking_active = False
        self.object_detected = False
        
        # 제어 파라미터
        self.image_width = 1280
        self.image_height = 720
        self.target_center_x = self.image_width / 2  # 640px (이미지 중앙)
        
        # 속도 제어 파라미터
        self.base_speed = 150.0  # 기본 속도 (왼쪽, 오른쪽 모터)
        self.min_speed = 50.0  # 최소 속도 (회전 시 각도가 클 때)
        self.max_turn_thrust = 150.0  # 최대 회전 추력
        
        # 거리 제어 파라미터 (깊이 맵 값 기준 - 가까울수록 큰 값)
        self.approach_distance = 0.05  # 접근 시작 거리 (미터) - 이 값 이상이면 접근 시작
        self.stop_distance = 0.02  # 정지 거리 (미터) - 이 값 이상이면 정지
        self.slow_distance = 0.03  # 감속 시작 거리 (미터) - 이 값 이상이면 감속
        
        # 상태 변수 
        self.frame_count = 0
        self.start_time = time.time()
        
        # 이전 명령값 저장 (인식 실패 시 사용)
        self.last_left_command = 0.0
        self.last_right_command = 0.0
        
        # 트랙바 설정
        self.setup_trackbars()
        
        # 제어 루프 타이머
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz
        
        self.get_logger().info('🔄 고깔 회전 제어기 시작!')
        self.get_logger().info(f'이미지 크기: {self.image_width}x{self.image_height}')
        self.get_logger().info(f'목표 중앙 위치: {self.target_center_x}px')
        self.get_logger().info(f'기본 속도: {self.base_speed}')
        self.get_logger().info(f'최소 속도: {self.min_speed}')
        self.get_logger().info(f'접근 거리: {self.approach_distance:.3f}m')
        self.get_logger().info(f'감속 거리: {self.slow_distance:.3f}m')
        self.get_logger().info(f'회전 시작 거리: {self.stop_distance:.3f}m')
    
    def setup_trackbars(self):
        """트랙바 설정"""
        # 트랙바 창 생성
        cv2.namedWindow("Object Approach Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Approach Control", 400, 300)
        
        # 트랙바 생성
        cv2.createTrackbar("Target_Color", "Object Approach Control", 2, 2, self.nothing)
        cv2.createTrackbar("Rotation_Direction", "Object Approach Control", 1, 2, self.nothing)  # 1: 시계방향, 2: 반시계방향
        cv2.createTrackbar("Base_Speed", "Object Approach Control", 150, 300, self.nothing)
        cv2.createTrackbar("Min_Speed", "Object Approach Control", 50, 200, self.nothing)
        cv2.createTrackbar("Max_Turn_Thrust", "Object Approach Control", 150, 250, self.nothing)
        cv2.createTrackbar("Approach_Distance", "Object Approach Control", 5, 15, self.nothing)  # 0.01-0.15m
        cv2.createTrackbar("Slow_Distance", "Object Approach Control", 3, 15, self.nothing)  # 0.01-0.15m
        cv2.createTrackbar("Stop_Distance", "Object Approach Control", 2, 15, self.nothing)  # 0.01-0.15m
        cv2.createTrackbar("PID_Kp", "Object Approach Control", 8, 50, self.nothing)  # 0.8-5.0
        
        # 초기값 설정
        cv2.setTrackbarPos("Target_Color", "Object Approach Control", 1)  # 1: 초록색, 2: 빨간색
        cv2.setTrackbarPos("Rotation_Direction", "Object Approach Control", 1)  # 1: 시계방향, 2: 반시계방향
        cv2.setTrackbarPos("Base_Speed", "Object Approach Control", 150)  # 기본 속도 150
        cv2.setTrackbarPos("Min_Speed", "Object Approach Control", 50)  # 최소 속도 50
        cv2.setTrackbarPos("Max_Turn_Thrust", "Object Approach Control", 150)
        cv2.setTrackbarPos("Approach_Distance", "Object Approach Control", 3)  # 0.05m
        cv2.setTrackbarPos("Slow_Distance", "Object Approach Control", 4)  # 0.03m
        cv2.setTrackbarPos("Stop_Distance", "Object Approach Control", 7)  # 0.02m
        cv2.setTrackbarPos("PID_Kp", "Object Approach Control", 8)  # Kp = 0.8
        
        self.get_logger().info('✅ 트랙바 설정 완료')
    
    def nothing(self, val):
        """트랙바 콜백 함수 (빈 함수)"""
        pass
    
    def update_parameters_from_trackbars(self):
        """트랙바에서 파라미터 업데이트"""
        # 트랙바 값 읽기
        target_color = cv2.getTrackbarPos("Target_Color", "Object Approach Control")
        rotation_direction = cv2.getTrackbarPos("Rotation_Direction", "Object Approach Control")
        base_speed = cv2.getTrackbarPos("Base_Speed", "Object Approach Control")
        min_speed = cv2.getTrackbarPos("Min_Speed", "Object Approach Control")
        max_turn_thrust = cv2.getTrackbarPos("Max_Turn_Thrust", "Object Approach Control")
        approach_distance = cv2.getTrackbarPos("Approach_Distance", "Object Approach Control") / 100.0  # 0.01-0.15m
        slow_distance = cv2.getTrackbarPos("Slow_Distance", "Object Approach Control") / 100.0  # 0.01-0.15m
        stop_distance = cv2.getTrackbarPos("Stop_Distance", "Object Approach Control") / 100.0  # 0.01-0.15m
        pid_kp = cv2.getTrackbarPos("PID_Kp", "Object Approach Control") / 10.0  # 0.1-5.0
        
        # 파라미터 업데이트
        self.base_speed = float(base_speed)
        self.min_speed = float(min_speed)
        self.max_turn_thrust = float(max_turn_thrust)
        self.approach_distance = approach_distance
        self.slow_distance = slow_distance
        self.stop_distance = stop_distance
        self.pid_controller.kp = pid_kp
        
        return target_color, rotation_direction
    
    def tracking_callback(self, msg):
        """부표 추적 결과 콜백"""
        if len(msg.data) >= 7:
            # 트랙바에서 파라미터 업데이트
            target_color, rotation_direction = self.update_parameters_from_trackbars()
            
            # 데이터 파싱: [red_x, red_y, red_depth, green_x, green_y, green_depth, timestamp]
            red_x = msg.data[0]
            red_y = msg.data[1]
            red_depth = msg.data[2]
            green_x = msg.data[3]
            green_y = msg.data[4]
            green_depth = msg.data[5]
            self.last_tracking_time = time.time()
            
            # 트랙바에서 선택한 색상만 탐지
            if target_color == 1:  # 초록색 선택
                if green_x > 0 and green_depth > 0:
                    self.target_x = green_x
                    self.target_y = green_y
                    self.target_depth = green_depth
                    self.tracking_active = True
                    self.object_detected = True
                    
                    # 디버그 정보 출력
                    if self.frame_count % 30 == 0:  # 1초마다
                        direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
                        self.get_logger().info(
                            f'🎯 타겟 객체: GREEN ({self.target_x:.1f}, {self.target_y:.1f}), '
                            f'깊이: {self.target_depth:.3f}m, 방향: {direction_name}'
                        )
                else:
                    self.tracking_active = False
                    self.object_detected = False
                    if self.frame_count % 30 == 0:
                        self.get_logger().info('❌ 초록색 객체 미탐지')
                        
            elif target_color == 2:  # 빨간색 선택
                if red_x > 0 and red_depth > 0:
                    self.target_x = red_x
                    self.target_y = red_y
                    self.target_depth = red_depth
                    self.tracking_active = True
                    self.object_detected = True
                    
                    # 디버그 정보 출력
                    if self.frame_count % 30 == 0:  # 1초마다
                        direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
                        self.get_logger().info(
                            f'🎯 타겟 객체: RED ({self.target_x:.1f}, {self.target_y:.1f}), '
                            f'깊이: {self.target_depth:.3f}m, 방향: {direction_name}'
                        )
                else:
                    self.tracking_active = False
                    self.object_detected = False
                    if self.frame_count % 30 == 0:
                        self.get_logger().info('❌ 빨간색 객체 미탐지')
    
    def calculate_steering_command(self, error):
        """조향 명령 계산"""
        # 오차 정규화 (이미지 너비의 절반으로 나누어 -1~1 범위로)
        normalized_error = error / (self.image_width / 2)
        
        # PID 제어기로 조향 명령 계산
        steering_command = self.pid_controller.update(normalized_error)
        
        # 조향 명령 제한
        steering_command = max(-1.0, min(1.0, steering_command))
        
        return steering_command
    
    def calculate_rotation_speed(self, turn_angle):
        """각도에 따른 적응형 속도 계산 (각도가 클수록 속도 감소)"""
        # 각도를 절댓값으로 변환 (0~180도)
        abs_angle = abs(turn_angle)
        
        # 각도가 클수록 속도 감소 (선형적)
        # 0도: 기본 속도, 90도: 최소 속도
        if abs_angle >= 90:
            return self.min_speed
        else:
            speed_ratio = 1.0 - (abs_angle / 90.0)
            adaptive_speed = self.min_speed + (self.base_speed - self.min_speed) * speed_ratio
            return max(self.min_speed, adaptive_speed)
    
    def calculate_rotation_target(self, rotation_direction, object_x):
        """회전 방향에 따른 목표 x 좌표 계산 (거리에 따라 동적 조정)"""
        if rotation_direction == 1:  # 시계방향
            # 시계방향: 1200 - 100x (멀수록 크게, 가까울수록 작게)
            target_x = 1240 - 3000 * object_x
            # 범위 제한 (640~1200)
            return max(640, min(1200, target_x))
        else:  # 반시계방향
            # 반시계방향: 40 + 100x (멀수록 크게, 가까울수록 작게)
            target_x = 3000 * object_x
            # 범위 제한 (40~640)
            return max(40, min(640, target_x))
    
    def publish_target_x(self, target_x):
        """목표 X값 퍼블리시"""
        target_msg = Float64()
        target_msg.data = float(target_x)
        self.target_x_pub.publish(target_msg)
    
    def control_loop(self):
        """메인 제어 루프"""
        self.frame_count += 1
        
        # OpenCV 창 업데이트
        cv2.waitKey(1)
        
        # 추적 데이터가 너무 오래된 경우 비활성화
        if time.time() - self.last_tracking_time > 2.0:  # 2초 이상 데이터 없음
            self.tracking_active = False
            self.object_detected = False
        
        if self.tracking_active and self.object_detected:
            # 객체 탐지된 경우
            self.control_with_object()
        else:
            # 객체가 없으면 정지
            self.control_stop()
        
        # 상태 퍼블리시 (5초마다)
        if self.frame_count % 50 == 0:
            self.publish_status()
    
    def control_with_object(self):
        """객체가 탐지된 경우의 제어 (고깔 회전 미션)"""
        # 트랙바에서 회전 방향 가져오기
        rotation_direction = cv2.getTrackbarPos("Rotation_Direction", "Object Approach Control")
        
        # stop_distance 기준 충족 시 회전 시작
        if self.target_depth >= self.stop_distance:
            # 회전 모드: 고깔을 기준으로 일정한 방향으로 회전
            target_x = self.calculate_rotation_target(rotation_direction, self.target_depth)
            error = target_x - self.target_x
            
            # 목표 X값 퍼블리시 (blob_depth_detector에서 표시용)
            self.publish_target_x(target_x)
            
            # 조향 명령 계산
            steering_command = self.calculate_steering_command(error)
            
            # 회전 추력 계산
            turn_thrust = steering_command * self.max_turn_thrust
            
            # 각도에 따른 적응형 속도 계산
            turn_angle = abs(steering_command * 90)  # 조향 명령을 각도로 변환
            forward_thrust = self.calculate_rotation_speed(turn_angle)
            
            # 스러스터 명령 계산
            left_command = forward_thrust - turn_thrust
            right_command = forward_thrust + turn_thrust
            
            # 명령 퍼블리시
            self.publish_thrust_commands(left_command, right_command)
            
            # 이전 명령값 저장
            self.last_left_command = left_command
            self.last_right_command = right_command
            
            # 로그 출력 (1초마다)
            if self.frame_count % 10 == 0:
                direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
                self.get_logger().info(
                    f'🔄 고깔 회전: 위치({self.target_x:.1f}, {self.target_y:.1f}), '
                    f'깊이: {self.target_depth:.3f}m, 목표: {target_x}px, '
                    f'오차: {error:.1f}px, 조향: {steering_command:.3f}, '
                    f'전진: {forward_thrust:.1f}, 회전: {turn_thrust:.1f}, '
                    f'방향: {direction_name}'
                )
        else:
            # 접근 모드: 객체에 접근
            error = self.target_center_x - self.target_x
            steering_command = self.calculate_steering_command(error)
            turn_thrust = steering_command * self.max_turn_thrust
            forward_thrust = self.base_speed * 0.5  # 접근 시 천천히
            
            # 스러스터 명령 계산
            left_command = forward_thrust - turn_thrust
            right_command = forward_thrust + turn_thrust
            
            # 명령 퍼블리시
            self.publish_thrust_commands(left_command, right_command)
            
            # 이전 명령값 저장
            self.last_left_command = left_command
            self.last_right_command = right_command
            
            # 접근 모드 목표 X값 퍼블리시 (중앙)
            self.publish_target_x(self.target_center_x)
            
            # 로그 출력 (1초마다)
            if self.frame_count % 10 == 0:
                self.get_logger().info(
                    f'🎯 객체 접근: 위치({self.target_x:.1f}, {self.target_y:.1f}), '
                    f'깊이: {self.target_depth:.3f}m, 오차: {error:.1f}px, '
                    f'조향: {steering_command:.3f}, 전진: {forward_thrust:.1f}, '
                    f'회전: {turn_thrust:.1f}'
                )
    
    def control_stop(self):
        """인식 실패 시 이전 명령값 유지"""
        # 이전 명령값 사용 (인식 실패 시에도 계속 움직임)
        left_command = self.last_left_command
        right_command = self.last_right_command
        
        # 명령 퍼블리시
        self.publish_thrust_commands(left_command, right_command)
        
        # 인식 실패 시 목표 X값 퍼블리시 (중앙)
        self.publish_target_x(self.target_center_x)
        
        # 로그 출력 (5초마다)
        if self.frame_count % 50 == 0:
            self.get_logger().info(f'⚠️ 객체 미탐지: 이전 명령값 유지 (L:{left_command:.1f}, R:{right_command:.1f})')
    
    def publish_thrust_commands(self, left_thrust, right_thrust):
        """스러스터 명령 퍼블리시"""
        # 왼쪽 스러스터
        left_msg = Float64()
        left_msg.data = float(left_thrust)
        self.left_thrust_pub.publish(left_msg)
        
        # 오른쪽 스러스터
        right_msg = Float64()
        right_msg.data = float(right_thrust)
        self.right_thrust_pub.publish(right_msg)
    
    def publish_status(self):
        """상태 정보 퍼블리시"""
        status_msg = String()
        
        # 현재 선택된 색상과 회전 방향 확인
        target_color = cv2.getTrackbarPos("Target_Color", "Object Approach Control")
        rotation_direction = cv2.getTrackbarPos("Rotation_Direction", "Object Approach Control")
        color_name = "GREEN" if target_color == 1 else "RED"
        direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
        
        if self.tracking_active and self.object_detected:
            if self.target_depth >= self.stop_distance:
                status_msg.data = f"MODE: {color_name} object detected - ROTATING {direction_name} (close enough to start rotation)"
            elif self.target_depth >= self.slow_distance:
                status_msg.data = f"MODE: {color_name} object detected - APPROACHING (getting closer)"
            elif self.target_depth >= self.approach_distance:
                status_msg.data = f"MODE: {color_name} object detected - MOVING TOWARD (approach distance)"
            else:
                status_msg.data = f"MODE: {color_name} object detected - MOVING TOWARD (far - outside approach distance)"
        else:
            status_msg.data = f"MODE: No {color_name} object detected - STOPPED"
        
        self.status_pub.publish(status_msg)
        
        # 성능 정보
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        self.get_logger().info(f'FPS: {fps:.2f}, Target: {color_name}, Tracking: {"OK" if self.tracking_active else "NO"}, Object: {"OK" if self.object_detected else "NO"}')
    
    def update_control_parameters(self, kp=None, base_speed=None, max_turn_thrust=None, 
                                approach_distance=None, slow_distance=None, stop_distance=None):
        """제어 파라미터 업데이트"""
        if kp is not None:
            self.pid_controller.kp = kp
        if base_speed is not None:
            self.base_speed = base_speed
        if max_turn_thrust is not None:
            self.max_turn_thrust = max_turn_thrust
        if approach_distance is not None:
            self.approach_distance = approach_distance
        if slow_distance is not None:
            self.slow_distance = slow_distance
        if stop_distance is not None:
            self.stop_distance = stop_distance

def main(args=None):
    rclpy.init(args=args)
    
    controller = ObjectApproachController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
