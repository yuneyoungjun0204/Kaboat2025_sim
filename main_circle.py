#!/usr/bin/env python3
"""
단일 미션 스크립트: 빨간색 부표를 기준으로 '시계방향' 회전만 수행
- NanoOWL 탐지 + MiDaS 깊이 사용
- ROS2 이미지/IMU 구독, 좌/우 스러스터 발행
"""

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import cv2

from utils import (
    MiDaSHybridDepthEstimator,
    SensorDataManager,
)
from utils.detection_system import DetectionSystem, MissionType
from utils.ros_communication import ROSCommunicationManager


class PIDController:
    """간단한 PID 제어기"""
    def __init__(self, kp=0.8, ki=0.001, kd=0.4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error: float) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0

        proportional = self.kp * error
        self.integral += error * dt
        integral = self.ki * self.integral
        derivative = self.kd * (error - self.previous_error) / dt

        output = proportional + integral + derivative
        self.previous_error = error
        self.last_time = current_time
        return output


class RedBuoyClockwiseCircle(Node):
    """빨간 부표 기준 시계방향 회전 전용 노드"""

    def __init__(self):
        super().__init__('red_buoy_clockwise_circle')

        # 기본 구성
        self.bridge = CvBridge()
        self.depth_estimator = MiDaSHybridDepthEstimator()
        self.detection_system = DetectionSystem(self.depth_estimator)
        self.sensor_manager = SensorDataManager()
        self.ros_comm = ROSCommunicationManager(self)

        # 상태
        self.current_image = None
        self.agent_heading = 0.0  # 0~360
        self.previous_heading = None
        self.total_rotation = 0.0

        # [수정] 마지막으로 성공한 스러스터 명령을 저장할 변수 추가 및 초기화
        self.last_known_left_cmd = 0.0
        self.last_known_right_cmd = 0.0

        # 제어 파라미터
        self.image_width = 1280
        self.target_center_x = self.image_width / 2
        self.base_speed = 150.0
        self.min_speed = 50.0
        self.max_turn_thrust = 150.0
        self.stop_distance = 0.0  # 항상 회전 모드 진입

        self.pid = PIDController(kp=0.8, ki=0.001, kd=0.4)

        # target_x 결정식 파라미터 (초기값)
        # target_x = clamp(min_x, max_x, base_x - slope * depth)
        self.tx_base_x = 1240.0
        self.tx_slope = 700.0
        self.tx_min_x = 800.0
        self.tx_max_x = 1200.0

        # OpenCV 윈도우/트랙바 설정
        self._setup_windows()

        # 통신 설정
        self.ros_comm.setup_subscribers({
            'image': self.image_callback,
            'imu': self.imu_callback,
        })
        self.ros_comm.setup_publishers()

        # 주기 루프 (20Hz)
        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info('✓ RedBuoyClockwiseCircle node started')

    def _setup_windows(self):
        cv2.namedWindow('Circle Parameters', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Circle Parameters', 480, 320)
        # 탐지 파라미터
        cv2.createTrackbar('Detect Thr x1000', 'Circle Parameters', 30, 200, lambda v: None)
        cv2.createTrackbar('Min Box Area', 'Circle Parameters', 500, 20000, lambda v: None)
        cv2.createTrackbar('Max Box Area', 'Circle Parameters', 80000, 200000, lambda v: None)
        cv2.createTrackbar('Min Depth x20', 'Circle Parameters', 0, 200, lambda v: None)
        cv2.createTrackbar('Max Depth (m)', 'Circle Parameters', 50, 200, lambda v: None)
        # 제어 파라미터
        cv2.createTrackbar('Base Speed', 'Circle Parameters', int(self.base_speed), 300, lambda v: None)
        cv2.createTrackbar('Min Speed', 'Circle Parameters', int(self.min_speed), 200, lambda v: None)
        cv2.createTrackbar('Max Turn', 'Circle Parameters', int(self.max_turn_thrust), 300, lambda v: None)
        cv2.createTrackbar('PID Kp x10', 'Circle Parameters', int(self.pid.kp * 10), 50, lambda v: None)
        # target_x 결정식 파라미터
        cv2.createTrackbar('TX BaseX', 'Circle Parameters', int(self.tx_base_x), 2000, lambda v: None)
        cv2.createTrackbar('TX Slope', 'Circle Parameters', int(self.tx_slope), 10000, lambda v: None)
        cv2.createTrackbar('TX MinX', 'Circle Parameters', int(self.tx_min_x), 2000, lambda v: None)
        cv2.createTrackbar('TX MaxX', 'Circle Parameters', int(self.tx_max_x), 2000, lambda v: None)

    def _read_trackbar_params(self):
        # Detection params
        det_thr = cv2.getTrackbarPos('Detect Thr x1000', 'Circle Parameters') / 1000.0
        min_box = cv2.getTrackbarPos('Min Box Area', 'Circle Parameters')
        max_box = cv2.getTrackbarPos('Max Box Area', 'Circle Parameters')
        min_depth = cv2.getTrackbarPos('Min Depth x20', 'Circle Parameters') / 20.0
        max_depth = float(cv2.getTrackbarPos('Max Depth (m)', 'Circle Parameters'))

        # Control params
        base_speed = float(cv2.getTrackbarPos('Base Speed', 'Circle Parameters'))
        min_speed = float(cv2.getTrackbarPos('Min Speed', 'Circle Parameters'))
        max_turn = float(cv2.getTrackbarPos('Max Turn', 'Circle Parameters'))
        kp = cv2.getTrackbarPos('PID Kp x10', 'Circle Parameters') / 10.0

        # target_x 결정식 파라미터
        tx_base_x = float(cv2.getTrackbarPos('TX BaseX', 'Circle Parameters'))
        tx_slope = float(cv2.getTrackbarPos('TX Slope', 'Circle Parameters'))
        tx_min_x = float(cv2.getTrackbarPos('TX MinX', 'Circle Parameters'))
        tx_max_x = float(cv2.getTrackbarPos('TX MaxX', 'Circle Parameters'))

        return {
            'detection_threshold': det_thr,
            'min_box_area': min_box,
            'max_box_area': max_box,
            'min_depth': min_depth,
            'max_depth': max_depth,
            'base_speed': base_speed,
            'min_speed': min_speed,
            'max_turn': max_turn,
            'kp': kp,
            'tx_base_x': tx_base_x,
            'tx_slope': tx_slope,
            'tx_min_x': tx_min_x,
            'tx_max_x': tx_max_x,
        }

    # ===== 콜백 =====
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def imu_callback(self, msg):
        imu_data = self.sensor_manager.process_imu_data(msg)
        heading = imu_data['yaw_degrees']
        if heading < 0:
            heading += 360.0
        self.agent_heading = heading

    # ===== 유틸 =====
    def calculate_rotation_target_x(self, depth_m: float) -> float:
        """시계방향 기준 목표 x 계산: base_x - slope * depth (min_x~max_x 클램프)"""
        target_x = self.tx_base_x - self.tx_slope * depth_m
        return max(self.tx_min_x, min(self.tx_max_x, target_x))

    def calculate_rotation_speed(self, turn_angle: float) -> float:
        """각도 커질수록 속도 감소 (선형), 90도에서 min_speed"""
        abs_angle = abs(turn_angle)
        if abs_angle >= 90.0:
            return self.min_speed
        ratio = 1.0 - (abs_angle / 90.0)
        return max(self.min_speed, self.min_speed + (self.base_speed - self.min_speed) * ratio)

    # ===== 메인 루프 =====
    def loop(self):
        if self.current_image is None:
            return

        # 트랙바 파라미터 적용
        tp = self._read_trackbar_params()
        self.detection_system.update_parameters(
            detection_threshold=tp['detection_threshold'],
            min_box_area=tp['min_box_area'],
            max_box_area=tp['max_box_area'],
            min_depth=tp['min_depth'],
            max_depth=tp['max_depth'],
        )
        self.base_speed = tp['base_speed']
        self.min_speed = tp['min_speed']
        self.max_turn_thrust = tp['max_turn']
        if self.pid.kp != tp['kp']:
            self.pid.kp = tp['kp']
        # target_x 결정식 파라미터 반영
        self.tx_base_x = tp['tx_base_x']
        self.tx_slope = tp['tx_slope']
        self.tx_min_x = min(tp['tx_min_x'], tp['tx_max_x'] - 1.0)
        self.tx_max_x = max(tp['tx_max_x'], self.tx_min_x + 1.0)

        # 빨간 부표 탐지: PASS_BETWEEN_BUOYS 쿼리를 사용하여 red_cone 선택
        detections = self.detection_system.detect_objects(self.current_image, MissionType.PASS_BETWEEN_BUOYS)
        red_buoy = None
        for det in detections:
            if det['label'] == 'red_cone':
                red_buoy = det
                break

        # 누적 회전 계산 준비
        if self.previous_heading is None:
            self.previous_heading = self.agent_heading
            self.total_rotation = 0.0

        heading_diff = self.agent_heading - self.previous_heading
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360
        self.total_rotation += abs(heading_diff)
        self.previous_heading = self.agent_heading

        # 360도 회전 완료 시 정지
        if self.total_rotation >= 350.0:
            self.ros_comm.publish_thrust_commands(0.0, 0.0)
            return

        # 시각화용 프레임 준비
        vis = self.current_image.copy()
        
        # [수정] 스러스터 명령 변수 초기화
        left_cmd, right_cmd = 0.0, 0.0

        if red_buoy is not None:
            # 회전 모드: 목표 x 계산 및 PID 조향
            depth_m = red_buoy['depth']
            x_pixel = red_buoy['center'][0]

            target_x = self.calculate_rotation_target_x(depth_m)
            error_px = target_x - x_pixel

            # 오차를 이미지 반폭으로 정규화 (-1~1)
            normalized_error = error_px / (self.image_width / 2.0)
            steering_cmd = self.pid.update(normalized_error)
            steering_cmd = max(-1.0, min(1.0, steering_cmd))

            # 회전/전진 추력 계산
            turn_thrust = steering_cmd * self.max_turn_thrust
            turn_angle = abs(steering_cmd * 90.0)
            forward_thrust = self.calculate_rotation_speed(turn_angle)

            left_cmd = forward_thrust - turn_thrust
            right_cmd = forward_thrust + turn_thrust
            
            # [수정] 마지막으로 성공한 명령을 저장
            self.last_known_left_cmd = left_cmd
            self.last_known_right_cmd = right_cmd

            # ===== 시각화 오버레이 (탐지 성공 시) =====
            x1, y1, x2, y2 = red_buoy['bbox']
            cx, cy = red_buoy['center']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.line(vis, (int(target_x), 0), (int(target_x), vis.shape[0]), (255, 0, 0), 2)
            
            info_lines = [
                f"depth: {depth_m:.2f} m",
                f"buoy_x: {x_pixel:.1f} px",
                f"target_x: {target_x:.1f} px",
                f"error_px: {error_px:.1f}",
                f"steer: {steering_cmd:.3f}",
                f"fwd: {forward_thrust:.1f}, turn: {turn_thrust:.1f}",
                f"rot_total: {self.total_rotation:.1f} deg",
            ]
        else:
            # [수정] 탐지 실패 시, 이전 명령을 그대로 사용
            left_cmd = self.last_known_left_cmd
            right_cmd = self.last_known_right_cmd
            cv2.putText(vis, 'Detection Lost! Using last command...', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # [수정] 탐지 실패 시에는 기존 정보 텍스트 대신 간단한 정보만 표시
            info_lines = [
                "Searching for red_cone...",
                f"Last L_CMD: {left_cmd:.1f}",
                f"Last R_CMD: {right_cmd:.1f}",
                f"rot_total: {self.total_rotation:.1f} deg",
            ]

        # ===== 공통 시각화 부분 =====
        # 중앙선
        cv2.line(vis, (int(self.target_center_x), 0), (int(self.target_center_x), vis.shape[0]), (0, 255, 0), 1)
        # 텍스트 정보
        y0 = 30
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (30, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 창 표시 및 퍼블리시
        cv2.imshow('Circle Parameters', vis)
        cv2.waitKey(1)

        # viz_image 토픽 퍼블리시
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            self.ros_comm.publishers['viz_image'].publish(img_msg)
        except Exception:
            pass

        self.ros_comm.publish_thrust_commands(left_cmd, right_cmd)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RedBuoyClockwiseCircle()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()