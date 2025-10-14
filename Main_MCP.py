#!/usr/bin/env python3
"""
VRX 통합 미션 제어 시스템 (Main Mission Control Platform)
- NanoOWL 기반 객체 탐지 + MiDaS 깊이 필터링
- 웨이포인트 기반 미션 전환
- 4가지 미션: 부표 사이 지나가기 → 부표 회전 → 웨이포인트 추종 → 장애물 회피
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Float64MultiArray, String
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import time
import sys
from enum import Enum
from PIL import Image as PILImage
import onnxruntime as ort

# NanoOWL 경로 추가
sys.path.insert(0, '/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/nanoowl')
from nanoowl.owl_predictor import OwlPredictor

# 유틸리티 모듈
from utils import (
    MiDaSHybridDepthEstimator,
    SensorDataManager,
    AvoidanceController
)


class MissionType(Enum):
    """미션 타입 정의"""
    PASS_BETWEEN_BUOYS = 1  # 부표 사이 지나가기
    CIRCLE_BUOY = 2         # 부표 주변 회전
    WAYPOINT_FOLLOW = 3     # 웨이포인트 추종
    OBSTACLE_AVOID = 4      # 장애물 회피


class VRXMissionController(Node):
    """VRX 통합 미션 제어 노드"""

    def __init__(self):
        super().__init__('vrx_mission_controller')

        self.get_logger().info("=" * 80)
        self.get_logger().info("VRX 통합 미션 제어 시스템 초기화")
        self.get_logger().info("=" * 80)

        # 브릿지 및 디바이스
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ========== 컴포넌트 초기화 ==========
        self.init_detection_system()
        self.init_depth_estimation()
        self.init_sensor_manager()
        self.init_obstacle_avoidance()

        # ========== ROS2 통신 설정 ==========
        self.setup_subscribers()
        self.setup_publishers()

        # ========== 미션 상태 ==========
        self.current_mission = MissionType.PASS_BETWEEN_BUOYS
        self.mission_params = {}  # 미션별 파라미터

        # ========== 웨이포인트 관리 ==========
        self.waypoints = []  # [(x, y, mission_type, radius, params), ...]
        self.current_waypoint_index = 0

        # 미리 정의된 웨이포인트 설정
        self.setup_predefined_waypoints()

        # ========== 센서 데이터 ==========
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.lidar_distances = np.zeros(201, dtype=np.float32)
        self.max_lidar_distance = 100.0
        self.current_image = None

        # ========== 탐지 결과 ==========
        self.detected_objects = []

        # ========== 제어 파라미터 ==========
        self.thrust_scale = 1000.0
        self.v_scale = 1.0
        self.w_scale = -1.0

        # ========== ONNX 모델 (미션 4용) ==========
        self.init_onnx_model()
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

        # ========== 미션별 상태 변수 ==========
        # Mission 2 (Circle) 상태
        self.circle_start_time = None
        self.circle_initial_heading = None
        self.total_rotation = 0.0
        self.previous_heading = None

        # ========== 시각화 설정 ==========
        self.setup_visualization()

        # ========== 타이머 ==========
        self.timer = self.create_timer(0.05, self.main_control_loop)  # 20Hz

        self.get_logger().info("✓ 초기화 완료!")
        self.get_logger().info("=" * 80)

    def init_detection_system(self):
        """NanoOWL 탐지 시스템 초기화"""
        self.get_logger().info("NanoOWL 모델 로딩 중...")
        model_name = 'google/owlvit-base-patch32'
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None
        )

        # 미션별 탐지 쿼리 정의
        self.detection_queries = {
            MissionType.PASS_BETWEEN_BUOYS: {
                'queries': [
                    "a red cone buoy", "a red conical marker", "a cone-shaped red buoy",
                    "a green cone buoy", "a green conical marker", "a cone-shaped green buoy"
                ],
                'label_mapping': {
                    0: "red_cone", 1: "red_cone", 2: "red_cone",
                    3: "green_cone", 4: "green_cone", 5: "green_cone"
                }
            },
            MissionType.CIRCLE_BUOY: {
                'queries': [
                    "a blue buoy", "a floating blue marker", "a blue navigation buoy",
                    "a round blue object in the water"
                ],
                'label_mapping': {
                    0: "blue_buoy", 1: "blue_buoy", 2: "blue_buoy", 3: "blue_buoy"
                }
            }
        }

        # 텍스트 인코딩 미리 수행
        for mission_type, query_info in self.detection_queries.items():
            self.get_logger().info(f"{mission_type.name} 쿼리 인코딩 중...")
            query_info['text_encodings'] = self.predictor.encode_text(query_info['queries'])

        # 탐지 파라미터
        self.detection_threshold = 0.1
        self.min_box_area = 500
        self.max_box_area = 80000

        self.get_logger().info("✓ NanoOWL 모델 로딩 완료")

    def init_depth_estimation(self):
        """깊이 추정 시스템 초기화"""
        self.get_logger().info("MiDaS 깊이 추정 모델 로딩 중...")
        self.depth_estimator = MiDaSHybridDepthEstimator()

        # 깊이 필터링 파라미터
        self.min_depth_threshold = 0.0
        self.max_depth_threshold = 50.0  # 50m 이상은 노이즈로 간주

        self.get_logger().info("✓ 깊이 추정 모델 로딩 완료")

    def init_sensor_manager(self):
        """센서 데이터 관리자 초기화"""
        self.sensor_manager = SensorDataManager()
        self.reference_point_set = False
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0

    def init_obstacle_avoidance(self):
        """장애물 회피 컨트롤러 초기화"""
        self.avoidance_controller = AvoidanceController(
            boat_width=5.0,
            boat_height=50.0,
            max_lidar_distance=100.0,
            los_delta=10.0,
            los_lookahead_min=30.0,
            los_lookahead_max=80.0,
            filter_alpha=0.5
        )

    def init_onnx_model(self):
        """ONNX 모델 초기화 (미션 4용)"""
        self.get_logger().info("ONNX 모델 로딩 중...")
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray-9558758.onnx'
        try:
            self.onnx_session = ort.InferenceSession(self.model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.get_logger().info("✓ ONNX 모델 로딩 완료")
        except Exception as e:
            self.get_logger().error(f"ONNX 모델 로딩 실패: {e}")
            self.onnx_session = None

    def setup_subscribers(self):
        """ROS2 구독자 설정"""
        self.create_subscription(Image, '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
                                self.image_callback, 10)
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
                                self.lidar_callback, 10)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix',
                                self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data',
                                self.imu_callback, 10)
        self.create_subscription(Point, '/vrx/waypoint',
                                self.waypoint_callback, 10)

    def setup_publishers(self):
        """ROS2 발행자 설정"""
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.mission_status_pub = self.create_publisher(String, '/vrx/mission_status', 10)
        self.detections_pub = self.create_publisher(Float64MultiArray, '/vrx/detections', 10)
        self.viz_image_pub = self.create_publisher(Image, '/vrx/visualization', 10)

    def setup_predefined_waypoints(self):
        """미리 정의된 웨이포인트 설정"""
        predefined_waypoints = [
            (40, 80, MissionType.PASS_BETWEEN_BUOYS, 10.0, {}),
            (42, 100, MissionType.CIRCLE_BUOY, 10.0, {'rotation_direction': 1, 'circle_radius': 15.0}),
            (0, 165, MissionType.WAYPOINT_FOLLOW, 10.0, {}),
            (0, 0, MissionType.OBSTACLE_AVOID, 10.0, {})
        ]

        for x, y, mission_type, radius, params in predefined_waypoints:
            waypoint = {
                'x': x,
                'y': y,
                'mission_type': mission_type,
                'radius': radius,
                'params': params
            }
            self.waypoints.append(waypoint)
            self.get_logger().info(f"웨이포인트 추가: {mission_type.name} at ({x:.1f}, {y:.1f})")

    def setup_visualization(self):
        """시각화 창 및 트랙바 설정"""
        # 메인 시각화 창
        cv2.namedWindow('VRX Mission Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VRX Mission Control', 1280, 720)

        # 깊이 맵 시각화 창
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth Map', 640, 480)

        # 제어 파라미터 트랙바 창
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parameters', 600, 400)

        # 탐지 파라미터 트랙바
        cv2.createTrackbar('Detect Threshold', 'Parameters',
                          int(self.detection_threshold * 1000), 1000, self._dummy_callback)
        cv2.createTrackbar('Min Box Area', 'Parameters',
                          self.min_box_area, 50000, self._dummy_callback)
        cv2.createTrackbar('Max Box Area', 'Parameters',
                          self.max_box_area, 150000, self._dummy_callback)
        cv2.createTrackbar('Max Depth (m)', 'Parameters',
                          int(self.max_depth_threshold), 200, self._dummy_callback)

        # 제어 파라미터 트랙바
        cv2.createTrackbar('Thrust Scale', 'Parameters',
                          int(self.thrust_scale), 3000, self._dummy_callback)
        cv2.createTrackbar('Forward Speed x100', 'Parameters',
                          50, 100, self._dummy_callback)  # 0.0 ~ 1.0
        cv2.createTrackbar('Steering Gain x1000', 'Parameters',
                          3, 20, self._dummy_callback)  # 0.000 ~ 0.020

        self.get_logger().info("✓ 시각화 창 및 트랙바 설정 완료")

    def _dummy_callback(self, val):
        """트랙바 콜백 (빈 함수)"""
        pass

    def update_parameters_from_trackbars(self):
        """트랙바에서 파라미터 업데이트"""
        self.detection_threshold = cv2.getTrackbarPos('Detect Threshold', 'Parameters') / 1000.0
        self.min_box_area = cv2.getTrackbarPos('Min Box Area', 'Parameters')
        self.max_box_area = cv2.getTrackbarPos('Max Box Area', 'Parameters')
        self.max_depth_threshold = float(cv2.getTrackbarPos('Max Depth (m)', 'Parameters'))
        self.thrust_scale = float(cv2.getTrackbarPos('Thrust Scale', 'Parameters'))

    # ========== 센서 콜백 함수들 ==========

    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

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
        self.angular_velocity_y = min(max(current_angular_velocity[2], -180), 180)

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

                idx = int(angle_deg + 100)
                idx = max(0, min(200, idx))
                raw_ranges[idx] = distance

        self.lidar_distances = raw_ranges.astype(np.float32)

    def waypoint_callback(self, msg):
        """
        웨이포인트 콜백
        trajectory_viz.py에서 클릭한 지점을 받아 웨이포인트에 추가
        """
        # 웨이포인트 순서에 따라 미션 타입 자동 할당
        mission_sequence = [
            MissionType.PASS_BETWEEN_BUOYS,
            MissionType.CIRCLE_BUOY,
            MissionType.WAYPOINT_FOLLOW,
            MissionType.OBSTACLE_AVOID
        ]

        waypoint_count = len(self.waypoints)
        mission_type = mission_sequence[min(waypoint_count, len(mission_sequence) - 1)]

        # 미션별 기본 파라미터
        params = {}
        if mission_type == MissionType.CIRCLE_BUOY:
            params['rotation_direction'] = 1  # 1: 시계방향, -1: 반시계방향
            params['circle_radius'] = 15.0

        waypoint = {
            'x': msg.y,  # GPS 좌표계 변환
            'y': msg.x,
            'mission_type': mission_type,
            'radius': 20.0,  # 웨이포인트 도달 판정 반경
            'params': params
        }

        self.waypoints.append(waypoint)
        self.get_logger().info(f"웨이포인트 추가: {mission_type.name} at ({msg.y:.1f}, {msg.x:.1f})")

    # ========== 객체 탐지 ==========

    def detect_objects(self):
        """
        현재 미션에 맞는 객체 탐지 수행
        NanoOWL + MiDaS 깊이 필터링
        """
        if self.current_image is None:
            return []

        # 탐지가 필요 없는 미션은 스킵
        if self.current_mission not in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY]:
            return []

        # 트랙바에서 파라미터 업데이트
        self.update_parameters_from_trackbars()

        # 깊이 맵 추정
        depth_map = self.depth_estimator.estimate_depth(self.current_image)

        # 깊이 맵 시각화
        self.visualize_depth_map(depth_map)

        # 프레임 전처리
        frame_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(frame_rgb)

        # 현재 미션에 맞는 쿼리 선택
        query_info = self.detection_queries.get(self.current_mission)
        if query_info is None:
            return []

        # NanoOWL 탐지
        output = self.predictor.predict(
            image=image_pil,
            text=query_info['queries'],
            text_encodings=query_info['text_encodings'],
            threshold=self.detection_threshold
        )

        # 결과 파싱
        detections = []
        for i in range(len(output.labels)):
            score = output.scores[i].item()
            bbox = output.boxes[i].detach().cpu().numpy()
            label_idx = output.labels[i].item()

            x1, y1, x2, y2 = [int(b) for b in bbox]

            # 박스 크기 필터링
            area = (x2 - x1) * (y2 - y1)
            if not (self.min_box_area <= area <= self.max_box_area):
                continue

            # 중심점에서 깊이 추출
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = max(0, min(depth_map.shape[1] - 1, cx))
            cy = max(0, min(depth_map.shape[0] - 1, cy))
            depth = depth_map[cy, cx]

            # 깊이 필터링
            if not (self.min_depth_threshold <= depth <= self.max_depth_threshold):
                continue

            label = query_info['label_mapping'].get(label_idx, "unknown")
            detections.append({
                "label": label,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "depth": float(depth)
            })

        # 클래스별 최고 신뢰도만 선택
        detections = self.select_best_per_class(detections)

        # 탐지 결과 시각화
        self.visualize_detections(detections, depth_map)

        return detections

    def select_best_per_class(self, detections):
        """클래스별로 가장 높은 신뢰도의 객체만 선택"""
        if not detections:
            return []

        best_detections = {}
        for det in detections:
            label = det['label']
            if label not in best_detections or det['confidence'] > best_detections[label]['confidence']:
                best_detections[label] = det

        return list(best_detections.values())

    def visualize_depth_map(self, depth_map):
        """깊이 맵 시각화"""
        # 깊이 맵을 컬러맵으로 변환
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # 정보 표시
        info_text = [
            f"Mission: {self.current_mission.name}",
            f"Max Depth: {self.max_depth_threshold:.1f}m",
            f"Min Depth: {self.min_depth_threshold:.1f}m"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(depth_colored, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

        cv2.imshow('Depth Map', depth_colored)

    def visualize_detections(self, detections, depth_map):
        """탐지 결과 시각화"""
        if self.current_image is None:
            return

        vis_image = self.current_image.copy()

        # 색상 매핑
        colors = {
            "red_cone": (0, 0, 255),      # 빨강
            "green_cone": (0, 255, 0),     # 초록
            "blue_buoy": (255, 0, 0)       # 파랑
        }

        # 탐지 결과 그리기
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            depth = det["depth"]
            cx, cy = det["center"]

            color = colors.get(label, (255, 255, 255))

            # 바운딩 박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

            # 중심점
            cv2.circle(vis_image, (cx, cy), 5, color, -1)

            # 라벨 및 정보
            text = f"{label}: {conf:.2f} | {depth:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # 배경 박스
            cv2.rectangle(vis_image, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1 - 5),
                         (0, 0, 0), -1)
            cv2.putText(vis_image, text, (x1 + 5, y1 - 10),
                       font, font_scale, color, thickness)

        # 미션 정보 표시
        mission_info = [
            f"Mission: {self.current_mission.name}",
            f"Waypoint: {self.current_waypoint_index + 1}/{len(self.waypoints)}",
            f"Detections: {len(detections)}",
            f"Threshold: {self.detection_threshold:.3f}",
            f"Box Area: {self.min_box_area}-{self.max_box_area}",
            f"Max Depth: {self.max_depth_threshold:.1f}m"
        ]

        # 반투명 배경
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0, vis_image)

        # 정보 텍스트
        y_offset = 25
        for text in mission_info:
            cv2.putText(vis_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

        # 화면 표시
        cv2.imshow('VRX Mission Control', vis_image)
        cv2.waitKey(1)

        # ROS 메시지로 발행
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            self.viz_image_pub.publish(viz_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    # ========== 메인 제어 루프 ==========

    def main_control_loop(self):
        """메인 제어 루프 - 모든 미션 처리"""
        # 웨이포인트 확인 및 미션 전환
        self.check_waypoint_transition()

        # 객체 탐지 (필요한 미션만)
        self.detected_objects = self.detect_objects()

        # 현재 미션 실행
        left_thrust, right_thrust = 0.0, 0.0

        if self.current_mission == MissionType.PASS_BETWEEN_BUOYS:
            left_thrust, right_thrust = self.mission_pass_between_buoys()

        elif self.current_mission == MissionType.CIRCLE_BUOY:
            left_thrust, right_thrust = self.mission_circle_buoy()

        elif self.current_mission == MissionType.WAYPOINT_FOLLOW:
            left_thrust, right_thrust = self.mission_waypoint_follow()

        elif self.current_mission == MissionType.OBSTACLE_AVOID:
            left_thrust, right_thrust = self.mission_obstacle_avoid()

        # 스러스터 명령 발행
        self.publish_thrust_commands(left_thrust, right_thrust)

        # 미션 상태 발행
        self.publish_mission_status()

        # 탐지 결과 발행
        self.publish_detections()

    def check_waypoint_transition(self):
        """웨이포인트 도달 확인 및 미션 전환"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return

        current_waypoint = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([current_waypoint['x'], current_waypoint['y']], dtype=np.float32)

        # 현재 위치에서 웨이포인트까지 거리 계산
        distance = np.linalg.norm(self.agent_position - target_pos)

        # 웨이포인트 도달 확인
        if distance < current_waypoint['radius']:
            self.get_logger().info(f"웨이포인트 {self.current_waypoint_index} 도달!")

            # 다음 웨이포인트로 이동
            self.current_waypoint_index += 1

            if self.current_waypoint_index < len(self.waypoints):
                next_waypoint = self.waypoints[self.current_waypoint_index]
                self.current_mission = next_waypoint['mission_type']
                self.mission_params = next_waypoint['params']

                # 미션 전환 시 상태 초기화
                self.reset_mission_state()

                self.get_logger().info(f"미션 전환: {self.current_mission.name}")
            else:
                self.get_logger().info("모든 미션 완료!")

    def reset_mission_state(self):
        """미션 전환 시 상태 초기화"""
        self.circle_start_time = None
        self.circle_initial_heading = None
        self.total_rotation = 0.0
        self.previous_heading = None

    # ========== 미션 1: 부표 사이 지나가기 ==========

    def mission_pass_between_buoys(self):
        """빨간색/초록색 고깔 부표 사이로 지나가기"""
        # 빨간색/초록색 부표 찾기
        red_buoy = None
        green_buoy = None

        for det in self.detected_objects:
            if det['label'] == 'red_cone':
                red_buoy = det
            elif det['label'] == 'green_cone':
                green_buoy = det

        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 두 부표의 중점 계산 (이미지 좌표)
            red_x = red_buoy['center'][0]
            green_x = green_buoy['center'][0]
            midpoint_x = (red_x + green_x) / 2

            # 이미지 중심
            image_center_x = self.current_image.shape[1] / 2

            # 오차 계산
            error = midpoint_x - image_center_x

            # 비례 제어
            steering_gain = 0.003
            forward_speed = 0.5

            steering = error * steering_gain  # 오차를 조향 명령으로 변환
            steering = np.clip(steering, -0.3, 0.3)

            # 스러스터 명령 계산
            left_thrust = (forward_speed + steering) * self.thrust_scale
            right_thrust = (forward_speed - steering) * self.thrust_scale

            self.get_logger().info(
                f"Pass Buoys: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
            )

        else:
            # 부표 미탐지 시 천천히 전진
            left_thrust = 0.2 * self.thrust_scale
            right_thrust = 0.2 * self.thrust_scale
            self.get_logger().warn("부표 미탐지: 천천히 전진")

        return left_thrust, right_thrust

    # ========== 미션 2: 부표 주변 회전 ==========

    def mission_circle_buoy(self):
        """파란색 부표 주변을 회전"""
        # 파란색 부표 찾기
        blue_buoy = None
        for det in self.detected_objects:
            if det['label'] == 'blue_buoy':
                blue_buoy = det
                break

        if not blue_buoy:
            # 부표 미탐지 시 정지
            self.get_logger().warn("파란색 부표 미탐지: 정지")
            return 0.0, 0.0

        # 회전 시작 시간 기록
        if self.circle_start_time is None:
            self.circle_start_time = time.time()
            self.circle_initial_heading = self.agent_heading
            self.previous_heading = self.agent_heading
            self.total_rotation = 0.0

        # 누적 회전 각도 계산
        heading_diff = self.agent_heading - self.previous_heading

        # 각도 차이 정규화 (-180 ~ 180)
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360

        self.total_rotation += abs(heading_diff)
        self.previous_heading = self.agent_heading

        # 360도 회전 완료 확인
        if self.total_rotation >= 350:  # 약간의 여유
            self.get_logger().info("부표 회전 완료!")
            return 0.0, 0.0

        # 부표 중심 위치
        buoy_x = blue_buoy['center'][0]
        image_center_x = self.current_image.shape[1] / 2

        # 부표를 일정 거리에 유지하면서 회전
        error = buoy_x - image_center_x

        # 회전 방향 (미션 파라미터에서 가져오기)
        rotation_direction = self.mission_params.get('rotation_direction', 1)

        # 제어 게인
        turn_rate = 0.3 * rotation_direction  # 회전 속도
        centering_gain = 0.002  # 부표 중심 유지 게인
        forward_speed = 0.3

        # 부표를 중앙에 유지하기 위한 조정
        centering_adjustment = -error * centering_gain

        # 스러스터 명령
        left_thrust = (forward_speed + turn_rate + centering_adjustment) * self.thrust_scale
        right_thrust = (forward_speed - turn_rate - centering_adjustment) * self.thrust_scale

        self.get_logger().info(
            f"Circle Buoy: rotation={self.total_rotation:.1f}°, buoy_x={buoy_x:.1f}, error={error:.1f}"
        )

        return left_thrust, right_thrust

    # ========== 미션 3: 웨이포인트 추종 ==========

    def mission_waypoint_follow(self):
        """단순 웨이포인트 추종"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return 0.0, 0.0

        # 목표 웨이포인트
        target_waypoint = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([target_waypoint['x'], target_waypoint['y']], dtype=np.float32)

        # 목표까지의 벡터
        delta = target_pos - self.agent_position
        distance = np.linalg.norm(delta)

        if distance < 1.0:
            return 0.0, 0.0

        # 목표 방향 계산
        target_heading = np.degrees(np.arctan2(delta[0], delta[1]))
        if target_heading < 0:
            target_heading += 360

        # 헤딩 오차 계산
        heading_error = target_heading - self.agent_heading

        # 각도 정규화
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        # 비례 제어
        steering_gain = 0.01
        forward_speed = 0.5

        steering = heading_error * steering_gain
        steering = np.clip(steering, -0.5, 0.5)

        # 스러스터 명령
        left_thrust = (forward_speed + steering) * self.thrust_scale
        right_thrust = (forward_speed - steering) * self.thrust_scale

        self.get_logger().info(
            f"Waypoint Follow: dist={distance:.1f}m, heading_err={heading_error:.1f}°"
        )

        return left_thrust, right_thrust

    # ========== 미션 4: 장애물 회피 ==========

    def mission_obstacle_avoid(self):
        """ONNX 모델 + 알고리즘 하이브리드 장애물 회피"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return 0.0, 0.0

        # 목표 웨이포인트
        target_waypoint = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([target_waypoint['x'], target_waypoint['y']], dtype=np.float32)

        # LOS target 계산
        waypoint_list = [[wp['x'], wp['y']] for wp in self.waypoints]
        los_target = self.avoidance_controller.get_los_target(
            self.agent_position, waypoint_list, self.current_waypoint_index
        )

        # 장애물 확인 및 제어
        use_direct_control, linear_velocity, angular_velocity, _ = \
            self.avoidance_controller.check_obstacles_and_get_control(
                self.agent_position, los_target, self.agent_heading,
                self.lidar_distances, self.get_lidar_distance_at_angle,
                self.get_onnx_control
            )

        # 필터 적용
        filtered_linear, filtered_angular = self.avoidance_controller.apply_filters(
            linear_velocity, angular_velocity
        )

        # 이전 명령 업데이트
        self.previous_moment_input = filtered_angular
        self.previous_force_input = filtered_linear

        # 스러스터 계산
        left_thrust, right_thrust = self.calculate_thruster_commands(
            filtered_linear, filtered_angular
        )

        # 스러스터 필터 적용
        left_thrust, right_thrust = self.avoidance_controller.apply_thrust_filters(
            left_thrust, right_thrust
        )

        mode = "DIRECT" if use_direct_control else "ONNX"
        self.get_logger().info(
            f"Obstacle Avoid [{mode}]: linear={filtered_linear:.3f}, angular={filtered_angular:.3f}"
        )

        return left_thrust, right_thrust

    def get_lidar_distance_at_angle(self, angle_deg):
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

    def get_onnx_control(self):
        """ONNX 모델 제어"""
        if self.onnx_session is None:
            return 0.0, 0.0

        # 웨이포인트 위치
        current_target, previous_target, next_target = self.get_waypoint_positions()

        # 관측값 구성
        observation_values = []

        # LiDAR 데이터
        for i in range(len(self.lidar_distances)):
            observation_values.append(float(self.lidar_distances[i]))

        # 헤딩
        observation_values.append(float(self.agent_heading))

        # 각속도
        observation_values.append(float(self.angular_velocity_y))

        # 위치 및 웨이포인트
        for val in [self.agent_position, current_target, previous_target, next_target]:
            for i in range(2):
                v = float(val[i])
                if np.isinf(v) or np.isnan(v):
                    v = 0.0
                observation_values.append(v)

        # 이전 입력
        observation_values.append(float(self.previous_moment_input))
        observation_values.append(float(self.previous_force_input))

        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([observation_array, observation_array]).reshape(1, 426)

        # ONNX 추론
        try:
            outputs = self.onnx_session.run(None, {self.onnx_input_name: stacked_input})

            if len(outputs) > 2 and outputs[2] is not None:
                linear_velocity = max(min(outputs[2][0][1] * self.v_scale, 1), 0.01)
                angular_velocity = max(min(outputs[2][0][0] * self.w_scale, 1.0), -1.0)
            else:
                linear_velocity = 0.0
                angular_velocity = 0.0
        except Exception as e:
            self.get_logger().error(f"ONNX 추론 오류: {e}")
            linear_velocity = 0.0
            angular_velocity = 0.0

        return linear_velocity, angular_velocity

    def get_waypoint_positions(self):
        """웨이포인트 위치 반환"""
        if not self.waypoints:
            zeros = np.zeros(2, dtype=np.float32)
            return zeros, zeros, zeros

        if self.current_waypoint_index < len(self.waypoints):
            current = self.waypoints[self.current_waypoint_index]
            current_target = np.array([current['x'], current['y']], dtype=np.float32)
        else:
            current_target = np.zeros(2, dtype=np.float32)

        if self.current_waypoint_index > 0:
            prev = self.waypoints[self.current_waypoint_index - 1]
            previous_target = np.array([prev['x'], prev['y']], dtype=np.float32)
        else:
            previous_target = np.zeros(2, dtype=np.float32)

        if self.current_waypoint_index + 1 < len(self.waypoints):
            next_wp = self.waypoints[self.current_waypoint_index + 1]
            next_target = np.array([next_wp['x'], next_wp['y']], dtype=np.float32)
        else:
            next_target = current_target.copy()

        return current_target, previous_target, next_target

    def calculate_thruster_commands(self, linear_velocity, angular_velocity):
        """스러스터 명령 계산"""
        forward_thrust = linear_velocity * self.thrust_scale
        turn_thrust = angular_velocity * self.thrust_scale
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        return left_thrust, right_thrust

    # ========== 발행 함수들 ==========

    def publish_thrust_commands(self, left_thrust, right_thrust):
        """스러스터 명령 발행"""
        left_msg = Float64()
        left_msg.data = float(left_thrust)
        self.left_thrust_pub.publish(left_msg)

        right_msg = Float64()
        right_msg.data = float(right_thrust)
        self.right_thrust_pub.publish(right_msg)

    def publish_mission_status(self):
        """미션 상태 발행"""
        status_msg = String()
        status_msg.data = f"Mission: {self.current_mission.name}, Waypoint: {self.current_waypoint_index}/{len(self.waypoints)}"
        self.mission_status_pub.publish(status_msg)

    def publish_detections(self):
        """탐지 결과 발행"""
        msg = Float64MultiArray()
        data = [float(len(self.detected_objects))]

        for det in self.detected_objects:
            # 라벨 ID 매핑
            label_to_id = {
                "red_cone": 0,
                "green_cone": 1,
                "blue_buoy": 2
            }
            label_id = label_to_id.get(det['label'], -1)

            data.extend([
                float(label_id),
                float(det['confidence']),
                float(det['bbox'][0]),
                float(det['bbox'][1]),
                float(det['bbox'][2]),
                float(det['bbox'][3]),
                float(det['depth'])
            ])

        msg.data = data
        self.detections_pub.publish(msg)

    def destroy_node(self):
        """노드 종료"""
        self.publish_thrust_commands(0.0, 0.0)
        cv2.destroyAllWindows()
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
