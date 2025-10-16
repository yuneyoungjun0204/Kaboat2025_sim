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
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import time
import onnxruntime as ort

# 유틸리티 모듈
from utils import (
    MiDaSHybridDepthEstimator,
    SensorDataManager,
    AvoidanceController,
    create_tracker
)
from utils.detection_system import DetectionSystem, MissionType
from utils.mission_strategies import MissionManager
from utils.visualization_system import VisualizationSystem
from utils.waypoint_manager import WaypointManager
from utils.ros_communication import ROSCommunicationManager


class VRXMissionController(Node):
    """VRX 통합 미션 제어 노드"""

    def __init__(self):
        super().__init__('vrx_mission_controller')
        self._log_header("VRX 통합 미션 제어 시스템 초기화")

        # 기본 설정
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 컴포넌트 초기화
        self._init_components()

        # ROS2 통신 설정
        self._setup_ros_communication()

        # 센서 데이터 초기화
        self._init_sensor_data()

        # ONNX 모델 초기화
        self._init_onnx_model()

        # 타이머 설정
        self.timer = self.create_timer(0.05, self.main_control_loop)  # 20Hz

        self.get_logger().info("✓ 초기화 완료!")
        self.get_logger().info("=" * 80)

    def _log_header(self, message: str):
        """헤더 로그 출력"""
        self.get_logger().info("=" * 80)
        self.get_logger().info(message)
        self.get_logger().info("=" * 80)

    def _init_components(self):
        """컴포넌트 초기화"""
        self.get_logger().info("컴포넌트 초기화 중...")

        # 깊이 추정
        self.depth_estimator = MiDaSHybridDepthEstimator()

        # 객체 탐지 시스템
        self.detection_system = DetectionSystem(
            self.depth_estimator,
            device=self.device
        )

        # 센서 데이터 관리자
        self.sensor_manager = SensorDataManager()
        self.reference_point_set = False

        # 장애물 회피 컨트롤러
        self.avoidance_controller = AvoidanceController(
            boat_width=5.0,
            boat_height=50.0,
            max_lidar_distance=100.0,
            los_delta=10.0,
            los_lookahead_min=30.0,
            los_lookahead_max=80.0,
            filter_alpha=0.5
        )

        # 미션 관리자
        self.mission_manager = MissionManager(
            thrust_scale=1000.0,
            avoidance_controller=self.avoidance_controller
        )

        # 웨이포인트 관리자
        self.waypoint_manager = WaypointManager()
        self.waypoint_manager.setup_predefined_waypoints()

        # 시각화 시스템
        self.visualization = VisualizationSystem()

        # IMM-PDAF 트래커 (20Hz = 1/20 = 0.05초)
        self.tracker = create_tracker(fps=20.0, max_coast_frames=10)

        self.get_logger().info("✓ 컴포넌트 초기화 완료")

    def _setup_ros_communication(self):
        """ROS2 통신 설정"""
        self.get_logger().info("ROS2 통신 설정 중...")

        self.ros_comm = ROSCommunicationManager(self)

        # 서브스크라이버 설정
        self.ros_comm.setup_subscribers({
            'image': self.image_callback,
            'lidar': self.lidar_callback,
            'gps': self.gps_callback,
            'imu': self.imu_callback,
            'waypoint': self.waypoint_callback
        })

        # 퍼블리셔 설정
        self.ros_comm.setup_publishers()

        self.get_logger().info("✓ ROS2 통신 설정 완료")

    def _init_sensor_data(self):
        """센서 데이터 초기화"""
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.lidar_distances = np.zeros(201, dtype=np.float32)
        self.max_lidar_distance = 100.0
        self.current_image = None
        self.detected_objects = []
        self.raw_detections = []  # 원본 탐지 결과 저장

    def _init_onnx_model(self):
        """ONNX 모델 초기화 (미션 4용)"""
        self.get_logger().info("ONNX 모델 로딩 중...")
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray-9558758.onnx'
        try:
            self.onnx_session = ort.InferenceSession(self.model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.previous_moment_input = 0.0
            self.previous_force_input = 0.0
            self.v_scale = 1.0
            self.w_scale = -1.0
            self.get_logger().info("✓ ONNX 모델 로딩 완료")
        except Exception as e:
            self.get_logger().error(f"ONNX 모델 로딩 실패: {e}")
            self.onnx_session = None

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
            self.agent_position = np.array([gps_data['utm_x'], gps_data['utm_y']], dtype=np.float32)
            if not self.reference_point_set:
                self.reference_point_set = True

    def imu_callback(self, msg):
        """IMU 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.agent_heading = imu_data['yaw_degrees']
        if self.agent_heading < 0:
            self.agent_heading += 360.0

        current_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
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
        """웨이포인트 콜백"""
        mission_sequence = [
            MissionType.PASS_BETWEEN_BUOYS,
            MissionType.CIRCLE_BUOY,
            MissionType.OBSTACLE_AVOID,
            MissionType.OBSTACLE_AVOID
        ]

        waypoint_count = self.waypoint_manager.get_total_waypoints()
        mission_type = mission_sequence[min(waypoint_count, len(mission_sequence) - 1)]

        params = {}
        if mission_type == MissionType.CIRCLE_BUOY:
            params['rotation_direction'] = 1
            params['circle_radius'] = 15.0

        self.waypoint_manager.add_waypoint(msg.y, msg.x, mission_type, 20.0, params)
        self.get_logger().info(f"웨이포인트 추가: {mission_type.name} at ({msg.y:.1f}, {msg.x:.1f})")

    # ========== 메인 제어 루프 ==========

    def main_control_loop(self):
        """메인 제어 루프"""
        # 트랙바에서 파라미터 업데이트
        params = self.visualization.update_parameters_from_trackbars()

        # DetectionSystem에는 탐지 관련 파라미터만 전달 (thrust_scale, IMM-PDAF 파라미터 제외)
        detection_params = {k: v for k, v in params.items()
                          if k not in ['thrust_scale', 'max_coast_frames', 'gate_threshold']}
        self.detection_system.update_parameters(**detection_params)

        # MissionManager에는 thrust_scale만 업데이트
        if 'thrust_scale' in params:
            self.mission_manager.update_thrust_scale(params['thrust_scale'])

        # IMM-PDAF 트래커 파라미터 동적 업데이트
        if 'max_coast_frames' in params:
            self.tracker.max_coast_frames = params['max_coast_frames']
        if 'gate_threshold' in params:
            self.tracker.gate_threshold = params['gate_threshold']

        # 웨이포인트 전환 확인
        self._check_waypoint_transition()

        # 현재 미션 타입 가져오기
        current_mission_type = self.waypoint_manager.get_current_mission_type()
        if current_mission_type is None:
            self.ros_comm.publish_thrust_commands(0.0, 0.0)
            return

        # 객체 탐지 (필요한 미션만)
        self.raw_detections = self.detection_system.detect_objects(
            self.current_image, current_mission_type
        )

        # IMM-PDAF 트래커로 강건한 추적
        self.tracker.predict_tracks()
        self.tracker.update_tracks(self.raw_detections)
        self.tracker.prune_tracks()

        # 추적된 객체 사용 (더 부드럽고 강건함)
        self.detected_objects = self.tracker.get_tracked_objects()

        # 디버그: 원본 탐지 vs 추적 결과 비교
        if len(self.raw_detections) > 0 or len(self.detected_objects) > 0:
            self.get_logger().info(
                f"Detection: raw={len(self.raw_detections)}, tracked={len(self.detected_objects)}"
            )

        # 시각화
        self._visualize()

        # 미션 실행
        left_thrust, right_thrust = self._execute_current_mission(current_mission_type)

        # 명령 발행
        self.ros_comm.publish_thrust_commands(left_thrust, right_thrust)
        self.ros_comm.publish_mission_status(
            current_mission_type.name,
            self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints()
        )
        self.ros_comm.publish_detections(self.detected_objects)

    def _check_waypoint_transition(self):
        """웨이포인트 도달 확인 및 미션 전환"""
        next_waypoint = self.waypoint_manager.check_waypoint_reached(self.agent_position)

        if next_waypoint is not None:
            if 'completed' in next_waypoint:
                self.get_logger().info("모든 미션 완료!")
            else:
                new_mission_type = next_waypoint['mission_type']
                self.mission_manager.set_mission(new_mission_type)
                self.get_logger().info(f"미션 전환: {new_mission_type.name}")

    def _execute_current_mission(self, mission_type: MissionType) -> tuple:
        """현재 미션 실행"""
        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                logger=self.get_logger()
            )

        elif mission_type == MissionType.CIRCLE_BUOY:
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                agent_heading=self.agent_heading,
                mission_params=self.waypoint_manager.get_current_mission_params(),
                logger=self.get_logger()
            )

        elif mission_type == MissionType.OBSTACLE_AVOID:
            return self.mission_manager.execute_mission(
                mission_type,
                agent_position=self.agent_position,
                agent_heading=self.agent_heading,
                waypoints=self.waypoint_manager.waypoints,
                current_waypoint_index=self.waypoint_manager.get_waypoint_index(),
                lidar_distances=self.lidar_distances,
                get_lidar_distance_func=self.get_lidar_distance_at_angle,
                get_onnx_control_func=self.get_onnx_control,
                logger=self.get_logger()
            )

        return 0.0, 0.0

    def _visualize(self):
        """시각화"""
        if self.current_image is None:
            return

        current_mission_type = self.waypoint_manager.get_current_mission_type()
        if current_mission_type is None:
            return

        # 깊이 맵 시각화
        depth_map = self.depth_estimator.estimate_depth(self.current_image)
        if depth_map is not None:
            self.visualization.visualize_depth_map(depth_map, current_mission_type.name)

        # 탐지 결과 시각화 (원본 탐지 + 추적 결과)
        self.visualization.visualize_detections(
            self.current_image,
            self.detected_objects,
            current_mission_type.name,
            self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints(),
            raw_detections=self.raw_detections,
            bridge=self.bridge,
            viz_image_pub=self.ros_comm.publishers.get('viz_image')
        )

    # ========== ONNX 제어 헬퍼 함수 ==========

    def get_lidar_distance_at_angle(self, angle_deg: float) -> float:
        """주어진 각도에서 LiDAR 거리 가져오기"""
        while angle_deg > 100:
            angle_deg -= 360
        while angle_deg < -100:
            angle_deg += 360

        if -180 <= angle_deg <= 180:
            idx = int(angle_deg + 100)
            idx = max(0, min(200, idx))
            return self.lidar_distances[idx]
        return self.max_lidar_distance

    def get_onnx_control(self) -> tuple:
        """ONNX 모델 제어"""
        if self.onnx_session is None:
            return 0.0, 0.0

        current_target, previous_target, next_target = self._get_waypoint_positions()

        # 관측값 구성
        observation_values = list(self.lidar_distances) + [
            float(self.agent_heading),
            float(self.angular_velocity_y)
        ]

        for val in [self.agent_position, current_target, previous_target, next_target]:
            for i in range(2):
                v = float(val[i])
                if np.isinf(v) or np.isnan(v):
                    v = 0.0
                observation_values.append(v)

        observation_values.extend([
            float(self.previous_moment_input),
            float(self.previous_force_input)
        ])

        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([observation_array, observation_array]).reshape(1, 426)

        try:
            outputs = self.onnx_session.run(None, {self.onnx_input_name: stacked_input})
            if len(outputs) > 2 and outputs[2] is not None:
                linear_velocity = max(min(outputs[4][0][1] * self.v_scale, 1), 0.00)
                angular_velocity = max(min(outputs[4][0][0] * self.w_scale, 1.0), -1.0)
            else:
                linear_velocity = 0.0
                angular_velocity = 0.0
        except Exception as e:
            self.get_logger().error(f"ONNX 추론 오류: {e}")
            linear_velocity = 0.0
            angular_velocity = 0.0

        return linear_velocity, angular_velocity

    def _get_waypoint_positions(self) -> tuple:
        """웨이포인트 위치 반환"""
        waypoints = self.waypoint_manager.waypoints
        current_idx = self.waypoint_manager.get_waypoint_index()

        zeros = np.zeros(2, dtype=np.float32)

        if current_idx < len(waypoints):
            current = waypoints[current_idx]
            current_target = np.array([current['x'], current['y']], dtype=np.float32)
        else:
            current_target = zeros

        if current_idx > 0:
            prev = waypoints[current_idx - 1]
            previous_target = np.array([prev['x'], prev['y']], dtype=np.float32)
        else:
            previous_target = zeros

        if current_idx + 1 < len(waypoints):
            next_wp = waypoints[current_idx + 1]
            next_target = np.array([next_wp['x'], next_wp['y']], dtype=np.float32)
        else:
            next_target = current_target.copy()

        return current_target, previous_target, next_target

    def destroy_node(self):
        """노드 종료"""
        self.ros_comm.publish_thrust_commands(0.0, 0.0)
        self.visualization.cleanup()
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
