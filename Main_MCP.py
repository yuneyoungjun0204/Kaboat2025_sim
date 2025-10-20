#!/usr/bin/env python3
"""
VRX 통합 미션 제어 시스템 (Main Mission Control Platform)
- NanoOWL 기반 객체 탐지 + MiDaS 깊이 필터링
- 웨이포인트 기반 미션 전환
- 4가지 미션: 부표 사이 지나가기 → 부표 회전 → 웨이포인트 추종 → 장애물 회피
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from typing import Tuple

# 유틸리티 모듈
from utils import (
    MiDaSHybridDepthEstimator,
    SensorDataManager,
    AvoidanceController,
    create_tracker,
    Constants,
    ParameterManager,
    SensorCallbackHandler,
    ONNXController,
    MissionExecutor
)
from utils.detection_system import DetectionSystem, MissionType
from utils.mission_strategies_new import MissionManager
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

        # 타이머 설정
        self.timer = self.create_timer(Constants.MAIN_LOOP_PERIOD, self.main_control_loop)

        # 강제 장애물 회피 모드 상태
        self.original_mission_type = None

        self.get_logger().info("✓ 초기화 완료!")
        self.get_logger().info("=" * 80)

    def _log_header(self, message: str) -> None:
        """헤더 로그 출력"""
        self.get_logger().info("=" * 80)
        self.get_logger().info(message)
        self.get_logger().info("=" * 80)

    def _init_components(self) -> None:
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
        sensor_manager = SensorDataManager()
        self.sensor_handler = SensorCallbackHandler(
            self.bridge, sensor_manager, self.get_logger()
        )

        # 장애물 회피 컨트롤러
        self.avoidance_controller = AvoidanceController(
            boat_width=Constants.BOAT_WIDTH,
            boat_height=Constants.BOAT_HEIGHT,
            max_lidar_distance=Constants.MAX_LIDAR_DISTANCE,
            los_delta=Constants.LOS_DELTA,
            los_lookahead_min=Constants.LOS_LOOKAHEAD_MIN,
            los_lookahead_max=Constants.LOS_LOOKAHEAD_MAX,
            filter_alpha=Constants.FILTER_ALPHA
        )

        # 미션 관리자
        self.mission_manager = MissionManager(
            thrust_scale=Constants.DEFAULT_THRUST_SCALE,
            avoidance_controller=self.avoidance_controller
        )

        # 웨이포인트 관리자
        self.waypoint_manager = WaypointManager()
        self.waypoint_manager.setup_predefined_waypoints()

        # 미션 실행자
        self.mission_executor = MissionExecutor(
            self.mission_manager,
            self.waypoint_manager
        )

        # 시각화 시스템
        self.visualization = VisualizationSystem()

        # 파라미터 관리자
        self.param_manager = ParameterManager(self.visualization)

        # ONNX 컨트롤러
        self.onnx_controller = ONNXController(
            Constants.ONNX_MODEL_PATH,
            self.get_logger()
        )

        # IMM-PDAF 트래커
        self.tracker = create_tracker(
            fps=Constants.TRACKER_FPS,
            max_coast_frames=Constants.MAX_COAST_FRAMES
        )

        # 탐지 결과 저장
        self.detected_objects = []
        self.raw_detections = []

        self.get_logger().info("✓ 컴포넌트 초기화 완료")

    def _setup_ros_communication(self) -> None:
        """ROS2 통신 설정"""
        self.get_logger().info("ROS2 통신 설정 중...")

        self.ros_comm = ROSCommunicationManager(self)

        # 서브스크라이버 설정
        self.ros_comm.setup_subscribers({
            'image': self.sensor_handler.image_callback,
            'lidar': self.sensor_handler.lidar_callback,
            'gps': self.sensor_handler.gps_callback,
            'imu': self.sensor_handler.imu_callback,
            'waypoint': self._waypoint_callback_wrapper
        })

        # 퍼블리셔 설정
        self.ros_comm.setup_publishers()

        self.get_logger().info("✓ ROS2 통신 설정 완료")

    def _waypoint_callback_wrapper(self, msg: Point) -> None:
        """웨이포인트 콜백 래퍼"""
        self.sensor_handler.waypoint_callback(msg, self._add_waypoint_from_click)

    def _add_waypoint_from_click(self, msg: Point) -> None:
        """클릭한 위치에서 웨이포인트 추가"""
        mission_sequence = [
            MissionType.OBSTACLE_AVOID,
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

        self.waypoint_manager.add_waypoint(
            msg.y, msg.x, mission_type,
            Constants.DEFAULT_WAYPOINT_RADIUS,
            params
        )
        self.get_logger().info(f"웨이포인트 추가: {mission_type.name} at ({msg.y:.1f}, {msg.x:.1f})")

    def main_control_loop(self) -> None:
        """메인 제어 루프"""
        try:
            # 파라미터 업데이트
            self._update_system_parameters()

            # 웨이포인트 전환 확인
            self._check_waypoint_transition()

            # 현재 미션 확인
            current_mission_type = self.waypoint_manager.get_current_mission_type()
            if current_mission_type is None:
                self.ros_comm.publish_thrust_commands(0.0, 0.0)
                return

            # 강제 장애물 회피 모드 처리
            effective_mission_type = self._handle_force_obstacle_mode(current_mission_type)

            # 객체 탐지 및 추적
            self._perform_detection_and_tracking(effective_mission_type)

            # 시각화
            self._visualize(effective_mission_type)

            # 미션 실행
            left_thrust, right_thrust = self._execute_current_mission(effective_mission_type)

            # 명령 발행
            self._publish_control_commands(left_thrust, right_thrust, effective_mission_type)

        except Exception as e:
            self.get_logger().error(f"제어 루프 오류: {e}")
            self.ros_comm.publish_thrust_commands(0.0, 0.0)

    def _update_system_parameters(self) -> None:
        """시스템 파라미터 업데이트"""
        # 모든 파라미터 업데이트
        self.param_manager.update_all_parameters()

        # DetectionSystem 파라미터 업데이트
        detection_params = self.param_manager.get_detection_parameters()
        self.detection_system.update_parameters(
            **{k: v for k, v in detection_params.items() if v is not None}
        )

        # MissionManager thrust_scale 업데이트
        thrust_scale = self.param_manager.get_thrust_scale()
        if thrust_scale is not None:
            self.mission_manager.update_thrust_scale(thrust_scale)

        # Tracker 파라미터 업데이트
        tracker_params = self.param_manager.get_tracker_parameters()
        if tracker_params.get('max_coast_frames') is not None:
            self.tracker.max_coast_frames = tracker_params['max_coast_frames']
        if tracker_params.get('gate_threshold') is not None:
            self.tracker.gate_threshold = tracker_params['gate_threshold']

    def _handle_force_obstacle_mode(self, current_mission_type: MissionType) -> MissionType:
        """강제 장애물 회피 모드 처리"""
        force_obstacle_avoid = self.param_manager.is_force_obstacle_avoid()

        if force_obstacle_avoid:
            if self.original_mission_type is None:
                self.original_mission_type = current_mission_type
                self.get_logger().info("🚨 강제 장애물 회피 모드 진입!")
            return MissionType.OBSTACLE_AVOID

        elif self.original_mission_type is not None:
            self.get_logger().info(f"✅ 강제 장애물 회피 모드 해제 -> {self.original_mission_type.name} 복귀")
            self.original_mission_type = None

        return current_mission_type

    def _perform_detection_and_tracking(self, mission_type: MissionType) -> None:
        """객체 탐지 및 추적 수행"""
        # 장애물 회피 및 웨이포인트 추종 미션에서는 탐지/추적 스킵
        if mission_type in [MissionType.OBSTACLE_AVOID, MissionType.WAYPOINT_FOLLOW]:
            self.raw_detections = []
            self.detected_objects = []
            return

        # 객체 탐지 (부표 관련 미션만)
        self.raw_detections = self.detection_system.detect_objects(
            self.sensor_handler.current_image, mission_type
        )

        # IMM-PDAF 트래커로 강건한 추적
        self.tracker.predict_tracks()
        self.tracker.update_tracks(self.raw_detections)
        self.tracker.prune_tracks()

        # 추적된 객체 사용
        self.detected_objects = self.tracker.get_tracked_objects()

        # 디버그 로그
        if len(self.raw_detections) > 0 or len(self.detected_objects) > 0:
            self.get_logger().info(
                f"Detection: raw={len(self.raw_detections)}, tracked={len(self.detected_objects)}"
            )

    def _check_waypoint_transition(self) -> None:
        """웨이포인트 도달 확인 및 미션 전환"""
        next_waypoint = self.waypoint_manager.check_waypoint_reached(
            self.sensor_handler.agent_position
        )

        if next_waypoint is not None:
            if 'completed' in next_waypoint:
                self.get_logger().info("모든 미션 완료!")
            else:
                new_mission_type = next_waypoint['mission_type']
                self.mission_manager.set_mission(new_mission_type)
                self.get_logger().info(f"미션 전환: {new_mission_type.name}")

    def _execute_current_mission(self, mission_type: MissionType) -> Tuple[float, float]:
        """현재 미션 실행"""
        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            mission_params = self.param_manager.get_mission_parameters(mission_type)
            return self.mission_executor.execute_pass_between_buoys(
                self.detected_objects,
                self.sensor_handler.current_image,
                self.raw_detections,
                mission_params,
                self.get_logger()
            )

        elif mission_type == MissionType.CIRCLE_BUOY:
            waypoint_params = self.waypoint_manager.get_current_mission_params()
            mission_params = self.param_manager.get_mission_parameters(
                mission_type, waypoint_params
            )
            return self.mission_executor.execute_circle_buoy(
                self.detected_objects,
                self.sensor_handler.current_image,
                self.sensor_handler.agent_heading,
                mission_params,
                self.raw_detections,
                self.get_logger()
            )

        elif mission_type == MissionType.OBSTACLE_AVOID:
            return self.mission_executor.execute_obstacle_avoid(
                self.sensor_handler.agent_position,
                self.sensor_handler.agent_heading,
                self.sensor_handler.lidar_distances,
                self.sensor_handler.get_lidar_distance_at_angle,
                self._get_onnx_control,
                self.param_manager.is_force_obstacle_avoid(),
                self.sensor_handler.manual_target_x,
                self.sensor_handler.manual_target_y,
                self.get_logger()
            )

        return 0.0, 0.0

    def _get_onnx_control(self) -> Tuple[float, float]:
        """ONNX 제어 래퍼"""
        current_target, previous_target, next_target = self.mission_executor.get_waypoint_positions()

        return self.onnx_controller.get_control(
            self.sensor_handler.lidar_distances,
            self.sensor_handler.agent_heading,
            self.sensor_handler.angular_velocity_y,
            self.sensor_handler.agent_position,
            current_target,
            previous_target,
            next_target
        )

    def _publish_control_commands(self, left_thrust: float, right_thrust: float,
                                 mission_type: MissionType) -> None:
        """제어 명령 발행"""
        self.ros_comm.publish_thrust_commands(left_thrust, right_thrust)
        self.ros_comm.publish_mission_status(
            mission_type.name,
            self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints()
        )
        self.ros_comm.publish_detections(self.detected_objects)

    def _visualize(self, mission_type: MissionType) -> None:
        """시각화"""
        if self.sensor_handler.current_image is None:
            return

        # 장애물 회피 미션에서는 시각화 최소화
        if mission_type == MissionType.OBSTACLE_AVOID:
            cv2.waitKey(1)
            return

        # 깊이 맵 시각화 (부표 관련 미션만)
        if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY]:
            depth_map = self.depth_estimator.estimate_depth(self.sensor_handler.current_image)
            if depth_map is not None:
                self.visualization.visualize_depth_map(depth_map, mission_type.name)

        # 탐지 결과 시각화
        if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY]:
            self.visualization.visualize_detections(
                self.sensor_handler.current_image,
                self.detected_objects,
                mission_type.name,
                self.waypoint_manager.get_waypoint_index(),
                self.waypoint_manager.get_total_waypoints(),
                raw_detections=self.raw_detections,
                bridge=self.bridge,
                viz_image_pub=self.ros_comm.publishers.get('viz_image')
            )
        else:
            cv2.waitKey(1)

    def destroy_node(self) -> None:
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
