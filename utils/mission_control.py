#!/usr/bin/env python3
"""
미션 제어 컴포넌트
- MissionLoopExecutor: 제어 루프 실행
- WaypointTransitionHandler: 웨이포인트 전환 관리
- ObstacleAvoidExecutor: 장애물 회피 실행
"""

from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np

from .detection_system import MissionType


class WaypointTransitionHandler:
    """
    웨이포인트 전환 핸들러

    웨이포인트 도달 확인 및 미션 전환 로직을 캡슐화
    """

    def __init__(self, waypoint_manager, mission_manager, logger):
        """
        Args:
            waypoint_manager: 웨이포인트 관리자
            mission_manager: 미션 관리자
            logger: ROS2 logger
        """
        self.waypoint_manager = waypoint_manager
        self.mission_manager = mission_manager
        self.logger = logger
        self.loop_counter = 0

    def check_and_transition(self, agent_position: np.ndarray) -> bool:
        """
        웨이포인트 도달 확인 및 전환

        Args:
            agent_position: 현재 에이전트 위치 [x, y]

        Returns:
            bool: 미션 완료 여부 (모든 웨이포인트 완료)
        """
        self.loop_counter += 1

        current_wp = self.waypoint_manager.get_current_waypoint()
        if current_wp is None:
            return True  # 모든 미션 완료

        # 거리 계산
        target_pos = np.array([current_wp['x'], current_wp['y']], dtype=np.float32)
        distance = np.linalg.norm(agent_position - target_pos)

        # 주기적 로그
        if self.loop_counter % 100 == 0:
            self._log_progress(current_wp, distance)

        # 도달 확인
        waypoint_reached = self._check_reached(current_wp, distance)

        if waypoint_reached:
            return self._transition_to_next()

        return False

    def _check_reached(self, waypoint: Dict, distance: float) -> bool:
        """웨이포인트 도달 여부 확인"""
        distance_reached = distance < waypoint['radius']

        # CIRCLE_BUOY는 회전 완료도 체크
        if waypoint['mission_type'] == MissionType.CIRCLE_BUOY:
            rotation_completed = self.mission_manager.is_circle_mission_completed()
            if distance_reached and not rotation_completed and self.loop_counter % 100 == 0:
                self.logger.info("⚠️ 웨이포인트 도달 - 360도 회전 미완료")
            return distance_reached and rotation_completed

        return distance_reached

    def _transition_to_next(self) -> bool:
        """다음 웨이포인트로 전환"""
        self.waypoint_manager.current_waypoint_index += 1

        if self.waypoint_manager.current_waypoint_index < len(self.waypoint_manager.waypoints):
            next_wp = self.waypoint_manager.waypoints[self.waypoint_manager.current_waypoint_index]
            new_mission = next_wp['mission_type']
            self.mission_manager.set_mission(new_mission)
            self.logger.info(
                f"✅ 웨이포인트 도달! {new_mission.name} "
                f"(#{self.waypoint_manager.get_waypoint_index()}/{self.waypoint_manager.get_total_waypoints()})"
            )
            return False
        else:
            self.logger.info("🎉 모든 미션 완료!")
            return True

    def _log_progress(self, waypoint: Dict, distance: float):
        """진행 상황 로그"""
        extra = ""
        if waypoint['mission_type'] == MissionType.CIRCLE_BUOY:
            completed = self.mission_manager.is_circle_mission_completed()
            extra = f", 회전완료={completed}"

        self.logger.info(
            f"웨이포인트 {self.waypoint_manager.get_waypoint_index()}: "
            f"거리={distance:.1f}m, 목표={waypoint['radius']:.1f}m, "
            f"미션={waypoint['mission_type'].name}{extra}"
        )


class ObstacleAvoidExecutor:
    """
    장애물 회피 실행자

    장애물 회피 미션의 실행 로직을 간소화
    """

    def __init__(
        self,
        avoidance_controller,
        onnx_controller,
        waypoint_manager,
        mission_executor,
        sensor_handler,
        ros_comm,
        logger
    ):
        """
        Args:
            avoidance_controller: 장애물 회피 컨트롤러
            onnx_controller: ONNX 컨트롤러
            waypoint_manager: 웨이포인트 관리자
            mission_executor: 미션 실행자
            sensor_handler: 센서 핸들러
            ros_comm: ROS 통신 관리자
            logger: ROS2 logger
        """
        self.avoidance_controller = avoidance_controller
        self.onnx_controller = onnx_controller
        self.waypoint_manager = waypoint_manager
        self.mission_executor = mission_executor
        self.sensor_handler = sensor_handler
        self.ros_comm = ros_comm
        self.logger = logger

    def execute(
        self,
        agent_position: np.ndarray,
        agent_heading: float,
        lidar_distances: np.ndarray,
        get_lidar_distance_at_angle: Callable,
        manual_target: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        장애물 회피 미션 실행

        Args:
            agent_position: 에이전트 위치
            agent_heading: 에이전트 헤딩
            lidar_distances: LiDAR 거리 배열
            get_lidar_distance_at_angle: LiDAR 거리 조회 함수
            manual_target: 수동 목표 (강제 모드용)

        Returns:
            (left_thrust, right_thrust): 스러스터 명령
        """
        # 웨이포인트 준비
        waypoints, waypoint_index = self._prepare_waypoints(manual_target)

        # LOS target 계산 및 발행
        los_target = self.avoidance_controller.get_los_target(
            agent_position, waypoints, waypoint_index
        )
        self.ros_comm.publish_los_target(los_target[0], los_target[1])

        # 제어 명령 계산
        use_direct, linear_vel, angular_vel, check_area = \
            self.avoidance_controller.check_obstacles_and_get_control(
                agent_position, los_target, agent_heading,
                lidar_distances, get_lidar_distance_at_angle,
                self._get_onnx_control
            )

        # 디버그 정보 발행
        self._publish_debug_info(use_direct, linear_vel, angular_vel, check_area)

        # 필터 및 변환
        filtered_linear, filtered_angular = self.avoidance_controller.apply_filters(
            linear_vel, angular_vel
        )
        self.onnx_controller.update_previous_inputs(filtered_angular, filtered_linear)

        return filtered_linear, filtered_angular

    def _prepare_waypoints(
        self,
        manual_target: Optional[Tuple[float, float]]
    ) -> Tuple[list, int]:
        """웨이포인트 준비"""
        if manual_target:
            manual_wp = {
                'x': manual_target[1], 'y': manual_target[0],
                'mission_type': MissionType.OBSTACLE_AVOID,
                'radius': 15.0, 'params': {}
            }
            return [manual_wp], 0

        waypoints = [[wp['x'], wp['y']] for wp in self.waypoint_manager.waypoints]
        return waypoints, self.waypoint_manager.get_waypoint_index()

    def _get_onnx_control(self) -> Tuple[float, float]:
        """ONNX 제어 래퍼"""
        current, previous, next_wp = self.mission_executor.get_waypoint_positions()
        return self.onnx_controller.get_control(
            self.sensor_handler.lidar_distances,
            self.sensor_handler.agent_heading,
            self.sensor_handler.angular_velocity_y,
            self.sensor_handler.agent_position,
            [current[1], current[0]],
            previous,
            next_wp
        )

    def _publish_debug_info(
        self,
        use_direct: bool,
        linear_vel: float,
        angular_vel: float,
        check_area: list
    ):
        """디버그 정보 발행"""
        # 장애물 체크 영역
        if len(check_area) > 0:
            area_points = [
                (check_area[i], check_area[i + 1])
                for i in range(0, len(check_area) - 1, 2)
            ]
            self.ros_comm.publish_obstacle_check_area(area_points)

        # 제어 모드 및 출력값
        mode = "DIRECT_CONTROL" if use_direct else "ONNX_MODEL"
        self.ros_comm.publish_control_mode(mode)
        self.ros_comm.publish_control_output(linear_vel, angular_vel)


class MissionLoopExecutor:
    """
    미션 제어 루프 실행자

    메인 제어 루프의 복잡한 로직을 캡슐화하여 간소화
    """

    def __init__(
        self,
        waypoint_manager,
        mission_manager,
        mission_executor,
        detection_system,
        tracker,
        param_manager,
        sensor_handler,
        visualization,
        ros_comm,
        obstacle_avoid_executor: ObstacleAvoidExecutor,
        waypoint_transition_handler: WaypointTransitionHandler,
        logger
    ):
        """
        Args:
            waypoint_manager: 웨이포인트 관리자
            mission_manager: 미션 관리자
            mission_executor: 미션 실행자
            detection_system: 객체 탐지 시스템
            tracker: IMM-PDAF 트래커
            param_manager: 파라미터 관리자
            sensor_handler: 센서 핸들러
            visualization: 시각화 시스템
            ros_comm: ROS 통신 관리자
            obstacle_avoid_executor: 장애물 회피 실행자
            waypoint_transition_handler: 웨이포인트 전환 핸들러
            logger: ROS2 logger
        """
        self.waypoint_manager = waypoint_manager
        self.mission_manager = mission_manager
        self.mission_executor = mission_executor
        self.detection_system = detection_system
        self.tracker = tracker
        self.param_manager = param_manager
        self.sensor_handler = sensor_handler
        self.visualization = visualization
        self.ros_comm = ros_comm
        self.obstacle_avoid_executor = obstacle_avoid_executor
        self.waypoint_transition_handler = waypoint_transition_handler
        self.logger = logger

        self.loop_counter = 0
        self.param_update_interval = 30  # 30 루프마다 업데이트 (Jetson 최적화: 10 -> 30)
        self.detection_skip_frames = 2  # 2 프레임마다 탐지 (Jetson 최적화: 성능 향상)
        self.detected_objects = []
        self.raw_detections = []

    def execute_loop(self) -> None:
        """제어 루프 실행 (메인 엔트리 포인트)"""
        try:
            self.loop_counter += 1

            # 1. 현재 미션 확인
            mission_type = self._get_effective_mission()
            if mission_type is None:
                self.ros_comm.publish_thrust_commands(0.0, 0.0)
                return

            # 2. 파라미터 업데이트 (주기적)
            if self.loop_counter % self.param_update_interval == 0:
                self._update_parameters(mission_type)

            # 3. 웨이포인트 전환 확인
            mission_completed = self.waypoint_transition_handler.check_and_transition(
                self.sensor_handler.agent_position
            )
            if mission_completed:
                self.ros_comm.publish_thrust_commands(0.0, 0.0)
                return

            # 4. 객체 탐지 및 추적 (부표 미션만, Jetson 최적화: 프레임 스킵)
            if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY]:
                if self.loop_counter % self.detection_skip_frames == 0:
                    self._perform_detection_and_tracking(mission_type)

            # 5. 미션 실행
            left_thrust, right_thrust = self._execute_mission(mission_type)

            # 6. 제어 명령 발행
            self._publish_commands(left_thrust, right_thrust, mission_type)

            # 7. 시각화 (부표 미션만, Jetson 최적화: 프레임 스킵과 동기화)
            if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY]:
                if self.loop_counter % self.detection_skip_frames == 0:
                    self._visualize(mission_type)

        except Exception as e:
            self.logger.error(f"제어 루프 오류: {e}")
            self.ros_comm.publish_thrust_commands(0.0, 0.0)

    def _get_effective_mission(self) -> Optional[MissionType]:
        """유효한 미션 타입 가져오기 (강제 모드 고려)"""
        current = self.waypoint_manager.get_current_mission_type()
        if current is None:
            return None

        forced = self.param_manager.get_forced_mission_type()
        return forced if forced else current

    def _update_parameters(self, mission_type: MissionType):
        """파라미터 업데이트"""
        self.param_manager.update_all_parameters()

        thrust_scale = self.param_manager.get_thrust_scale()
        if thrust_scale:
            self.mission_manager.update_thrust_scale(thrust_scale)

        # 부표 미션만 추가 파라미터 업데이트
        if mission_type not in [MissionType.OBSTACLE_AVOID, MissionType.WAYPOINT_FOLLOW]:
            detection_params = self.param_manager.get_detection_parameters()
            self.detection_system.update_parameters(
                **{k: v for k, v in detection_params.items() if v is not None}
            )

            tracker_params = self.param_manager.get_tracker_parameters()
            if tracker_params.get('max_coast_frames'):
                self.tracker.max_coast_frames = tracker_params['max_coast_frames']
            if tracker_params.get('gate_threshold'):
                self.tracker.gate_threshold = tracker_params['gate_threshold']

    def _perform_detection_and_tracking(self, mission_type: MissionType):
        """객체 탐지 및 추적"""
        self.raw_detections = self.detection_system.detect_objects(
            self.sensor_handler.current_image, mission_type
        )

        self.tracker.predict_tracks()
        self.tracker.update_tracks(self.raw_detections)
        self.tracker.prune_tracks()

        self.detected_objects = self.tracker.get_tracked_objects()

        # 로깅 빈도 감소 (Jetson 최적화: 매번 -> 100 루프마다)
        if (len(self.raw_detections) > 0 or len(self.detected_objects) > 0) and self.loop_counter % 100 == 0:
            self.logger.info(
                f"Detection: raw={len(self.raw_detections)}, tracked={len(self.detected_objects)}"
            )

    def _execute_mission(self, mission_type: MissionType) -> Tuple[float, float]:
        """미션 실행"""
        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return self._execute_buoy_mission(mission_type, "BUOY_MISSION")

        elif mission_type == MissionType.CIRCLE_BUOY:
            return self._execute_circle_mission()

        elif mission_type == MissionType.OBSTACLE_AVOID:
            return self._execute_obstacle_avoid()

        return 0.0, 0.0

    def _execute_buoy_mission(
        self,
        mission_type: MissionType,
        mode: str
    ) -> Tuple[float, float]:
        """부표 미션 실행"""
        params = self.param_manager.get_mission_parameters(mission_type)
        left, right = self.mission_executor.execute_pass_between_buoys(
            self.detected_objects, self.sensor_handler.current_image,
            self.raw_detections, params, self.logger
        )
        self._publish_control_info(left, right, mode)
        return left, right

    def _execute_circle_mission(self) -> Tuple[float, float]:
        """부표 회전 미션 실행"""
        wp_params = self.waypoint_manager.get_current_mission_params()
        params = self.param_manager.get_mission_parameters(MissionType.CIRCLE_BUOY, wp_params)
        left, right = self.mission_executor.execute_circle_buoy(
            self.detected_objects, self.sensor_handler.current_image,
            self.sensor_handler.agent_heading, params,
            self.raw_detections, self.logger
        )
        self._publish_control_info(left, right, "CIRCLE_MISSION")
        return left, right

    def _execute_obstacle_avoid(self) -> Tuple[float, float]:
        """장애물 회피 미션 실행"""
        manual_target = None
        if self.param_manager.is_force_obstacle_avoid():
            if self.sensor_handler.manual_target_x and self.sensor_handler.manual_target_y:
                manual_target = (
                    self.sensor_handler.manual_target_x,
                    self.sensor_handler.manual_target_y
                )

        linear_vel, angular_vel = self.obstacle_avoid_executor.execute(
            self.sensor_handler.agent_position,
            self.sensor_handler.agent_heading,
            self.sensor_handler.lidar_distances,
            self.sensor_handler.get_lidar_distance_at_angle,
            manual_target
        )

        return self._convert_to_thrust(linear_vel, angular_vel)

    def _convert_to_thrust(self, linear_vel: float, angular_vel: float) -> Tuple[float, float]:
        """속도를 스러스터 명령으로 변환"""
        from .config import Constants
        thrust_scale = self.param_manager.get_thrust_scale() or Constants.DEFAULT_THRUST_SCALE

        forward = linear_vel * thrust_scale
        turn = angular_vel * thrust_scale

        left = np.clip(forward + turn, -2000, 2000)
        right = np.clip(forward - turn, -2000, 2000)

        return float(left), float(right)

    def _publish_control_info(self, left: float, right: float, mode: str):
        """제어 정보 발행 (역변환)"""
        from .config import Constants
        thrust_scale = self.param_manager.get_thrust_scale() or Constants.DEFAULT_THRUST_SCALE

        forward = (left + right) / 2.0
        turn = (left - right) / 2.0

        linear = forward / thrust_scale
        angular = turn / thrust_scale

        self.ros_comm.publish_control_output(linear, angular)
        self.ros_comm.publish_control_mode(mode)

    def _publish_commands(self, left: float, right: float, mission_type: MissionType):
        """제어 명령 발행"""
        self.ros_comm.publish_thrust_commands(left, right)
        self.ros_comm.publish_mission_status(
            mission_type.name,
            self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints()
        )

        # 부표 미션은 탐지 정보도 발행
        if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY]:
            self.ros_comm.publish_detections(self.detected_objects)

    def _visualize(self, mission_type: MissionType):
        """시각화"""
        if self.sensor_handler.current_image is None:
            return

        # 탐지 결과만 시각화 (깊이 맵 시각화 제거로 성능 향상)
        # ROS 이미지 발행 비활성화로 CPU 사용량 감소 (Jetson Nano Orin 최적화)
        self.visualization.visualize_detections(
            self.sensor_handler.current_image, self.detected_objects,
            mission_type.name, self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints(),
            raw_detections=self.raw_detections,
            bridge=None,  # ROS 이미지 발행 비활성화
            viz_image_pub=None  # ROS 이미지 발행 비활성화
        )
