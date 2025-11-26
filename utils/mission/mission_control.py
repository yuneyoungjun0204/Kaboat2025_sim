#!/usr/bin/env python3
"""
미션 제어 컴포넌트
- MissionLoopExecutor: 제어 루프 실행
- WaypointTransitionHandler: 웨이포인트 전환 관리
- ObstacleAvoidExecutor: 장애물 회피 실행
"""

from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np

from ..detection.detection_system import MissionType
from ..core.config import Constants


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
        target_pos = np.array([current_wp['y'], current_wp['x']], dtype=np.float32)
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

        # CIRCLE_BUOY는 360도 회전 완료만 체크 (거리는 무관)
        if waypoint['mission_type'] == MissionType.CIRCLE_BUOY:
            rotation_completed = self.mission_manager.is_circle_mission_completed()
            if not rotation_completed and self.loop_counter % 100 == 0:
                self.logger.info(f"⚠️ 360도 회전 진행 중 (거리={distance:.1f}m)")
            return rotation_completed

        if waypoint['mission_type'] == MissionType.ROTATION:
            rotation_completed = self.mission_manager.is_rotation_mission_completed()
            if self.loop_counter % 100 == 0:
                self.logger.info(
                    f"ROTATION 상태: completed={rotation_completed}, 거리={distance:.1f}m"
                )
            return rotation_completed

        if waypoint['mission_type'] == MissionType.DOCK_MODE:
            dock_completed = self.mission_manager.is_dock_mission_completed()
            if self.loop_counter % 100 == 0:
                self.logger.info(
                    f"DOCK 상태: completed={dock_completed}, 거리={distance:.1f}m"
                )
            return dock_completed

        if waypoint['mission_type'] == MissionType.STOP:
            # STOP 미션은 거리 체크 없이 바로 시작 (시간 기반 미션)
            # 미션이 시작되었는지 확인
            stop_mission = self.mission_manager.missions.get(MissionType.STOP)
            if stop_mission and hasattr(stop_mission, 'stop_start_time') and stop_mission.stop_start_time is not None:
                # 미션이 시작되었으면 완료 여부만 체크
                stop_completed = self.mission_manager.is_stop_mission_completed()
                if self.loop_counter % 100 == 0:
                    self.logger.info(
                        f"STOP 상태: completed={stop_completed}"
                    )
                return stop_completed
            else:
                # 미션이 아직 시작되지 않았으면 거리 체크를 건너뛰고 미션 시작
                # STOP 미션은 거리와 무관하게 바로 시작 가능
                # 미션을 강제로 시작하기 위해 미션 매니저에 현재 미션 설정
                if self.loop_counter % 10 == 0:  # 더 자주 로그 출력
                    self.logger.info(
                        f"🛑 STOP 미션 진입: 거리={distance:.1f}m (거리 체크 건너뜀, 미션 시작)"
                    )
                # 미션을 강제로 시작하기 위해 미션 매니저에 현재 미션 설정
                self.mission_manager.set_mission(MissionType.STOP)
                # 미션이 시작되도록 하기 위해 execute_body_forces를 한 번 호출
                # 이렇게 하면 stop_start_time이 설정됨
                try:
                    params = self.waypoint_manager.get_current_mission_params()
                    stop_mission.execute_body_forces(params, logger=self.logger)
                except Exception as e:
                    self.logger.warn(f"STOP 미션 초기화 중 오류: {e}")
                # False 반환하여 아직 완료되지 않았음을 표시 (다음 루프에서 미션이 실행됨)
                return False

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

        # avoid_yaw 토글 상태 (ONNX 모드가 아닐 때 0, 1 번갈아 발행)
        self.avoid_yaw_toggle = 0

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
        if self.onnx_controller is not None:
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

        # 미션 시작 위치 기준으로 웨이포인트 좌표를 m 단위로 변환
        waypoints = []
        for i in range(len(self.waypoint_manager.waypoints)):
            x, y = self.waypoint_manager.get_waypoint_relative_to_initial(i)
            waypoints.append([x, y])  # [y, x] 형태로 저장 (기존 호환성)

        return waypoints, self.waypoint_manager.get_waypoint_index()

    def _get_onnx_control(self) -> Tuple[float, float]:
        """ONNX 제어 래퍼 (v2 API - previous/next waypoint 제거)"""
        # 지연 로딩된 경우 기본값 반환
        if self.onnx_controller is None:
            return 0.0, 0.0
        
        current, previous, next_wp = self.mission_executor.get_waypoint_positions()
        return self.onnx_controller.get_control(
            self.sensor_handler.lidar_distances,
            self.sensor_handler.agent_heading,
            self.sensor_handler.angular_velocity_y,
            self.sensor_handler.agent_position,
            [current[1], current[0]],
            [previous[1], previous[0]],
            [next_wp[1], next_wp[0]]
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

        # avoid_yaw 토픽 발행 (ONNX 모드가 아닐 때 0, 1 번갈아 발행)
        if use_direct:
            self.ros_comm.publish_avoid_yaw(self.avoid_yaw_toggle)
            self.avoid_yaw_toggle = 1 - self.avoid_yaw_toggle  # 0 <-> 1 토글


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
        self.param_update_interval = Constants.ThrusterControl.PARAM_UPDATE_INTERVAL  # Jetson 최적화
        self.ros_publish_skip_frames = Constants.ThrusterControl.ROS_PUBLISH_SKIP_FRAMES  # Jetson 최적화
        self.detected_objects = []
        self.raw_detections = []

        # DOCK_MODE용 추가 상태 변수
        self.thruster_positions = (None, None)  # (left_pos, right_pos)
        self.target_depth = None  # 목표 객체 depth

        # PX4 명령 변환기
        if Constants.PX4.ENABLED:
            from ..communication.px4_adapter import PX4CommandConverter
            self.px4_converter = PX4CommandConverter()
        else:
            self.px4_converter = None

        # Jetson 최적화: 초기 파라미터 한번만 설정
        self._initialize_parameters_once()

    def _initialize_parameters_once(self):
        """파라미터를 한 번만 초기화 (Jetson 최적화)"""
        # 탐지 시스템 파라미터 (지연 로딩된 경우 스킵)
        if self.detection_system is not None:
            detection_params = self.param_manager.get_detection_parameters()
            if detection_params:
                self.detection_system.update_parameters(
                    **{k: v for k, v in detection_params.items() if v is not None}
                )

        # 트래커 파라미터
        tracker_params = self.param_manager.get_tracker_parameters()
        if tracker_params:
            if tracker_params.get('max_coast_frames'):
                self.tracker.max_coast_frames = tracker_params['max_coast_frames']
            if tracker_params.get('gate_threshold'):
                self.tracker.gate_threshold = tracker_params['gate_threshold']

        # 스러스트 스케일
        thrust_scale = self.param_manager.get_thrust_scale()
        if thrust_scale:
            self.mission_manager.update_thrust_scale(thrust_scale)

    def execute_loop(self) -> None:
        """제어 루프 실행 (메인 엔트리 포인트)"""
        try:
            self.loop_counter += 1

            # 1. 현재 미션 확인
            mission_type = self._get_effective_mission()
            if mission_type is None:
                # 미션이 없을 때: PX4에 0 명령 발행
                self.ros_comm.publish_thrust_commands(0.0, 0.0)
                current_yaw_rad = np.radians(self.sensor_handler.agent_heading) if self.sensor_handler.agent_heading is not None else 0.0
                self.ros_comm.publish_desired_control(
                    0.0, 0.0, 0.0,
                    current_yaw=current_yaw_rad, is_dock_mode=False
                )
                return

            # 2. 파라미터 업데이트 (거의 안함 - Jetson 최적화)
            # if self.loop_counter % self.param_update_interval == 0:
            #     self._update_parameters(mission_type)

            # 3. 웨이포인트 전환 확인
            mission_completed = self.waypoint_transition_handler.check_and_transition(
                self.sensor_handler.agent_position
            )
            if mission_completed:
                # 모든 미션 완료: PX4에 0 명령 발행
                self.ros_comm.publish_thrust_commands(0.0, 0.0)
                # PX4 브릿지에도 0 명령 발행 (속도 제어 모드로 전환)
                current_yaw_rad = np.radians(self.sensor_handler.agent_heading) if self.sensor_handler.agent_heading is not None else 0.0
                self.ros_comm.publish_desired_control(
                    0.0, 0.0, 0.0,
                    current_yaw=current_yaw_rad, is_dock_mode=False
                )
                return

            # 4. 객체 탐지 및 추적 (부표 미션 및 도킹 미션)
            if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY, MissionType.DOCK_MODE]:
                self._perform_detection_and_tracking(mission_type)

            # 5. 미션 실행
            left_thrust, right_thrust = self._execute_mission(mission_type)

            # 6. 제어 명령 발행
            self._publish_commands(left_thrust, right_thrust, mission_type)

            # 7. 시각화 (모든 미션에서 활성화)
            self._visualize(mission_type)

        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            try:
                self.logger.error(f"제어 루프 오류: {error_msg}")
                self.logger.error(f"스택 트레이스:\n{error_traceback}")
            except:
                # logger 자체에 문제가 있는 경우
                print(f"제어 루프 오류: {error_msg}")
                print(f"스택 트레이스:\n{error_traceback}")
            self.ros_comm.publish_thrust_commands(0.0, 0.0)

    def _get_effective_mission(self) -> Optional[MissionType]:
        """유효한 미션 타입 가져오기"""
        return self.waypoint_manager.get_current_mission_type()

    def _update_parameters(self, mission_type: MissionType):
        """파라미터 업데이트"""
        self.param_manager.update_all_parameters()

        thrust_scale = self.param_manager.get_thrust_scale()
        if thrust_scale:
            self.mission_manager.update_thrust_scale(thrust_scale)

        # 부표 미션 및 도킹 미션만 추가 파라미터 업데이트 (지연 로딩된 경우 스킵)
        if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY, MissionType.DOCK_MODE]:
            if self.detection_system is not None:
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
        # 지연 로딩된 경우 스킵
        if self.detection_system is None:
            self.raw_detections = []
            return
        
        self.raw_detections = self.detection_system.detect_objects(
            self.sensor_handler.current_image, mission_type
        )

        self.tracker.predict_tracks()
        self.tracker.update_tracks(self.raw_detections)  # <--- 수정된 부분
        self.tracker.prune_tracks()

        self.detected_objects = self.tracker.get_tracked_objects()

        # Depth 정보 ROS 발행 (원본 탐지 결과)
        if len(self.raw_detections) > 0:
            self.ros_comm.publish_detection_depths(self.raw_detections)

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

        elif mission_type == MissionType.DOCK_MODE:
            return self._execute_dock_mission()

        elif mission_type == MissionType.ROTATION:
            return self._execute_rotation_mission()

        elif mission_type == MissionType.STOP:
            return self._execute_stop_mission()

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
            self.raw_detections, params, self.logger,
            self.sensor_handler.agent_position,
            self.sensor_handler.agent_heading
        )
        
        # LOS guidance 사용 여부 확인 및 제어 모드/LOS target 발행
        pass_mission = self.mission_manager.missions.get(MissionType.PASS_BETWEEN_BUOYS)
        if pass_mission and hasattr(pass_mission, 'using_los_guidance') and pass_mission.using_los_guidance:
            # LOS guidance 사용 중
            control_mode = "PASS_BETWEEN_BUOYS_LOS"
            if hasattr(pass_mission, 'los_target_position') and pass_mission.los_target_position is not None:
                # LOS target 발행 (NED 좌표계: [North, East])
                los_target = pass_mission.los_target_position
                self.ros_comm.publish_los_target(los_target[0], los_target[1])
        else:
            # 부표 탐지 중
            control_mode = mode
        
        self._publish_control_info(left, right, control_mode)
        return left, right

    def _execute_circle_mission(self) -> Tuple[float, float]:
        """
        웨이포인트 기반 원 궤적 그리기 미션 실행

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)

        Note:
            thruster positions는 self.thruster_positions에 저장됨
        """
        wp_params = self.waypoint_manager.get_current_mission_params()
        params = self.param_manager.get_mission_parameters(MissionType.CIRCLE_BUOY, wp_params)
        result = self.mission_executor.execute_circle_buoy(
            self.sensor_handler.agent_position,
            self.sensor_handler.agent_heading,
            params,
            self.logger
        )

        # 4개 값 언패킹: (left_thrust, right_thrust, left_pos, right_pos)
        left_thrust, right_thrust, left_pos, right_pos = result

        # thruster positions 저장
        self.thruster_positions = (left_pos, right_pos)

        # CIRCLE_BUOY 목표 포인트 publish (시각화용)
        circle_waypoints = self.mission_manager.get_circle_buoy_waypoints()
        if circle_waypoints:
            self.ros_comm.publish_circle_buoy_waypoints(circle_waypoints)
            self.logger.info(
                f"📤 CIRCLE_BUOY 웨이포인트 발행: {len(circle_waypoints)}개, "
                f"포인트={circle_waypoints[0]}"
            )
        else:
            self.logger.warn("⚠️ CIRCLE_BUOY 웨이포인트가 비어있음")
        
        # CIRCLE_BUOY 현재 목표 포인트 publish (시각화용)
        current_wp = self.mission_manager.get_circle_buoy_current_waypoint()
        if current_wp is not None:
            self.ros_comm.publish_circle_buoy_current_waypoint(current_wp)
            self.logger.info(f"📤 CIRCLE_BUOY 현재 웨이포인트 발행: {current_wp}")

        self._publish_control_info(left_thrust, right_thrust, "CIRCLE_MISSION")
        return left_thrust, right_thrust

    def _execute_rotation_mission(self) -> Tuple[float, float]:
        """제자리 선회 미션 실행"""
        wp_params = self.waypoint_manager.get_current_mission_params()
        params = self.param_manager.get_mission_parameters(MissionType.ROTATION, wp_params)
        left, right = self.mission_executor.execute_rotation(
            self.sensor_handler.agent_heading,
            params,
            self.logger
        )
        self._publish_control_info(left, right, "ROTATION_MISSION")
        return left, right

    def _execute_obstacle_avoid(self) -> Tuple[float, float]:
        """장애물 회피 미션 실행"""
        # mission_params에서 los_delta 가져오기
        wp_params = self.waypoint_manager.get_current_mission_params()
        params = self.param_manager.get_mission_parameters(MissionType.OBSTACLE_AVOID, wp_params)
        los_delta = params.get('los_delta', Constants.AvoidControl.OBSTACLE_AVOID_LOS_DELTA)
        
        # LOS delta 업데이트
        self.obstacle_avoid_executor.avoidance_controller.update_los_delta(los_delta)
        
        linear_vel, angular_vel = self.obstacle_avoid_executor.execute(
            self.sensor_handler.agent_position,
            self.sensor_handler.agent_heading,
            self.sensor_handler.lidar_distances,
            self.sensor_handler.get_lidar_distance_at_angle
        )

        return self._convert_to_thrust(linear_vel, angular_vel)

    def _execute_dock_mission(self) -> Tuple[float, float]:
        """
        도킹 미션 실행 (좌표 기반 LOS 가이던스)

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)

        Note:
            thruster positions와 target_depth는 self.thruster_positions와
            self.target_depth에 저장됨
        """
        wp_params = self.waypoint_manager.get_current_mission_params()
        params = self.param_manager.get_mission_parameters(MissionType.DOCK_MODE, wp_params)
        
        result = self.mission_executor.execute_dock_mission(
            detected_objects=self.detected_objects,
            current_image=self.sensor_handler.current_image,
            agent_heading=self.sensor_handler.agent_heading,
            agent_position=self.sensor_handler.agent_position,
            raw_detections=self.raw_detections,
            mission_params=params,
            logger=self.logger
        )

        # 5개 값 언패킹: (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        left_thrust, right_thrust, left_pos, right_pos, target_depth = result

        # thruster positions 및 target_depth 저장
        self.thruster_positions = (left_pos, right_pos)
        self.target_depth = target_depth

        # Position Control 모드일 때 오차 정보 발행 (시각화용 및 PX4 직접 발행)
        # REVERSING 단계에서는 position 제어를 사용하지 않음
        dock_mission = self.mission_manager.missions.get(MissionType.DOCK_MODE)
        is_reversing = dock_mission and hasattr(dock_mission, 'docking_phase') and dock_mission.docking_phase == "REVERSING"
        
        if dock_mission and hasattr(dock_mission, 'control_mode') and dock_mission.control_mode == 'POSITION_CONTROL' and not is_reversing:
            if hasattr(dock_mission, 'last_x_error') and hasattr(dock_mission, 'last_y_error'):
                desired_psi = dock_mission.desired_psi if hasattr(dock_mission, 'desired_psi') else None
                # 시각화용 발행
                self.ros_comm.publish_dock_position_error(
                    dock_mission.last_x_error,
                    dock_mission.last_y_error,
                    desired_psi
                )
                
                # PX4 직접 발행: body-frame → world-frame 변환
                x_error_body = dock_mission.last_x_error
                y_error_body = dock_mission.last_y_error
                agent_heading = self.sensor_handler.agent_heading
                if agent_heading is not None:
                    heading_rad = np.radians(agent_heading)
                    cos_h = np.cos(heading_rad)
                    sin_h = np.sin(heading_rad)
                    
                    # Body-frame → World-frame 변환 (NED)
                    # error_north = x_error_body * cos_h - y_error_body * sin_h
                    # error_east = x_error_body * sin_h + y_error_body * cos_h
                    
                    # PX4 position_error 토픽으로 직접 발행 (desired_psi 포함)
                    self.ros_comm.publish_px4_position_command(x_error_body, y_error_body, desired_psi)
            

        # # REVERSING 단계에서 velocity 제어 명령 발행 (각도 고정)
        # if is_reversing:
        #     # reverse_speed를 velocity로 사용
        #     reverse_speed = dock_mission.reverse_speed if hasattr(dock_mission, 'reverse_speed') else -0.3
        #     # 각도 고정: desired_moment = 0.0, current_yaw 사용
        #     current_yaw_rad = np.radians(self.sensor_handler.agent_heading) if self.sensor_handler.agent_heading is not None else 0.0
        #     # velocity 제어 모드로 발행 (is_dock_mode=False)
        #     self.ros_comm.publish_desired_control(
        #         -0.5,  # desired_speed
        #         0.0,            # desired_moment (각도 고정)
        #         0.0,            # desired_force_y
        #         current_yaw=current_yaw_rad,
        #         is_dock_mode=False  # velocity 제어 모드
        #     )

        self._publish_control_info(left_thrust, right_thrust, "DOCK_MISSION")
        return left_thrust, right_thrust

    def _execute_stop_mission(self) -> Tuple[float, float]:
        """정지 미션 실행"""
        wp_params = self.waypoint_manager.get_current_mission_params()
        params = self.param_manager.get_mission_parameters(MissionType.STOP, wp_params)
        agent_heading = self.sensor_handler.agent_heading if self.sensor_handler.agent_heading is not None else 0.0
        left, right = self.mission_executor.execute_stop(
            params,
            agent_heading,
            self.logger
        )
        self._publish_control_info(left, right, "STOP_MISSION")
        return left, right

    def _convert_to_thrust(self, linear_vel: float, angular_vel: float) -> Tuple[float, float]:
        """속도를 스러스터 명령으로 변환"""
        from ..core.config import Constants
        thrust_scale = self.param_manager.get_thrust_scale() or Constants.DEFAULT_THRUST_SCALE

        forward = linear_vel * thrust_scale
        turn = angular_vel * thrust_scale

        tc = Constants.ThrusterControl
        left = np.clip(forward + turn, tc.THRUST_MIN, tc.THRUST_MAX)
        right = np.clip(forward - turn, tc.THRUST_MIN, tc.THRUST_MAX)

        return float(left), float(right)

    def _publish_control_info(self, left: float, right: float, mode: str):
        """제어 정보 발행 (Jetson 최적화: ROS 발행 빈도 감소)"""
        # Jetson 최적화: 디버그 정보는 간헐적으로만 발행
        if self.loop_counter % self.ros_publish_skip_frames == 0:
            from ..core.config import Constants
            thrust_scale = self.param_manager.get_thrust_scale() or Constants.DEFAULT_THRUST_SCALE

            forward = (left + right) / 2.0
            turn = (left - right) / 2.0

            linear = forward / thrust_scale
            angular = turn / thrust_scale

            self.ros_comm.publish_control_output(linear, angular)
            self.ros_comm.publish_control_mode(mode)

    def _publish_commands(self, left: float, right: float, mission_type: MissionType):
        """제어 명령 발행 (Jetson 최적화: ROS 발행 빈도 감소)"""
        from ..core.config import Constants

        # STOP 미션은 명시적으로 0으로 설정
        if mission_type == MissionType.STOP:
            left = 0.0
            right = 0.0
            desired_speed = 0.0
            desired_moment = 0.0
            desired_force_y = 0.0
        else:
            thrust_scale = self.param_manager.get_thrust_scale() or Constants.DEFAULT_THRUST_SCALE
            thrust_scale = max(thrust_scale, 1e-3)
            forward = (left + right) / 2.0
            turn = (left - right) / 2.0
            desired_speed = float(np.clip(forward / thrust_scale, -1.0, 1.0))
            desired_moment = float(np.clip(turn / thrust_scale, -1.0, 1.0))
            desired_force_y = 0.0

        if mission_type == MissionType.DOCK_MODE:
            sway_force, yaw_moment, surge_velocity = self.mission_manager.get_dock_body_forces()
            if surge_velocity is not None:
                desired_speed = float(np.clip(surge_velocity, -1.0, 1.0))
            if yaw_moment is not None:
                desired_moment = float(np.clip(yaw_moment, -1.0, 1.0))
            if sway_force is not None:
                desired_force_y = float(np.clip(sway_force, -1.0, 1.0))

        # STOP 미션은 명시적으로 모든 제어 명령을 0으로 설정
        if mission_type == MissionType.STOP:
            desired_speed = 0.0
            desired_moment = 0.0
            desired_force_y = 0.0
        
        # 현재 yaw를 라디안으로 변환
        current_yaw_rad = np.radians(self.sensor_handler.agent_heading) if self.sensor_handler.agent_heading is not None else 0.0
        
        # STOP 미션일 때: 속도와 각도를 0으로 명령
        # if mission_type == MissionType.STOP:
        #     # velocity와 yaw를 모두 0으로 설정하여 PX4에 직접 발행
        #     self.ros_comm.publish_px4_velocity_command(0.0, 0.0)
        #     return
        
        # 도킹 미션 단계 확인: REVERSING 단계에서는 velocity 제어 모드 사용
        is_dock_mode = (mission_type == MissionType.DOCK_MODE)
        is_dock_reversing = False
        if is_dock_mode:
            dock_mission = self.mission_manager.missions.get(MissionType.DOCK_MODE)
            if dock_mission and hasattr(dock_mission, 'docking_phase'):
                is_dock_reversing = (dock_mission.docking_phase == "REVERSING")
        
            # REVERSING 단계에서는 velocity 제어 모드 사용 (position 제어 비활성화)
            is_dock_position_control = is_dock_mode and not is_dock_reversing
            # REVERSING 단계에서는 명시적으로 velocity 명령 발행 (현재 헤딩 유지)
            if is_dock_reversing:
                # velocity 제어 모드로 전환
                from ..core.config import Constants
            # STOP 미션일 때: 속도와 각도를 0으로 명령
                velocity = float(desired_speed) * Constants.PX4.VELOCITY_SCALE
                velocity = np.clip(velocity, Constants.PX4.MIN_VELOCITY, Constants.PX4.MAX_VELOCITY)
                # 현재 헤딩 유지 (yaw_moment가 0이므로 target_yaw = current_yaw)
                self.ros_comm.publish_px4_velocity_command(-1.0, current_yaw_rad)
        elif mission_type == MissionType.STOP:
            # STOP 미션: velocity는 0, yaw는 desired_psi 사용 (없으면 현재 각도 유지)
            stop_mission = self.mission_manager.missions.get(MissionType.STOP)
            if stop_mission and hasattr(stop_mission, 'desired_psi') and stop_mission.desired_psi is not None:
                # desired_psi를 라디안으로 변환
                desired_psi_rad = np.radians(stop_mission.desired_psi)
                self.ros_comm.publish_px4_velocity_command(0.0, current_yaw_rad)
            else:
                # desired_psi가 없으면 현재 각도 유지
                self.ros_comm.publish_px4_velocity_command(0.0, current_yaw_rad)
        else:
            self.ros_comm.publish_desired_control(
                desired_speed, desired_moment, desired_force_y,
                current_yaw=current_yaw_rad, is_dock_mode=is_dock_position_control
            )

        # # REVERSING 단계에서는 명시적으로 velocity 명령 발행 (현재 헤딩 유지)
        # if is_dock_reversing:
        #     # velocity 제어 모드로 전환
        #     from ..core.config import Constants
        #     velocity = float(desired_speed) * Constants.PX4.VELOCITY_SCALE
        #     velocity = np.clip(velocity, Constants.PX4.MIN_VELOCITY, Constants.PX4.MAX_VELOCITY)
        #     # 현재 헤딩 유지 (yaw_moment가 0이므로 target_yaw = current_yaw)
        #     self.ros_comm.publish_px4_velocity_command(-0.5, current_yaw_rad)

        # PX4 브릿지 명령은 publish_desired_control 내에서 자동으로 발행됨

        # 스러스트 명령은 매번 발행 (중요)
        self.ros_comm.publish_thrust_commands(left, right)

        # DOCK_MODE: thruster position 발행
        if mission_type == MissionType.DOCK_MODE:
            left_pos, right_pos = self.thruster_positions
            if left_pos is not None and right_pos is not None:
                self.ros_comm.publish_thruster_positions(left_pos, right_pos)

            # target_depth 발행
            if self.target_depth is not None:
                self.ros_comm.publish_target_depth(self.target_depth)

        # 상태 메시지는 간헐적으로 발행 (Jetson 최적화)
        if self.loop_counter % self.ros_publish_skip_frames == 0:
            self.ros_comm.publish_mission_status(
                mission_type.name,
                self.waypoint_manager.get_waypoint_index(),
                self.waypoint_manager.get_total_waypoints()
            )

            # 부표 미션 및 도킹 미션은 탐지 정보도 발행
            if mission_type in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY, MissionType.DOCK_MODE]:
                self.ros_comm.publish_detections(self.detected_objects)
            
            # 모든 웨이포인트 정보 발행 (시각화용)
            self.ros_comm.publish_all_waypoints(
                self.waypoint_manager.waypoints,
                self.waypoint_manager.get_waypoint_index()
            )

    def _visualize(self, mission_type: MissionType):
        """시각화"""
        # OBSTACLE_AVOID 미션일 때는 이미지 창에 plot하지 않음
        if mission_type == MissionType.OBSTACLE_AVOID:
            return
        
        # 이미지가 없으면 대기 메시지 표시
        if self.sensor_handler.current_image is None:
            # 검은 화면에 대기 메시지 표시
            import cv2
            import numpy as np
            waiting_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_image, "Waiting for camera image...", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(waiting_image, f"Mission: {mission_type.name}", (50, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(Constants.VisualizationParams.WINDOW_NAME, waiting_image)
            cv2.waitKey(1)
            return

        # 탐지 결과만 시각화 (깊이 맵 시각화 제거로 성능 향상)
        # ROS 이미지 발행 비활성화로 CPU 사용량 감소 (Jetson Nano Orin 최적화)
        # 도킹 미션인 경우 누적 각도 전달
        accumulated_angle = None
        if mission_type.name == "DOCK_MODE":
            accumulated_angle = self.mission_manager.get_dock_accumulated_angle()

        # Depth map 가져오기 (있는 경우)
        depth_map = getattr(self.detection_system, 'last_depth_map', None)

        self.visualization.visualize_detections(
            self.sensor_handler.current_image, self.detected_objects,
            mission_type.name, self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints(),
            raw_detections=self.raw_detections,
            bridge=None,  # ROS 이미지 발행 비활성화
            viz_image_pub=None,  # ROS 이미지 발행 비활성화
            accumulated_angle=accumulated_angle,
            depth_map=depth_map  # Depth map 전달
        )