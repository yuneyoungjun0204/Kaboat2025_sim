#!/usr/bin/env python3
"""
미션 전략 모듈
- 4가지 미션의 제어 로직을 캡슐화
- 리팩토링: 중복 제거, 매직 넘버 상수화, 타입 힌트 보완
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Callable

from .detection_system import MissionType
from .config import Constants


# ============================================================================
# 헬퍼 함수
# ============================================================================

def find_buoy_with_fallback(
    label: str,
    detected_objects: List[Dict],
    raw_detections: Optional[List[Dict]] = None,
    logger=None
) -> Tuple[Optional[Dict], str]:
    """
    부표 탐지: 추적값 우선, 없으면 원본 측정값 사용

    Args:
        label: 찾을 부표 라벨 ('red_cone', 'green_cone', 'blue_buoy' 등)
        detected_objects: 추적된 객체 리스트 (IMM-PDAF 출력)
        raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력)
        logger: 로거

    Returns:
        Tuple[Optional[Dict], str]: (부표 딕셔너리, 데이터 소스)
            데이터 소스는 'TRACKED' 또는 'RAW'
    """
    # 1. 추적값에서 먼저 찾기
    if detected_objects:
        for det in detected_objects:
            if det['label'] == label:
                return det, "TRACKED"

    # 2. 원본 측정값에서 찾기
    if raw_detections:
        for det in raw_detections:
            if det['label'] == label:
                if logger:
                    logger.info(f"⚠️ {label}: 추정값 없음 → 측정값 사용")
                return det, "RAW"

    # 3. 둘 다 없으면 None
    return None, "NONE"


# ============================================================================
# PID 제어기
# ============================================================================

class PIDController:
    """PID 제어기"""

    def __init__(self, kp: float = 0.5, ki: float = 0.011, kd: float = 0.4):
        """
        Args:
            kp: 비례 게인
            ki: 적분 게인
            kd: 미분 게인
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error: float) -> float:
        """
        PID 제어 업데이트

        Args:
            error: 오차 값

        Returns:
            float: PID 출력
        """
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

    def reset(self):
        """PID 제어기 상태 초기화"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


# ============================================================================
# 미션 전략 클래스
# ============================================================================

class BaseMissionStrategy:
    """미션 전략 베이스 클래스"""

    def __init__(self, thrust_scale: float = 1000.0):
        self.thrust_scale = thrust_scale

    def execute(self, **kwargs) -> Tuple[float, float]:
        """
        미션 실행

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        raise NotImplementedError

    def reset(self):
        """미션 상태 초기화"""
        pass


class PassBetweenBuoysMission(BaseMissionStrategy):
    """미션 1: 부표 사이 지나가기"""

    def __init__(self, thrust_scale: float = 1000.0):
        super().__init__(thrust_scale)
        # PID 제어기 (사용하지 않지만 하위 호환성 유지)
        self.pid_controller = PIDController(
            kp=Constants.PASS_BETWEEN_PID_KP,
            ki=Constants.PASS_BETWEEN_PID_KI,
            kd=Constants.PASS_BETWEEN_PID_KD
        )

    def reset(self):
        """미션 상태 초기화"""
        self.pid_controller.reset()

    def execute(
        self,
        detected_objects: List[Dict],
        current_image: np.ndarray,
        logger=None,
        raw_detections: Optional[List[Dict]] = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        빨간색/초록색 고깔 부표 사이로 지나가기

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            current_image: 현재 카메라 이미지
            logger: 로거
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        # 빨간색/초록색 부표 찾기 (헬퍼 함수 사용)
        red_buoy, red_source = find_buoy_with_fallback(
            'red_cone', detected_objects, raw_detections, logger
        )
        green_buoy, green_source = find_buoy_with_fallback(
            'green_cone', detected_objects, raw_detections, logger
        )

        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 깊이 차이 필터링 (너무 차이 나면 멀리 있는 것 무시)
            max_depth_diff = kwargs.get('mission_params', {}).get(
                'pass_max_depth_diff', Constants.PASS_BETWEEN_MAX_DEPTH_DIFF
            )
            red_depth = red_buoy.get('depth', 0.0)
            green_depth = green_buoy.get('depth', 0.0)
            depth_diff = abs(red_depth - green_depth)

            if depth_diff > max_depth_diff:
                # 깊이 차이가 너무 크면 멀리 있는 부표 무시
                if red_depth > green_depth:
                    if logger:
                        logger.warn(
                            f"빨간 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)"
                        )
                    red_buoy['filtered_by_depth_diff'] = True
                    red_buoy = None
                else:
                    if logger:
                        logger.warn(
                            f"초록 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)"
                        )
                    green_buoy['filtered_by_depth_diff'] = True
                    green_buoy = None

        # 필터링 후 두 부표가 모두 있는 경우
        if red_buoy and green_buoy:
            # 두 부표의 중점 계산 (이미지 좌표)
            red_x = red_buoy['center'][0]
            green_x = green_buoy['center'][0]
            midpoint_x = (red_x + green_x) / 2

            # 이미지 중심
            image_center_x = current_image.shape[1] / 2

            # 오차 계산
            error = midpoint_x - image_center_x

            # 비례 제어
            steering = error * Constants.PASS_BETWEEN_STEERING_GAIN
            steering = np.clip(steering, -Constants.PASS_BETWEEN_MAX_STEERING,
                             Constants.PASS_BETWEEN_MAX_STEERING)

            # 스러스터 명령 계산
            left_thrust = (Constants.PASS_BETWEEN_FORWARD_SPEED + steering) * self.thrust_scale
            right_thrust = (Constants.PASS_BETWEEN_FORWARD_SPEED - steering) * self.thrust_scale

            if logger:
                data_source = f"R:{red_source}/G:{green_source}"
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, "
                    f"error={error:.1f}, steering={steering:.3f}"
                )
        else:
            # 부표 미탐지 시 천천히 전진
            left_thrust = Constants.PASS_BETWEEN_FALLBACK_SPEED * self.thrust_scale
            right_thrust = Constants.PASS_BETWEEN_FALLBACK_SPEED * self.thrust_scale
            if logger:
                logger.warn("부표 미탐지: 천천히 전진")

        return left_thrust, right_thrust


class CircleBuoyMission(BaseMissionStrategy):
    """미션 2: 부표 주변 회전"""

    def __init__(self, thrust_scale: float = 1000.0):
        super().__init__(thrust_scale)
        self.circle_start_time: Optional[float] = None
        self.circle_initial_heading: Optional[float] = None
        self.total_rotation = 0.0
        self.previous_heading: Optional[float] = None
        self.circling_started = False
        self.is_completed = False  # 360도 회전 완료 플래그

        # PID 제어기 초기화
        self.pid_controller = PIDController(
            kp=Constants.CIRCLE_PID_KP,
            ki=Constants.CIRCLE_PID_KI,
            kd=Constants.CIRCLE_PID_KD
        )

        # 제어 파라미터
        self.image_width = Constants.CIRCLE_IMAGE_WIDTH
        self.image_height = Constants.CIRCLE_IMAGE_HEIGHT
        self.target_center_x = self.image_width / 2

        # 속도 제어 파라미터
        self.base_speed = Constants.CIRCLE_BASE_SPEED
        self.min_speed = Constants.CIRCLE_MIN_SPEED
        self.max_turn_thrust = Constants.CIRCLE_MAX_TURN_THRUST

        # target_x 결정식 파라미터 (시계방향)
        self.tx_base_x = Constants.CIRCLE_TX_BASE_X
        self.tx_slope = Constants.CIRCLE_TX_SLOPE
        self.tx_min_x = Constants.CIRCLE_TX_MIN_X
        self.tx_max_x = Constants.CIRCLE_TX_MAX_X

        # 목표 X 위치 저장 (시각화용)
        self.target_x: Optional[float] = None

        # 마지막으로 성공한 스러스터 명령 저장 (탐지 실패 시 사용)
        self.last_known_left_cmd = 0.0
        self.last_known_right_cmd = 0.0

    def reset(self):
        """미션 상태 초기화"""
        self.circle_start_time = None
        self.circle_initial_heading = None
        self.total_rotation = 0.0
        self.previous_heading = None
        self.circling_started = False
        self.is_completed = False
        self.pid_controller.reset()
        self.target_x = None
        self.last_known_left_cmd = 0.0
        self.last_known_right_cmd = 0.0

    def calculate_rotation_target(self, rotation_direction: int, object_depth: float) -> float:
        """
        회전 방향에 따른 목표 x 좌표 계산

        Args:
            rotation_direction: 1=시계방향, 2 or -1=반시계방향
            object_depth: 부표까지의 깊이 (미터)

        Returns:
            float: 목표 x 좌표 (픽셀)
        """
        if rotation_direction == 1:  # 시계방향
            target_x = self.tx_base_x - self.tx_slope * object_depth
            return max(self.tx_min_x, min(self.tx_max_x, target_x))
        else:  # 반시계방향
            target_x = Constants.CIRCLE_CCW_SLOPE * object_depth
            return max(Constants.CIRCLE_CCW_MIN_X, min(Constants.CIRCLE_CCW_MAX_X, target_x))

    def calculate_steering_command(self, error: float) -> float:
        """
        조향 명령 계산 (PID 제어)

        Args:
            error: 위치 오차 (픽셀)

        Returns:
            float: 조향 명령 (-1.0 ~ 1.0)
        """
        # 오차 정규화 (이미지 너비의 절반으로 나누어 -1~1 범위로)
        normalized_error = error / (self.image_width / 2)

        # PID 제어기로 조향 명령 계산
        steering_command = self.pid_controller.update(normalized_error)

        # 조향 명령 제한
        return max(-1.0, min(1.0, steering_command))

    def calculate_rotation_speed(self, turn_angle: float) -> float:
        """
        각도에 따른 적응형 속도 계산 (각도가 클수록 속도 감소)

        Args:
            turn_angle: 회전 각도 (도)

        Returns:
            float: 전진 속도
        """
        abs_angle = abs(turn_angle)

        # 각도가 클수록 속도 감소 (선형적)
        # 0도: 기본 속도, 90도: 최소 속도
        if abs_angle >= 90:
            return self.min_speed
        else:
            speed_ratio = 1.0 - (abs_angle / 90.0)
            adaptive_speed = self.min_speed + (self.base_speed - self.min_speed) * speed_ratio
            return max(self.min_speed, adaptive_speed)

    def execute(
        self,
        detected_objects: List[Dict],
        current_image: np.ndarray,
        agent_heading: float,
        mission_params: Dict,
        logger=None,
        raw_detections: Optional[List[Dict]] = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        파란색 부표 주변을 회전

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            current_image: 현재 카메라 이미지
            agent_heading: 로봇 헤딩 (도)
            mission_params: 미션 파라미터
            logger: 로거
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        # 트랙바 파라미터 업데이트 (동적 조정)
        self.base_speed = mission_params.get('circle_base_speed', self.base_speed)
        self.min_speed = mission_params.get('circle_min_speed', self.min_speed)
        self.max_turn_thrust = mission_params.get('circle_max_turn', self.max_turn_thrust)

        circle_pid_kp = mission_params.get('circle_pid_kp', self.pid_controller.kp)
        if circle_pid_kp != self.pid_controller.kp:
            self.pid_controller.kp = circle_pid_kp

        # target_x 결정식 파라미터 업데이트
        self.tx_base_x = mission_params.get('circle_tx_base_x', self.tx_base_x)
        self.tx_slope = mission_params.get('circle_tx_slope', self.tx_slope)
        tx_min_x = mission_params.get('circle_tx_min_x', self.tx_min_x)
        tx_max_x = mission_params.get('circle_tx_max_x', self.tx_max_x)
        # min/max 순서 보장
        self.tx_min_x = min(tx_min_x, tx_max_x - 1.0)
        self.tx_max_x = max(tx_max_x, self.tx_min_x + 1.0)

        # 파란색 부표 찾기 (헬퍼 함수 사용)
        blue_buoy, data_source = find_buoy_with_fallback(
            'blue_buoy', detected_objects, raw_detections, logger
        )

        # 회전 시작 시간 기록 (완료 확인용)
        if self.circle_start_time is None:
            self.circle_start_time = time.time()
            self.circle_initial_heading = agent_heading if agent_heading else 0.0
            self.previous_heading = agent_heading if agent_heading else 0.0
            self.total_rotation = 0.0

        # 누적 회전 각도 계산
        if agent_heading is not None:
            heading_diff = agent_heading - self.previous_heading

            # 각도 차이 정규화 (-180 ~ 180)
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360

            self.total_rotation += abs(heading_diff)
            self.previous_heading = agent_heading

        # 360도 회전 완료 확인
        if self.total_rotation >= Constants.CIRCLE_COMPLETION_ROTATION:
            self.is_completed = True
            self.target_x = None
            if logger:
                logger.info("🎉 부표 360도 회전 완료! 다음 미션으로 전환됩니다.")
            # 미션 완료 후 천천히 전진
            left_thrust = Constants.CIRCLE_COMPLETION_SPEED * self.thrust_scale
            right_thrust = Constants.CIRCLE_COMPLETION_SPEED * self.thrust_scale
            return left_thrust, right_thrust

        # 부표 미탐지 시 이전 명령 사용
        if not blue_buoy:
            self.target_x = None
            left_cmd = self.last_known_left_cmd
            right_cmd = self.last_known_right_cmd
            if logger:
                logger.warn(
                    f"파란색 부표 미탐지: 이전 명령 사용 L={left_cmd:.1f}, R={right_cmd:.1f}"
                )
            return left_cmd, right_cmd

        # 미션 파라미터
        rotation_direction = mission_params.get('rotation_direction', 1)

        # 부표 측정값
        buoy_depth = blue_buoy['depth']
        buoy_x = blue_buoy['center'][0]

        # 회전 모드: 부표를 기준으로 일정한 방향으로 회전
        target_x = self.calculate_rotation_target(rotation_direction, buoy_depth)
        self.target_x = target_x
        error = target_x - buoy_x

        # 조향 명령 계산 (PID)
        steering_command = self.calculate_steering_command(error)

        # 회전 추력 계산
        turn_thrust = steering_command * self.max_turn_thrust

        # 각도에 따른 적응형 속도 계산
        turn_angle = abs(steering_command * 90)
        forward_thrust = self.calculate_rotation_speed(turn_angle)

        # 스러스터 명령 계산
        left_command = forward_thrust - turn_thrust
        right_command = forward_thrust + turn_thrust

        # 마지막으로 성공한 명령 저장
        self.last_known_left_cmd = left_command
        self.last_known_right_cmd = right_command

        if logger:
            direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
            logger.info(
                f"Circle [ROTATE][{data_source}]: rotation={self.total_rotation:.1f}°, "
                f"부표 위치=({buoy_x:.1f}px), 깊이={buoy_depth:.3f}m, "
                f"목표={target_x:.1f}px, 오차={error:.1f}px, "
                f"조향={steering_command:.3f}, 방향={direction_name}"
            )

        # 스러스터를 thrust_scale로 변환하여 반환
        left_thrust = (left_command / 1000.0) * self.thrust_scale
        right_thrust = (right_command / 1000.0) * self.thrust_scale

        return left_thrust, right_thrust


class WaypointFollowMission(BaseMissionStrategy):
    """미션 3: 웨이포인트 추종"""

    def execute(
        self,
        agent_position: np.ndarray,
        agent_heading: float,
        target_waypoint: Dict,
        logger=None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        단순 웨이포인트 추종

        Args:
            agent_position: 로봇 위치 (x, y)
            agent_heading: 로봇 헤딩 (도)
            target_waypoint: 목표 웨이포인트 {'x': float, 'y': float}
            logger: 로거

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        # 목표 웨이포인트
        target_pos = np.array([target_waypoint['x'], target_waypoint['y']], dtype=np.float32)

        # 목표까지의 벡터
        delta = target_pos - agent_position
        distance = np.linalg.norm(delta)

        if distance < Constants.WAYPOINT_MIN_DISTANCE:
            return 0.0, 0.0

        # 목표 방향 계산
        target_heading = np.degrees(np.arctan2(delta[0], delta[1]))
        if target_heading < 0:
            target_heading += 360

        # 헤딩 오차 계산
        heading_error = target_heading - agent_heading

        # 각도 정규화
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        # 비례 제어
        steering = heading_error * Constants.WAYPOINT_STEERING_GAIN
        steering = np.clip(steering, -Constants.WAYPOINT_MAX_STEERING,
                         Constants.WAYPOINT_MAX_STEERING)

        # 스러스터 명령
        left_thrust = (Constants.WAYPOINT_FORWARD_SPEED + steering) * self.thrust_scale
        right_thrust = (Constants.WAYPOINT_FORWARD_SPEED - steering) * self.thrust_scale

        if logger:
            logger.info(
                f"Waypoint Follow: dist={distance:.1f}m, heading_err={heading_error:.1f}°"
            )

        return left_thrust, right_thrust


class ObstacleAvoidMission(BaseMissionStrategy):
    """미션 4: 장애물 회피"""

    def __init__(self, thrust_scale: float = 1000.0, avoidance_controller=None):
        super().__init__(thrust_scale)
        self.avoidance_controller = avoidance_controller
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

    def reset(self):
        """미션 상태 초기화"""
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

    def execute(
        self,
        agent_position: np.ndarray,
        agent_heading: float,
        waypoints: List[Dict],
        current_waypoint_index: int,
        lidar_distances: np.ndarray,
        get_lidar_distance_func: Callable,
        get_onnx_control_func: Callable,
        logger=None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        ONNX 모델 + 알고리즘 하이브리드 장애물 회피

        Args:
            agent_position: 로봇 위치
            agent_heading: 로봇 헤딩
            waypoints: 웨이포인트 리스트
            current_waypoint_index: 현재 웨이포인트 인덱스
            lidar_distances: LiDAR 거리 배열
            get_lidar_distance_func: LiDAR 거리 조회 함수
            get_onnx_control_func: ONNX 제어 명령 조회 함수
            logger: 로거

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        if not waypoints or current_waypoint_index >= len(waypoints):
            return 0.0, 0.0

        # LOS target 계산
        waypoint_list = [[wp['x'], wp['y']] for wp in waypoints]
        los_target = self.avoidance_controller.get_los_target(
            agent_position, waypoint_list, current_waypoint_index
        )

        # 장애물 확인 및 제어
        use_direct_control, linear_velocity, angular_velocity, _ = \
            self.avoidance_controller.check_obstacles_and_get_control(
                agent_position, los_target, agent_heading,
                lidar_distances, get_lidar_distance_func,
                get_onnx_control_func
            )

        # 필터 적용
        filtered_linear, filtered_angular = self.avoidance_controller.apply_filters(
            linear_velocity, angular_velocity
        )

        # 이전 명령 업데이트
        self.previous_moment_input = filtered_angular
        self.previous_force_input = filtered_linear

        # 스러스터 계산
        left_thrust, right_thrust = self._calculate_thruster_commands(
            filtered_linear, filtered_angular
        )

        # 스러스터 필터 적용
        left_thrust, right_thrust = self.avoidance_controller.apply_thrust_filters(
            left_thrust, right_thrust
        )

        mode = "DIRECT" if use_direct_control else "ONNX"
        if logger:
            logger.info(
                f"Obstacle Avoid [{mode}]: linear={filtered_linear:.3f}, "
                f"angular={filtered_angular:.3f}"
            )

        return left_thrust, right_thrust

    def _calculate_thruster_commands(
        self, linear_velocity: float, angular_velocity: float
    ) -> Tuple[float, float]:
        """
        스러스터 명령 계산

        Args:
            linear_velocity: 선속도
            angular_velocity: 각속도

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        forward_thrust = linear_velocity * self.thrust_scale
        turn_thrust = angular_velocity * self.thrust_scale
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        return left_thrust, right_thrust


# ============================================================================
# 미션 관리자
# ============================================================================

class MissionManager:
    """미션 관리 시스템"""

    def __init__(self, thrust_scale: float = 1000.0, avoidance_controller=None):
        """
        Args:
            thrust_scale: 추력 스케일
            avoidance_controller: 장애물 회피 컨트롤러
        """
        self.thrust_scale = thrust_scale
        self.missions = {
            MissionType.PASS_BETWEEN_BUOYS: PassBetweenBuoysMission(thrust_scale),
            MissionType.CIRCLE_BUOY: CircleBuoyMission(thrust_scale),
            MissionType.WAYPOINT_FOLLOW: WaypointFollowMission(thrust_scale),
            MissionType.OBSTACLE_AVOID: ObstacleAvoidMission(thrust_scale, avoidance_controller)
        }
        self.current_mission: Optional[MissionType] = None

    def set_mission(self, mission_type: MissionType):
        """
        현재 미션 설정 및 초기화

        Args:
            mission_type: 설정할 미션 타입
        """
        if mission_type != self.current_mission:
            # 이전 미션 리셋
            if self.current_mission and self.current_mission in self.missions:
                self.missions[self.current_mission].reset()
            self.current_mission = mission_type

    def update_thrust_scale(self, thrust_scale: float):
        """
        모든 미션의 thrust_scale 업데이트

        Args:
            thrust_scale: 새 추력 스케일
        """
        self.thrust_scale = thrust_scale
        for mission in self.missions.values():
            mission.thrust_scale = thrust_scale

    def execute_mission(self, mission_type: MissionType, **kwargs) -> Tuple[float, float]:
        """
        미션 실행

        Args:
            mission_type: 실행할 미션 타입
            **kwargs: 미션별 파라미터

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust)
        """
        mission = self.missions.get(mission_type)
        if mission:
            return mission.execute(**kwargs)
        return 0.0, 0.0

    def get_circle_buoy_target_x(self) -> Optional[float]:
        """
        CIRCLE_BUOY 미션의 target_x 가져오기 (시각화용)

        Returns:
            Optional[float]: target_x 값 또는 None
        """
        circle_mission = self.missions.get(MissionType.CIRCLE_BUOY)
        if circle_mission and hasattr(circle_mission, 'target_x'):
            return circle_mission.target_x
        return None

    def is_circle_mission_completed(self) -> bool:
        """
        CircleBuoyMission의 완료 여부 확인 (360도 회전 완료)

        Returns:
            bool: 360도 회전이 완료되었으면 True, 아니면 False
        """
        circle_mission = self.missions.get(MissionType.CIRCLE_BUOY)
        if circle_mission:
            # is_completed 플래그가 설정되어 있으면 그 값 사용
            if hasattr(circle_mission, 'is_completed'):
                return circle_mission.is_completed
            # 없으면 total_rotation으로 직접 확인 (후방 호환성)
            if hasattr(circle_mission, 'total_rotation'):
                return circle_mission.total_rotation >= Constants.CIRCLE_COMPLETION_ROTATION
        return False
