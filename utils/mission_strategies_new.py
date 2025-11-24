#!/usr/bin/env python3
"""
미션 전략 모듈
- 4가지 미션의 제어 로직을 캡슐화
- 리팩토링: 중복 제거, 매직 넘버 상수화, 타입 힌트 보완
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Callable, Union

from ..detection.detection_system import MissionType
from ..core.config import Constants
from ..control.thruster_allocation import body_forces_to_thruster_commands
from ..core.helpers import normalize_heading, calculate_heading_error, find_buoy_with_fallback
from ..control.avoid_control import LOSGuidance


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
    """미션 전략 베이스 클래스 (점진적 마이그레이션: 두 인터페이스 모두 지원)"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE):
        self.thrust_scale = thrust_scale

    def _update_params(self, mission_params: Dict, **attr_mapping):
        """
        파라미터 업데이트 헬퍼 (반복 코드 제거)

        Args:
            mission_params: 미션 파라미터 딕셔너리
            **attr_mapping: attr_name='param_key' 형태의 매핑

        Example:
            self._update_params(mission_params,
                base_speed='circle_base_speed',
                min_speed='circle_min_speed')
        """
        for attr_name, param_key in attr_mapping.items():
            if param_key in mission_params:
                setattr(self, attr_name, mission_params[param_key])

    def execute_body_forces(self, **kwargs) -> Tuple[float, float, float]:
        """
        미션 실행 (새 인터페이스: 통일된 body force 명령 반환)

        Returns:
            Tuple[float, float, float]: (desired_speed, desired_yaw, desired_force_y)
        """
        raise NotImplementedError("Subclass must implement execute_body_forces()")

    def execute(self, **kwargs):
        """
        미션 실행 (기존 인터페이스: 하위 호환성 유지)

        Returns:
            - 대부분: Tuple[float, float] (left_thrust, right_thrust)
            - CircleBuoy/Dock: Tuple[float, float, float, float] (left, right, left_pos, right_pos)
        """
        desired_speed, desired_yaw, desired_force_y = self.execute_body_forces(**kwargs)
        left_thrust, right_thrust, left_pos, right_pos = body_forces_to_thruster_commands(
            desired_speed, desired_yaw, desired_force_y, self.thrust_scale
        )
        return left_thrust, right_thrust

    def reset(self):
        """미션 상태 초기화"""
        pass


class PassBetweenBuoysMission(BaseMissionStrategy):
    """미션 1: 부표 사이 지나가기 (리팩토링: body force 반환)"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        # PID 제어기 (사용하지 않지만 하위 호환성 유지)
        self.pid_controller = PIDController(
            kp=Constants.PASS_BETWEEN_PID_KP,
            ki=Constants.PASS_BETWEEN_PID_KI,
            kd=Constants.PASS_BETWEEN_PID_KD
        )
        # 미션 시작 위치 저장 (부표 미탐지 시 LOS guidance의 이전 웨이포인트로 사용)
        self.mission_start_position: Optional[np.ndarray] = None
        # LOS Guidance 시스템 (부표 미탐지 시 사용)
        self.los_guidance = LOSGuidance()

    def reset(self):
        """미션 상태 초기화"""
        self.pid_controller.reset()
        self.mission_start_position = None

    def execute_body_forces(
        self,
        detected_objects: List[Dict],
        current_image: np.ndarray,
        logger=None,
        raw_detections: Optional[List[Dict]] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        previous_waypoint: Optional[Dict] = None,
        current_waypoint: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        빨간색/초록색 고깔 부표 사이로 지나가기

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            current_image: 현재 카메라 이미지
            logger: 로거
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)
            agent_position: 로봇 현재 위치 (x, y)
            agent_heading: 로봇 현재 헤딩 (도)
            previous_waypoint: 이전 웨이포인트 (미사용, 호환성 유지)
            current_waypoint: 현재 목표 웨이포인트 {'x': float, 'y': float}

        Returns:
            Tuple[float, float, float]: (desired_speed, desired_yaw, desired_force_y)

        Note:
            부표 미탐지 시 LOS guidance 사용:
            - 이전 웨이포인트가 없을 경우(첫 번째 WP) → 미션 시작 위치를 이전 기준점으로 사용
            - 이전 웨이포인트가 있을 경우 → 미션 시작 위치를 이전 기준점으로 사용 (일관성 유지)
        """
        # 미션 시작 위치 저장 (첫 실행 시) - 부표 미탐지 시 LOS의 이전 기준점으로 사용
        if self.mission_start_position is None and agent_position is not None:
            self.mission_start_position = agent_position.copy()
            if logger:
                logger.info(
                    f"PassBetweenBuoys 미션 시작 위치 저장 (LOS 이전 기준점): "
                    f"({agent_position[0]:.2f}, {agent_position[1]:.2f})"
                )
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

            # Body force 명령 계산
            desired_speed = Constants.PASS_BETWEEN_FORWARD_SPEED
            desired_yaw = steering
            desired_force_y = 0.0

            if logger:
                data_source = f"R:{red_source}/G:{green_source}"
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, "
                    f"error={error:.1f}, steering={steering:.3f}"
                )
        else:
            # 부표 미탐지 시: LOS guidance로 다음 웨이포인트 추종
            #
            # LOS guidance 구조:
            #   - waypoint_start (이전 기준점) = 미션 시작 위치
            #     * 첫 번째 웨이포인트인 경우: config.py에서 이전 WP 지정 안 함 → 자동으로 미션 시작 위치 사용
            #     * 이후 웨이포인트인 경우: 일관성을 위해 동일하게 미션 시작 위치 사용
            #   - waypoint_end (목표) = 현재 웨이포인트
            if (agent_position is not None and agent_heading is not None and
                self.mission_start_position is not None and current_waypoint is not None):

                # LOS guidance 계산
                # waypoint_start: 미션 시작 위치 (이전 기준점, 자동 설정)
                # waypoint_end: 현재 목표 웨이포인트
                waypoint_start = self.mission_start_position
                waypoint_end = np.array([current_waypoint['x'], current_waypoint['y']])

                # LOS target 계산
                los_target = self.los_guidance.calculate_los_point(
                    agent_position, waypoint_start, waypoint_end
                )

                # LOS target 방향 계산
                delta = los_target - agent_position
                distance = np.linalg.norm(delta)

                if distance > 0.5:  # 최소 거리 체크
                    # 목표 방향 계산 (NED 좌표계: x=North, y=East)
                    target_heading = np.degrees(np.arctan2(delta[0], delta[1]))
                    if target_heading < 0:
                        target_heading += 360

                    # 헤딩 오차 계산
                    heading_error = calculate_heading_error(target_heading, agent_heading)

                    # 비례 제어
                    steering = heading_error * Constants.PASS_BETWEEN_STEERING_GAIN
                    steering = np.clip(steering, -Constants.PASS_BETWEEN_MAX_STEERING,
                                     Constants.PASS_BETWEEN_MAX_STEERING)

                    desired_speed = Constants.PASS_BETWEEN_FALLBACK_SPEED
                    desired_yaw = steering
                    desired_force_y = 0.0

                    if logger:
                        logger.warn(
                            f"부표 미탐지: LOS guidance로 다음 WP 추종 "
                            f"(목표={target_heading:.1f}°, 오차={heading_error:.1f}°, steering={steering:.3f})"
                        )
                else:
                    # LOS target 도착 - 천천히 전진
                    desired_speed = Constants.PASS_BETWEEN_FALLBACK_SPEED
                    desired_yaw = 0.0
                    desired_force_y = 0.0
                    if logger:
                        logger.warn("부표 미탐지 & LOS target 도착: 천천히 전진")
            else:
                # 위치/헤딩/웨이포인트 정보 없으면 천천히 전진 (기존 동작)
                desired_speed = Constants.PASS_BETWEEN_FALLBACK_SPEED
                desired_yaw = 0.0
                desired_force_y = 0.0
                if logger:
                    logger.warn("부표 미탐지 (LOS 정보 부족): 천천히 전진")

        return desired_speed, desired_yaw, desired_force_y


class CircleBuoyMission(BaseMissionStrategy):
    """미션 2: 웨이포인트 기반 원 궤적 그리기"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        # 상태 변수
        self.mission_start_time: Optional[float] = None
        self.mission_start_position: Optional[np.ndarray] = None
        self.mission_start_heading: Optional[float] = None
        self.is_completed = False

        # 웨이포인트 관리
        self.generated_waypoints: List[np.ndarray] = []
        self.current_waypoint_index = 0

        # 미션 파라미터 (mission_params에서 설정됨)
        self.length = 10.0  # 마름모 크기
        self.radius_reach = 5.0  # 웨이포인트 도착 판정 거리
        self.max_duration = 80.0  # 최대 미션 시간 (초)
        self.rotation_direction = 1  # 1=시계방향, 0=반시계방향

        # PID 제어기 (헤딩 오차 제어용)
        # 기본 게인: kp=0.003 (기존 steering_gain), ki=0.001, kd=0.002
        self.pid_controller = PIDController(
            kp=0.01,  # 비례 게인 (기존 steering_gain과 동일)
            ki=0.00001,  # 적분 게인
            kd=0.002   # 미분 게인
        )

    def reset(self):
        """미션 상태 초기화"""
        self.mission_start_time = None
        self.mission_start_position = None
        self.mission_start_heading = None
        self.is_completed = False
        self.generated_waypoints = []
        self.current_waypoint_index = 0
        self.pid_controller.reset()  # PID 제어기 상태 초기화

    def get_generated_waypoints(self) -> List[np.ndarray]:
        """
        생성된 웨이포인트 반환 (시각화용)

        Returns:
            List[np.ndarray]: world-frame 웨이포인트 리스트 [[x, y], ...]
        """
        return self.generated_waypoints

    def _body_to_world_frame(self, body_x: float, body_y: float,
                             robot_pos: np.ndarray, robot_heading: float) -> np.ndarray:
        """
        body-frame 좌표를 world-frame 좌표로 변환

        Args:
            body_x: body-frame x 좌표 (전방, 미터)
            body_y: body-frame y 좌표 (좌측, 미터)
            robot_pos: 로봇 현재 위치 [Easting, Northing] (world-frame)
            robot_heading: 로봇 현재 헤딩 (도, 0=North, 90=East)

        Returns:
            np.ndarray: world-frame 좌표 [Easting, Northing]

        Note:
            - body-frame: 로봇 기준 (전방=x+, 좌측=y+)
            - world-frame: [Easting, Northing] 형식 (agent_position과 동일)
            - heading: 0도=North, 90도=East, 180도=South, 270도=West

        변환 공식:
            - Forward 벡터 (body_x): heading 방향
              → North 성분: body_x * cos(θ)
              → East 성분: body_x * sin(θ)
            - Left 벡터 (body_y): heading - 90도 방향
              → North 성분: -body_y * sin(θ)
              → East 성분: body_y * cos(θ)
            
            [Easting]  = [robot_pos[0]] + [sin(θ)   cos(θ)] [body_x]
            [Northing]   [robot_pos[1]]   [cos(θ)  -sin(θ)] [body_y]
        """
        # 헤딩을 라디안으로 변환
        heading_rad = np.radians(robot_heading)
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)

        # body-frame → world-frame 변환
        # robot_pos = [Easting, Northing]
        # Forward (body_x) → (East: sin(θ), North: cos(θ))
        # Left (body_y) → (East: cos(θ), North: -sin(θ))
        easting = robot_pos[0] + (body_x * sin_h + body_y * cos_h)
        northing = robot_pos[1] + (body_x * cos_h - body_y * sin_h)

        return np.array([easting, northing], dtype=np.float32)

    def _generate_waypoints(self, robot_pos: np.ndarray, robot_heading: float,
                           rotation_direction: int, length: float) -> List[np.ndarray]:
        """
        마름모 형태의 웨이포인트 생성 (로봇의 현재 각도 기준 body-frame)

        Args:
            robot_pos: 로봇 현재 위치 [Easting, Northing] (전역 좌표)
            robot_heading: 로봇 현재 헤딩 (도, 0=North, 90=East) - 이 각도만큼 회전된 좌표계에서 생성
            rotation_direction: 회전 방향 (1=시계방향, 0=반시계방향)
            length: 마름모 크기 (미터)

        Returns:
            List[np.ndarray]: 전역 좌표 웨이포인트 리스트 (로봇 위치 및 각도 기준)
            형식: [[Easting_global, Northing_global], ...]

        Note:
            - length 기준으로 body-frame 좌표 생성
            - 로봇의 현재 각도(robot_heading)만큼 회전된 좌표계에서 생성됨
            - _body_to_world_frame 함수를 사용하여 전역 좌표로 변환
            - 시계방향(1): [(-length, length), (0, 2*length), (length, length), (0, 0)]
            - 반시계방향(0): [(length, length), (0, 2*length), (-length, length), (0, 0)]
            - body-frame: Forward=x+, Left=y+
            - world-frame: Easting=x+, Northing=y+
        """
        # body-frame 좌표 정의 (length 기준)
        # 로봇의 현재 각도(robot_heading)를 기준으로 회전된 좌표계에서 생성됨
        # Forward = 로봇의 현재 heading 방향, Left = 로봇의 왼쪽 방향
        if rotation_direction == 1:  # 시계방향
            body_waypoints = [
                (-length, length),   # [Forward, Left] - body-frame 기준
                (0, 2 * length),
                (length, length),
                (0, 0)
            ]
        else:  # 반시계방향
            body_waypoints = [
                (length, length),
                (0, 2 * length),
                (-length, length),
                (0, 0)
            ]

        # body-frame → world-frame 변환 (로봇의 현재 각도 적용)
        # 회전 좌표계: 로봇의 현재 각도(robot_heading)만큼 회전된 좌표계에서 생성
        # _body_to_world_frame 함수가 회전 변환을 수행하여 전역 좌표로 변환
        global_waypoints = []
        for body_forward, body_left in body_waypoints:
            # body-frame 좌표를 로봇의 현재 각도만큼 회전하여 전역 좌표로 변환
            # 이렇게 하면 로봇의 현재 방향을 기준으로 웨이포인트가 생성됨
            global_wp = self._body_to_world_frame(
                body_forward,  # body-frame Forward (로봇 전방 방향)
                body_left,     # body-frame Left (로봇 왼쪽 방향)
                robot_pos,     # 로봇 현재 위치 [Easting, Northing]
                robot_heading  # 로봇 현재 헤딩 (도) - 이 각도만큼 회전된 좌표계
            )
            global_waypoints.append(global_wp)

        return global_waypoints

    def _check_waypoint_reached(self, robot_pos: np.ndarray, target_wp: np.ndarray,
                                radius: float) -> bool:
        """
        웨이포인트 도착 판정

        Args:
            robot_pos: 로봇 현재 위치 [x, y]
            target_wp: 목표 웨이포인트 [x, y]
            radius: 도착 판정 반경 (미터)

        Returns:
            bool: 도착했으면 True
        """
        distance = np.linalg.norm(robot_pos - target_wp)
        return distance < radius

    def execute(
        self,
        agent_position: Optional[np.ndarray],
        agent_heading: Optional[float],
        mission_params: Dict,
        logger=None,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """
        웨이포인트 기반 원 궤적 그리기

        Args:
            agent_position: 로봇 현재 위치 [x, y]
            agent_heading: 로봇 헤딩 (도)
            mission_params: 미션 파라미터 (rotation_direction, length, radius_reach, max_duration)
            logger: 로거

        Returns:
            Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos, right_pos)
        """
        # 미션 파라미터 업데이트
        self.rotation_direction = mission_params.get('rotation_direction', 1)
        self.length = mission_params.get('length', 10.0)
        self.radius_reach = mission_params.get('radius_reach', 5.0)
        self.max_duration = mission_params.get('max_duration', 80.0)

        # 위치/헤딩 정보가 없으면 정지
        if agent_position is None or agent_heading is None:
            if logger:
                logger.warn("CIRCLE_BUOY: 위치/헤딩 정보 없음 - 정지")
            return 0.0, 0.0, 0.0, 0.0

        # 미션 시작 시 초기화
        if self.mission_start_time is None:
            self.mission_start_time = time.time()
            self.mission_start_position = agent_position.copy()
            self.mission_start_heading = agent_heading

            # 웨이포인트 생성
            self.generated_waypoints = self._generate_waypoints(
                agent_position, agent_heading, self.rotation_direction, self.length
            )
            self.current_waypoint_index = 0
            
            # PID 제어기 초기화 (미션 시작 시)
            self.pid_controller.reset()

            if logger:
                direction_str = "시계방향" if self.rotation_direction == 1 else "반시계방향"
                logger.info(
                    f"🎯 CIRCLE_BUOY 미션 시작: 위치=({agent_position[0]:.2f}, {agent_position[1]:.2f}), "
                    f"헤딩={agent_heading:.1f}°, 방향={direction_str}, length={self.length:.1f}m"
                )
                for i, wp in enumerate(self.generated_waypoints):
                    logger.info(f"  WP{i}: ({wp[0]:.2f}, {wp[1]:.2f})")

        # 타임아웃 체크
        elapsed_time = time.time() - self.mission_start_time
        if elapsed_time > self.max_duration:
            self.is_completed = True
            if logger:
                logger.warn(
                    f"⏱️ CIRCLE_BUOY 타임아웃 ({elapsed_time:.1f}s > {self.max_duration:.1f}s): "
                    f"미션 종료"
                )
            return 0.0, 0.0, 0.0, 0.0

        # 모든 웨이포인트 통과 확인
        if self.current_waypoint_index >= len(self.generated_waypoints):
            self.is_completed = True
            if logger:
                logger.info(
                    f"✅ CIRCLE_BUOY 완료: 모든 웨이포인트 통과 ({elapsed_time:.1f}s)"
                )
            return 0.0, 0.0, 0.0, 0.0

        # 현재 목표 웨이포인트
        target_waypoint = self.generated_waypoints[self.current_waypoint_index]

        # 웨이포인트 도착 판정
        if self._check_waypoint_reached(agent_position, target_waypoint, self.radius_reach):
            if logger:
                logger.info(
                    f"✓ WP{self.current_waypoint_index} 도착: "
                    f"({target_waypoint[0]:.2f}, {target_waypoint[1]:.2f})"
                )
            self.current_waypoint_index += 1

            # 마지막 웨이포인트 도착 확인
            if self.current_waypoint_index >= len(self.generated_waypoints):
                self.is_completed = True
                if logger:
                    logger.info(
                        f"✅ CIRCLE_BUOY 완료: 모든 웨이포인트 통과 ({elapsed_time:.1f}s)"
                    )
                return 0.0, 0.0, 0.0, 0.0

            # 다음 웨이포인트로 업데이트
            target_waypoint = self.generated_waypoints[self.current_waypoint_index]

        # 목표 방향 계산
        # agent_position = [Easting, Northing]
        # target_waypoint = [Easting, Northing]
        delta = target_waypoint - agent_position
        distance = np.linalg.norm(delta)

        if distance < 0.5:  # 너무 가까우면 정지
            return 0.0, 0.0, 0.0, 0.0

        # 목표 헤딩 계산
        # delta = [Easting_diff, Northing_diff]
        # atan2(Easting, Northing) = 목표 방향 (0도=North, 90도=East)
        target_heading = np.degrees(np.arctan2(delta[0], delta[1]))
        if target_heading < 0:
            target_heading += 360

        # 헤딩 오차 계산
        heading_error = calculate_heading_error(target_heading, agent_heading)

        # PID 제어 (헤딩 오차 기반)
        # heading_error는 도(degree) 단위이므로 라디안으로 변환하여 PID 제어
        heading_error_rad = np.radians(heading_error)
        steering = self.pid_controller.update(heading_error_rad)
        
        # 최대 조향 제한
        max_steering = 0.5
        steering = np.clip(steering, -max_steering, max_steering)

        # 전진 속도 (일정하게 유지)
        forward_speed = 0.4

        # Body force 명령 계산
        desired_speed = forward_speed
        desired_yaw = steering
        desired_force_y = 0.0

        # Thruster 명령으로 변환
        left_thrust, right_thrust, left_pos, right_pos = body_forces_to_thruster_commands(
            desired_speed, desired_yaw, desired_force_y, self.thrust_scale
        )

        if logger:
            logger.info(
                f"CIRCLE_BUOY [WP{self.current_waypoint_index}/{len(self.generated_waypoints)-1}]: "
                f"목표=({target_waypoint[0]:.2f}, {target_waypoint[1]:.2f}), "
                f"거리={distance:.2f}m, 헤딩오차={heading_error:.1f}°, "
                f"조향={steering:.3f}, 경과시간={elapsed_time:.1f}s"
            )

        return left_thrust, right_thrust, left_pos, right_pos


class WaypointFollowMission(BaseMissionStrategy):
    """미션 3: 웨이포인트 추종 (리팩토링: body force 반환)"""

    def execute_body_forces(
        self,
        agent_position: np.ndarray,
        agent_heading: float,
        target_waypoint: Dict,
        logger=None,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        단순 웨이포인트 추종

        Args:
            agent_position: 로봇 위치 (x, y)
            agent_heading: 로봇 헤딩 (도)
            target_waypoint: 목표 웨이포인트 {'x': float, 'y': float}
            logger: 로거

        Returns:
            Tuple[float, float, float]: (desired_speed, desired_yaw, desired_force_y)
        """
        # 목표 웨이포인트
        target_pos = np.array([target_waypoint['x'], target_waypoint['y']], dtype=np.float32)

        # 목표까지의 벡터
        delta = target_pos - agent_position
        distance = np.linalg.norm(delta)

        if distance < Constants.WAYPOINT_MIN_DISTANCE:
            return 0.0, 0.0, 0.0

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

        # Body force 명령
        desired_speed = Constants.WAYPOINT_FORWARD_SPEED
        desired_yaw = steering
        desired_force_y = 0.0

        if logger:
            logger.info(
                f"Waypoint Follow: dist={distance:.1f}m, heading_err={heading_error:.1f}°"
            )

        return desired_speed, desired_yaw, desired_force_y


class ObstacleAvoidMission(BaseMissionStrategy):
    """미션 4: 장애물 회피 (리팩토링: body force 반환)"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE, avoidance_controller=None):
        super().__init__(thrust_scale)
        self.avoidance_controller = avoidance_controller
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

    def reset(self):
        """미션 상태 초기화"""
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

    def execute_body_forces(
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
    ) -> Tuple[float, float, float]:
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
            Tuple[float, float, float]: (desired_speed, desired_yaw, desired_force_y)
        """
        if not waypoints or current_waypoint_index >= len(waypoints):
            return 0.0, 0.0, 0.0

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

        mode = "DIRECT" if use_direct_control else "ONNX"
        if logger:
            logger.info(
                f"Obstacle Avoid [{mode}]: linear={filtered_linear:.3f}, "
                f"angular={filtered_angular:.3f}"
            )

        # Body force 반환 (sway는 사용하지 않음)
        return filtered_linear, filtered_angular, 0.0


class DockMission(BaseMissionStrategy):
    """미션 5: 도킹 스테이션 미션"""

    def __init__(self, thrust_scale: float = Constants.DOCK_DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        # 상태 변수
        self.docking_phase = "TRACKING"
        self.phase_start_time: Optional[float] = None
        self.target_shape_label: Optional[str] = None
        self.initial_heading: Optional[float] = None
        self.initial_imu_heading: Optional[float] = None
        self.accumulated_angle = 0.0
        self.previous_heading: Optional[float] = None
        self.rotation_direction = 1
        self.is_completed = False

        # 동적 파라미터 (트랙바로 변경 가능)
        self.sway_gain = Constants.DOCK_SWAY_GAIN
        self.yaw_gain = Constants.DOCK_YAW_GAIN
        self.max_sway_thrust = Constants.DOCK_MAX_SWAY_THRUST
        self.max_yaw_thrust = Constants.DOCK_MAX_YAW_THRUST
        self.base_surge = Constants.DOCK_BASE_SURGE
        self.depth_threshold = Constants.DOCK_DEPTH_THRESHOLD
        self.sway_to_yaw_threshold = 0.5
        self.approach_time = Constants.DOCK_APPROACH_TIME
        self.reverse_time = Constants.DOCK_REVERSE_TIME
        self.approach_speed = Constants.DOCK_APPROACH_SPEED
        self.reverse_speed = Constants.DOCK_REVERSE_SPEED
        self.center_tolerance = Constants.DOCK_CENTER_TOLERANCE
        self.sway_strength = Constants.DOCK_SWAY_STRENGTH

        # Body-force 명령 저장 (ROS 퍼블리시용)
        self.last_sway_force = self.last_yaw_moment = self.last_surge_velocity = 0.0

    def reset(self):
        """미션 상태 초기화"""
        self.docking_phase = "TRACKING"
        self.phase_start_time = self.target_shape_label = None
        self.initial_heading = self.initial_imu_heading = self.previous_heading = None
        self.accumulated_angle = 0.0
        self.rotation_direction = 1
        self.is_completed = False

    def set_target_shape(self, shape_label: str):
        """
        추적할 도형 설정

        Args:
            shape_label: 도형 라벨 (예: "red_circle", "blue_square")
        """
        self.target_shape_label = shape_label

    def get_last_body_forces(self) -> Tuple[float, float, float]:
        """최근 SWAY/YAW/SURGE 명령 반환"""
        return self.last_sway_force, self.last_yaw_moment, self.last_surge_velocity

    def _set_body_forces(self, sway_force: float = 0.0, yaw_moment: float = 0.0,
                         surge_velocity: float = 0.0):
        """최근 body-force 명령 저장"""
        clip_value = Constants.DOCK_BODY_FORCE_CLIP_VALUE
        self.last_sway_force = float(np.clip(sway_force, -clip_value, clip_value))
        self.last_yaw_moment = float(np.clip(yaw_moment, -clip_value, clip_value))
        self.last_surge_velocity = float(np.clip(surge_velocity, -clip_value, clip_value))

    def execute(
        self,
        detected_objects: List[Dict],
        current_image: np.ndarray,
        logger=None,
        raw_detections: Optional[List[Dict]] = None,
        mission_params: Optional[Dict] = None,
        agent_heading: Optional[float] = None,
        **kwargs
    ) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
        """
        도킹 스테이션 미션 실행

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력)
            current_image: 현재 카메라 이미지
            logger: 로거
            raw_detections: 원본 탐지 결과
            mission_params: 미션 파라미터
            agent_heading: 로봇 헤딩 (도, -180~180)

        Returns:
            Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
                (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        # 초기 헤딩 설정 (도킹 미션 시작 시점, 처음 각도를 0으로 고정)
        if self.initial_heading is None and agent_heading is not None:
            self.initial_heading = 0.0  # 처음 누적 각도는 0으로 고정
            self.initial_imu_heading = agent_heading  # 처음 IMU 헤딩 값 저장 (보정 기준)
            self.previous_heading = agent_heading
            self.accumulated_angle = 0.0
            self.rotation_direction = 1  # 초기값 오른쪽
            if logger:
                logger.info(f"🎯 도킹 미션 시작: 초기 각도 = 0° (실제 IMU 헤딩 = {agent_heading:.2f}°)")

        # 누적 각도 계산 (변화율만 측정, 부호 반대)
        if agent_heading is not None and self.previous_heading is not None:
            heading_diff = agent_heading - self.previous_heading

            # 각도 차이 정규화 (-180 ~ 180)
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360

            if self.initial_imu_heading is not None:
                # 현재 각도를 DOCK_MODE 시작 각도 대비 0 기준으로 환산
                self.accumulated_angle = calculate_heading_error(
                    agent_heading,
                    self.initial_imu_heading
                )
            else:
                # 초기 헤딩을 아직 모르면 상대적 변화량으로 추적
                self.accumulated_angle -= heading_diff

            # 회전 방향 플래그 업데이트 (0=왼쪽, 1=오른쪽)
            # heading_diff < 0: 왼쪽 회전 (실제 헤딩 감소)
            # heading_diff > 0: 오른쪽 회전 (실제 헤딩 증가)
            if heading_diff < 0:
                self.rotation_direction = 0  # 왼쪽 회전
            elif heading_diff > 0:
                self.rotation_direction = 1  # 오른쪽 회전
            # heading_diff == 0인 경우 방향 유지

            self.previous_heading = agent_heading

        # 미션 파라미터 업데이트
        if mission_params:
            self.sway_gain = mission_params.get('dock_sway_gain', self.sway_gain)
            self.yaw_gain = mission_params.get('dock_yaw_gain', self.yaw_gain)
            self.depth_threshold = mission_params.get('dock_depth_threshold', self.depth_threshold)
            self.center_tolerance = mission_params.get('dock_center_tolerance', self.center_tolerance)
            self.sway_strength = mission_params.get('dock_sway_strength', self.sway_strength)
            if 'target_shape' in mission_params:
                self.set_target_shape(mission_params['target_shape'])

        # 목표 도형이 설정되지 않은 경우 첫 번째 탐지된 객체 사용
        if self.target_shape_label is None and (detected_objects or raw_detections):
            if detected_objects:
                self.target_shape_label = detected_objects[0]['label']
            elif raw_detections:
                self.target_shape_label = raw_detections[0]['label']
            if logger:
                logger.info(f"🎯 Dock 목표 도형 설정: {self.target_shape_label}")

        # 목표 도형 찾기
        target_object, data_source = find_buoy_with_fallback(
            self.target_shape_label, detected_objects, raw_detections, logger
        )

        # 도킹 단계별 처리
        if self.docking_phase == "TRACKING":
            return self._execute_tracking_phase(
                target_object, current_image, data_source, logger
            )
        elif self.docking_phase == "APPROACHING":
            return self._execute_approaching_phase(logger)
        elif self.docking_phase == "REVERSING":
            return self._execute_reversing_phase(logger)
        else:
            # 미션 완료
            return 0.0, 0.0, 0.0, 0.0, None

    def _execute_tracking_phase(
        self, target_object: Optional[Dict], current_image: np.ndarray,
        data_source: str, logger
    ) -> Tuple[float, float, float, float, Optional[float]]:
        """
        추적 단계: 목표 도형을 이미지 중심으로 추적

        Returns:
            Tuple: (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        if not target_object:
            # 목표 미탐지 시 정지
            if logger:
                logger.warn(f"목표 도형 '{self.target_shape_label}' 미탐지: 정지")
            self._set_body_forces(0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, None

        # 목표 객체 정보
        target_x = target_object['center'][0]
        target_depth = target_object.get('depth', 0.0)

        # 이미지 중심
        image_center_x = current_image.shape[1] / 2
        image_height = current_image.shape[0]

        # Depth 임계값 확인 (가까워지면 접근 단계로 전환)
        if target_depth >= self.depth_threshold:
            self.docking_phase = "APPROACHING"
            self.phase_start_time = time.time()
            if logger:
                logger.info(f"🚢 Depth 임계값 도달 (depth={target_depth:.3f}): APPROACHING 단계 시작")
            return self._execute_approaching_phase(logger)

        # 픽셀 오차 계산
        error_x = target_x - image_center_x

        # === X 좌표 기반 SWAY 제어 (중앙 허용 범위 적용) ===
        # error_x가 중앙 ±center_tolerance 범위 밖에 있을 경우만 SWAY 적용
        if abs(error_x) > self.center_tolerance:
            # 중앙 허용 범위를 벗어난 오차만 계산
            if error_x > self.center_tolerance:
                # 오른쪽으로 벗어남 → 오른쪽으로 SWAY
                effective_error_x = error_x - self.center_tolerance
                sway_direction = 1.0  # 오른쪽
            else:
                # 왼쪽으로 벗어남 → 왼쪽으로 SWAY
                effective_error_x = error_x + self.center_tolerance
                sway_direction = -1.0  # 왼쪽
        else:
            # 중앙 허용 범위 내에 있으면 SWAY 적용 안함
            effective_error_x = 0.0
            sway_direction = 0.0

        # 정규화된 오차 계산 (-1 ~ 1)
        normalized_error_x = error_x / (current_image.shape[1] / 2)

        # === SWAY와 Moment(YAW) 동시 제어 ===

        # 1. SWAY 제어: X 좌표 기반 횡방향 이동 (오차 크기에 비례)
        if effective_error_x != 0.0:
            # 오차 크기에 비례하도록 개선 (너무 강하지 않게)
            error_ratio = abs(effective_error_x) / (current_image.shape[1] / 2)
            sway_force = sway_direction * self.sway_strength * min(
                1.0, error_ratio * Constants.DOCK_ERROR_RATIO_MULTIPLIER
            )
            sway_force = np.clip(
                sway_force,
                -Constants.DOCK_SWAY_INITIAL_CLIP_VALUE,
                Constants.DOCK_SWAY_INITIAL_CLIP_VALUE
            )
        else:
            sway_force = 0.0

        # Accumulated angle 피드백 추가 (게인 감소로 안정성 향상)
        # accumulated_angle < 0 (음수, 왼쪽 회전 누적) → SWAY를 오른쪽으로 (+)
        # accumulated_angle > 0 (양수, 오른쪽 회전 누적) → SWAY를 왼쪽으로 (-)
        angle_feedback = self.accumulated_angle * Constants.DOCK_ANGLE_FEEDBACK_GAIN
        sway_force += angle_feedback
        sway_force = np.clip(
            sway_force,
            -Constants.DOCK_SWAY_FINAL_CLIP_VALUE,
            Constants.DOCK_SWAY_FINAL_CLIP_VALUE
        )  # SWAY 최대값 제한 (과도한 횡이동 방지)

        # 2. Moment(YAW) 제어: error_x 기반 회전 (객체 x좌표를 이미지 중심으로)
        # error_x > 0: 객체가 오른쪽에 있음 → 오른쪽으로 회전 (양수 moment)
        # error_x < 0: 객체가 왼쪽에 있음 → 왼쪽으로 회전 (음수 moment)
        yaw_moment = normalized_error_x * self.yaw_gain
        yaw_moment = np.clip(
            yaw_moment,
            -Constants.DOCK_YAW_CLIP_VALUE,
            Constants.DOCK_YAW_CLIP_VALUE
        )

        # 3. SURGE: 전진 속도 (SWAY 사용 시에도 최소 속도 보장)
        # SWAY와 SURGE를 동시에 사용 가능하도록 개선
        surge_velocity = self.base_surge * max(
            Constants.DOCK_MIN_SURGE_RATIO,
            1.0 - Constants.DOCK_SURGE_REDUCTION_FACTOR * abs(sway_force)
        )
        surge_velocity = max(0.0, surge_velocity)

        control_mode = "SWAY_YAW_CONTROL"

        # 2-Motor Vectored Thruster Allocation (통합 모듈 사용)
        left_thrust, right_thrust, left_pos, right_pos = body_forces_to_thruster_commands(
            surge_velocity, yaw_moment, sway_force, self.thrust_scale, use_vectored_thrusters=True
        )
        self._set_body_forces(sway_force, yaw_moment, surge_velocity)

        if logger:
            direction_str = "왼쪽" if self.rotation_direction == 0 else "오른쪽"
            logger.info(
                f"Dock TRACKING [{control_mode}][{data_source}]: "
                f"target_x={target_x:.1f}, error_x={error_x:.1f}, depth={target_depth:.3f}, "
                f"누적각도={self.accumulated_angle:.1f}° (방향={direction_str}), "
                f"angle_fb={angle_feedback:.3f}, "
                f"sway={sway_force:.3f}, yaw={yaw_moment:.3f}, surge={surge_velocity:.3f}, "
                f"L_pos={left_pos:.2f}, L={left_thrust:.1f}, R_pos={right_pos:.2f}, R={right_thrust:.1f}"
            )

        return left_thrust, right_thrust, left_pos, right_pos, target_depth

    def _execute_approaching_phase(self, logger) -> Tuple[float, float, float, float, Optional[float]]:
        """
        접근 단계: 직진하여 도킹 스테이션에 접근

        Returns:
            Tuple: (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        if self.phase_start_time is None:
            self.phase_start_time = time.time()

        elapsed_time = time.time() - self.phase_start_time

        if elapsed_time < self.approach_time:
            # 직진 (thruster 각도 0, 일정한 추력)
            left_pos = 0.0
            right_pos = 0.0
            left_thrust = self.approach_speed * self.thrust_scale
            right_thrust = self.approach_speed * self.thrust_scale
            self._set_body_forces(0.0, 0.0, self.approach_speed)

            if logger:
                logger.info(
                    f"Dock APPROACHING: 직진 중... ({elapsed_time:.1f}/{self.approach_time:.1f}s)"
                )

            return left_thrust, right_thrust, left_pos, right_pos, None
        else:
            # 접근 완료, 정지 후 후진 단계로 전환
            self.docking_phase = "REVERSING"
            self.phase_start_time = time.time()

            if logger:
                logger.info("🛑 도킹 완료! REVERSING 단계 시작")

            # 정지
            self._set_body_forces(0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, None

    def _execute_reversing_phase(self, logger) -> Tuple[float, float, float, float, Optional[float]]:
        """
        후진 단계: 도킹 스테이션에서 후진

        Returns:
            Tuple: (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        if self.phase_start_time is None:
            self.phase_start_time = time.time()

        elapsed_time = time.time() - self.phase_start_time

        if elapsed_time < self.reverse_time:
            # 후진 (thruster 각도 0, 음수 추력)
            left_pos = 0.0
            right_pos = 0.0
            left_thrust = self.reverse_speed * self.thrust_scale
            right_thrust = self.reverse_speed * self.thrust_scale
            self._set_body_forces(0.0, 0.0, self.reverse_speed)

            if logger:
                logger.info(
                    f"Dock REVERSING: 후진 중... ({elapsed_time:.1f}/{self.reverse_time:.1f}s)"
                )

            return left_thrust, right_thrust, left_pos, right_pos, None
        else:
            # 후진 완료, 미션 종료
            self.docking_phase = "COMPLETED"
            self.is_completed = True

            if logger:
                logger.info("✅ Dock 미션 완료!")

            # 정지
            self._set_body_forces(0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, None


class RotationMission(BaseMissionStrategy):
    """미션 6: 제자리 선회 (리팩토링: body force 반환)"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        self.target_angle: Optional[float] = None
        self.is_completed: bool = False
        self.stable_frames: int = 0

    def reset(self):
        self.target_angle = None
        self.is_completed = False
        self.stable_frames = 0

    def execute_body_forces(
        self,
        agent_heading: Optional[float],
        mission_params: Optional[Dict],
        logger=None,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        제자리에서 목표 각도까지 선회

        Args:
            agent_heading: 현재 헤딩 (deg)
            mission_params: {'desired_angle': float, ...}

        Returns:
            Tuple[float, float, float]: (desired_speed, desired_yaw, desired_force_y)
        """
        if agent_heading is None or mission_params is None:
            return 0.0, 0.0, 0.0

        desired_angle = mission_params.get('desired_angle', Constants.ROTATION_DEFAULT_TARGET)
        if self.target_angle is None or self.target_angle != desired_angle:
            self.target_angle = desired_angle
            self.stable_frames = 0
            self.is_completed = False

        gain = mission_params.get('rotation_gain', Constants.ROTATION_GAIN)
        tolerance = mission_params.get('rotation_tolerance', Constants.ROTATION_TOLERANCE)
        required_stable = mission_params.get(
            'rotation_stable_frames', Constants.ROTATION_STABLE_FRAMES
        )
        max_thrust = mission_params.get('rotation_max_thrust', Constants.ROTATION_MAX_THRUST)

        error = calculate_heading_error(self.target_angle, agent_heading)

        if abs(error) <= tolerance:
            self.stable_frames += 1
        else:
            self.stable_frames = 0

        if self.stable_frames >= required_stable:
            if not self.is_completed and logger:
                logger.info(
                    f"Rotation 완료: 목표={self.target_angle:.1f}°, 현재={agent_heading:.1f}°"
                )
            self.is_completed = True
            return 0.0, 0.0, 0.0

        rotation_cmd = np.clip(error * gain, -max_thrust, max_thrust)

        if logger:
            logger.info(
                f"Rotation: 목표={self.target_angle:.1f}°, 오차={error:.1f}°, "
                f"명령={rotation_cmd:.3f}"
            )

        # Body force 반환 (전진 없이 제자리 회전)
        desired_speed = 0.0
        desired_yaw = -rotation_cmd
        desired_force_y = 0.0

        return desired_speed, desired_yaw, desired_force_y


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
            MissionType.OBSTACLE_AVOID: ObstacleAvoidMission(thrust_scale, avoidance_controller),
            MissionType.DOCK_MODE: DockMission(thrust_scale),
            MissionType.ROTATION: RotationMission(thrust_scale)
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
            if mission_type in self.missions:
                self.missions[mission_type].reset()

    def update_thrust_scale(self, thrust_scale: float):
        """
        모든 미션의 thrust_scale 업데이트

        Args:
            thrust_scale: 새 추력 스케일
        """
        self.thrust_scale = thrust_scale
        for mission in self.missions.values():
            mission.thrust_scale = thrust_scale

    def execute_mission(
        self, mission_type: MissionType, **kwargs
    ) -> Union[Tuple[float, float], Tuple[float, float, Optional[float], Optional[float], Optional[float]]]:
        """
        미션 실행

        Args:
            mission_type: 실행할 미션 타입
            **kwargs: 미션별 파라미터

        Returns:
            Tuple[float, float]: (left_thrust, right_thrust) - 일반 미션
            또는
            Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
                (left_thrust, right_thrust, left_pos, right_pos, target_depth) - DOCK_MODE
        """
        mission = self.missions.get(mission_type)
        if mission:
            return mission.execute(**kwargs)
        return 0.0, 0.0

    def get_circle_buoy_target_x(self) -> Optional[float]:
        """
        CIRCLE_BUOY 미션의 target_x 가져오기 (시각화용)

        Returns:
            Optional[float]: target_x 값 또는 None (웨이포인트 기반 방식으로 변경되어 항상 None 반환)
        """
        # 웨이포인트 기반 방식으로 변경되어 target_x가 더 이상 없음
        return None

    def is_circle_mission_completed(self) -> bool:
        """
        CircleBuoyMission의 완료 여부 확인 (모든 웨이포인트 통과 또는 타임아웃)

        Returns:
            bool: 모든 웨이포인트를 통과했거나 타임아웃되었으면 True, 아니면 False
        """
        circle_mission = self.missions.get(MissionType.CIRCLE_BUOY)
        if circle_mission and hasattr(circle_mission, 'is_completed'):
            return circle_mission.is_completed
        return False

    def get_circle_buoy_waypoints(self) -> List[np.ndarray]:
        """
        CIRCLE_BUOY 미션의 생성된 웨이포인트 가져오기 (시각화용)

        Returns:
            List[np.ndarray]: world-frame 웨이포인트 리스트 [[x, y], ...]
        """
        circle_mission = self.missions.get(MissionType.CIRCLE_BUOY)
        if circle_mission and hasattr(circle_mission, 'get_generated_waypoints'):
            return circle_mission.get_generated_waypoints()
        return []

    def is_rotation_mission_completed(self) -> bool:
        """
        ROTATION 미션 완료 여부 확인
        """
        rotation_mission = self.missions.get(MissionType.ROTATION)
        if rotation_mission and hasattr(rotation_mission, 'is_completed'):
            return rotation_mission.is_completed
        return False

    def get_dock_accumulated_angle(self) -> Optional[float]:
        """
        DOCK_MODE 미션의 누적 각도 가져오기 (시각화용)

        Returns:
            Optional[float]: 누적 각도 값 또는 None
        """
        dock_mission = self.missions.get(MissionType.DOCK_MODE)
        if dock_mission and hasattr(dock_mission, 'accumulated_angle'):
            return dock_mission.accumulated_angle
        return None

    def is_dock_mission_completed(self) -> bool:
        """Dock 미션 완료 여부"""
        dock_mission = self.missions.get(MissionType.DOCK_MODE)
        if dock_mission and hasattr(dock_mission, 'is_completed'):
            return dock_mission.is_completed
        return False

    def get_dock_body_forces(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Dock 미션의 최근 body-force 명령 반환 (sway, yaw, surge)
        """
        dock_mission = self.missions.get(MissionType.DOCK_MODE)
        if dock_mission and hasattr(dock_mission, 'get_last_body_forces'):
            return dock_mission.get_last_body_forces()
        return None, None, None
