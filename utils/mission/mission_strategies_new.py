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
    """미션 2: 원 궤적 그리기 - 로봇 정면에 포인트 생성"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        # 상태 변수
        self.mission_start_time: Optional[float] = None
        self.is_completed = False
        self.target_points: List[np.ndarray] = []  # 생성된 목표 포인트 리스트 [[Easting, Northing], ...]
        self.current_waypoint_index: int = 0  # 현재 추종 중인 웨이포인트 인덱스
        self.initial_position: Optional[np.ndarray] = None  # 미션 시작 위치

    def reset(self):
        """미션 상태 초기화"""
        self.mission_start_time = None
        self.is_completed = False
        self.target_points = []
        self.current_waypoint_index = 0
        self.initial_position = None

    def _calculate_point_at_angle(self, robot_pos: np.ndarray, robot_heading: float, 
                                   distance: float, angle_offset: float) -> np.ndarray:
        """
        로봇 기준 특정 각도 방향에 포인트 생성

        Args:
            robot_pos: 로봇 현재 위치 [Easting, Northing]
            robot_heading: 로봇 헤딩 (도, 0=North, 90=East)
            distance: 거리 (미터)
            angle_offset: 헤딩 기준 각도 오프셋 (도, 양수=시계방향, 음수=반시계방향)

        Returns:
            np.ndarray: 목표 포인트 [Northing, Easting]
        """
        # 헤딩과 오프셋을 라디안으로 변환
        heading_rad = np.radians(robot_heading)
        offset_rad = np.radians(angle_offset)
        total_angle_rad = heading_rad + offset_rad
        
        # NED 좌표계: 0도=North, 90도=East
        # Easting = robot_pos[0] + distance * sin(heading + offset)
        # Northing = robot_pos[1] + distance * cos(heading + offset)
        easting = robot_pos[0] + distance * np.sin(total_angle_rad)
        northing = robot_pos[1] + distance * np.cos(total_angle_rad)
        
        return np.array([northing, easting], dtype=np.float32)

    def _calculate_forward_point(self, robot_pos: np.ndarray, robot_heading: float, length: float) -> np.ndarray:
        """
        로봇 정면 방향에 포인트 생성 (호환성 유지)

        Args:
            robot_pos: 로봇 현재 위치 [Easting, Northing]
            robot_heading: 로봇 헤딩 (도, 0=North, 90=East)
            length: 로봇 정면으로부터의 거리 (미터)

        Returns:
            np.ndarray: 목표 포인트 [Northing, Easting]
        """
        return self._calculate_point_at_angle(robot_pos, robot_heading, length, 0.0)

    def get_target_points(self) -> List[np.ndarray]:
        """
        생성된 목표 포인트 리스트 반환 (시각화용)

        Returns:
            List[np.ndarray]: 목표 포인트 리스트 [[Northing, Easting], ...]
        """
        return self.target_points

    def get_target_point(self) -> Optional[np.ndarray]:
        """
        생성된 첫 번째 목표 포인트 반환 (호환성 유지)

        Returns:
            Optional[np.ndarray]: 첫 번째 목표 포인트 [Northing, Easting] 또는 None
        """
        if len(self.target_points) > 0:
            return self.target_points[0]
        return None

    def execute(
        self,
        agent_position: Optional[np.ndarray],
        agent_heading: Optional[float],
        mission_params: Dict,
        logger=None,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """
        원 궤적 그리기 미션 - 로봇 정면에 포인트 생성

        Args:
            agent_position: 로봇 현재 위치 [Easting, Northing]
            agent_heading: 로봇 헤딩 (도, 0=North, 90=East)
            mission_params: 미션 파라미터 (length 필수)
            logger: 로거

        Returns:
            Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos, right_pos)
        """
        # 위치/헤딩 정보가 없으면 정지
        if agent_position is None or agent_heading is None:
            if logger:
                logger.warn("CIRCLE_BUOY: 위치/헤딩 정보 없음 - 정지")
            return 0.0, 0.0, 0.0, 0.0

        # length 파라미터 확인 (음수 허용: 뒤로 가는 경우)
        length = mission_params.get('length', 20.0)
        if length == 0:
            if logger:
                logger.warn(f"CIRCLE_BUOY: length 값이 0입니다 ({length}) - 정지")
            return 0.0, 0.0, 0.0, 0.0

        # angle 파라미터 확인 (기본값: 0도, angle이 없으면 정면만)
        angle = mission_params.get('angle', 0.0)
        side_distance = length / 1.4  # 양쪽 포인트 거리
        turn_flag = mission_params.get('turn_flag', 0)  # 0=시계방향, 1=반시계방향
        radius = mission_params.get('radius', 2.0)  # 웨이포인트 도달 판정 반경 (미터)
        waypoints_param = mission_params.get('waypoints', None)  # 직접 좌표 입력 (GPS 모드용)

        # 미션 시작 시 목표 포인트 생성
        if self.mission_start_time is None:
            self.mission_start_time = time.time()
            self.initial_position = agent_position.copy()
            self.target_points = []
            self.current_waypoint_index = 0
            
            # waypoints 파라미터가 있으면 직접 좌표 사용 (GPS 모드)
            if waypoints_param and len(waypoints_param) > 0:
                # waypoints_param: [[Easting, Northing], ...] 또는 [[lat, lon], ...]
                # GPS 모드인 경우 waypoint_manager에서 이미 로컬 좌표로 변환되어 전달됨
                # 따라서 [Easting, Northing] 형식으로 가정
                for wp in waypoints_param:
                    if len(wp) >= 2:
                        # [Easting, Northing] 형식으로 변환
                        point = np.array([float(wp[0]), float(wp[1])], dtype=np.float32)
                        self.target_points.append(point)
                
                if logger:
                    logger.info(f"📍 CIRCLE_BUOY: 직접 좌표 사용 ({len(self.target_points)}개 웨이포인트)")
            else:
                # 자동 생성 (기존 로직)
                if angle != 0.0:
                    # +angle, -angle 포인트 생성
                    right_point = self._calculate_point_at_angle(agent_position, agent_heading, side_distance, angle)
                    left_point = self._calculate_point_at_angle(agent_position, agent_heading, side_distance, -angle)
                    front_point = self._calculate_point_at_angle(agent_position, agent_heading, length, 0.0)
                    start_point = np.array([agent_position[0], agent_position[1]], dtype=np.float32)  # 현재 위치
                    
                    # turn_flag에 따라 순서 결정
                    if turn_flag == 0:  # 시계방향: +angle → 정면 → -angle → 시작위치
                        self.target_points = [right_point, front_point, left_point, start_point]
                    else:  # 반시계방향: -angle → 정면 → +angle → 시작위치
                        self.target_points = [left_point, front_point, right_point, start_point]
                else:
                    # angle이 0이면 정면만
                    front_point = self._calculate_point_at_angle(agent_position, agent_heading, length, 0.0)
                    start_point = np.array([agent_position[0], agent_position[1]], dtype=np.float32)
                    self.target_points = [front_point, start_point]
            
            if logger:
                direction_str = "시계방향" if turn_flag == 0 else "반시계방향"
                logger.info(
                    f"🎯 CIRCLE_BUOY 미션 시작: 위치=({agent_position[0]:.2f}, {agent_position[1]:.2f}), "
                    f"헤딩={agent_heading:.1f}°, length={length:.1f}m, angle={angle:.1f}°, "
                    f"방향={direction_str}, radius={radius:.1f}m"
                )
                for i, wp in enumerate(self.target_points):
                    logger.info(
                        f"📍 웨이포인트 {i}: ({wp[0]:.2f}, {wp[1]:.2f})"
                    )

        # 목표 포인트가 없으면 재생성
        if len(self.target_points) == 0:
            if angle != 0.0:
                right_point = self._calculate_point_at_angle(agent_position, agent_heading, side_distance, angle)
                left_point = self._calculate_point_at_angle(agent_position, agent_heading, side_distance, -angle)
                front_point = self._calculate_point_at_angle(agent_position, agent_heading, length, 0.0)
                start_point = np.array([agent_position[0], agent_position[1]], dtype=np.float32)
                
                if turn_flag == 0:
                    self.target_points = [right_point, front_point, left_point, start_point]
                else:
                    self.target_points = [left_point, front_point, right_point, start_point]
            else:
                front_point = self._calculate_point_at_angle(agent_position, agent_heading, length, 0.0)
                start_point = np.array([agent_position[0], agent_position[1]], dtype=np.float32)
                self.target_points = [front_point, start_point]

        # 현재 목표 웨이포인트
        if self.current_waypoint_index >= len(self.target_points):
            # 모든 웨이포인트 완료
            self.is_completed = True
            if logger:
                logger.info("✅ CIRCLE_BUOY 미션 완료: 모든 웨이포인트 통과")
            return 0.0, 0.0, 0.0, 0.0

        target_point = self.target_points[self.current_waypoint_index]
        
        # 목표 포인트까지의 거리 계산
        # agent_position: [Easting, Northing], target_point: [Easting, Northing]
        delta = target_point - agent_position
        distance = np.linalg.norm(delta)

        # 웨이포인트 도달 판정 (radius 이내)
        if distance <= radius:
            # 마지막 웨이포인트인지 확인
            is_last_waypoint = (self.current_waypoint_index == len(self.target_points) - 1)
            
            if logger:
                if is_last_waypoint:
                    logger.info(
                        f"✓ 마지막 웨이포인트 {self.current_waypoint_index} 도착: "
                        f"({target_point[0]:.2f}, {target_point[1]:.2f}) - 미션 종료"
                    )
                else:
                    logger.info(
                        f"✓ 웨이포인트 {self.current_waypoint_index} 도착: "
                        f"({target_point[0]:.2f}, {target_point[1]:.2f}), "
                        f"다음 웨이포인트: {self.current_waypoint_index + 1}/{len(self.target_points)}"
                    )
            
            # 마지막 웨이포인트 도달 시 미션 종료
            if is_last_waypoint:
                self.is_completed = True
                if logger:
                    logger.info("✅ CIRCLE_BUOY 미션 완료: 마지막 웨이포인트 도달")
                return 0.0, 0.0, 0.0, 0.0
            
            # 다음 웨이포인트로 전환
            self.current_waypoint_index += 1
            
            # 다음 웨이포인트로 업데이트
            target_point = self.target_points[self.current_waypoint_index]
            delta = target_point - agent_position
            distance = np.linalg.norm(delta)

        # atan2 방식으로 목표 헤딩 계산
        # delta: [Easting, Northing]
        # atan2(Easting, Northing) = 목표 방향 (0도=North, 90도=East)
        target_heading = np.degrees(np.arctan2(delta[0], delta[1]))
        if target_heading < 0:
            target_heading += 360

        # 헤딩 오차 계산
        heading_error = calculate_heading_error(target_heading, agent_heading)

        # atan 방식 제어: 헤딩 오차 기반 조향
        max_steering = 0.5
        steering = np.clip(np.radians(heading_error) * 0.01, -max_steering, max_steering)
        forward_speed = 0.4 if distance > 1.0 else 0.0  # 1m 이내면 정지

        # Body force 명령 계산
        desired_speed = forward_speed
        desired_yaw = steering
        desired_force_y = 0.0

        # Thruster 명령으로 변환
        left_thrust, right_thrust, left_pos, right_pos = body_forces_to_thruster_commands(
            desired_speed, desired_yaw, desired_force_y, self.thrust_scale
        )

        if logger and self.mission_start_time is not None:
            elapsed_time = time.time() - self.mission_start_time
            if int(elapsed_time * 10) % 10 == 0:  # 1초마다 로그
                logger.info(
                    f"CIRCLE_BUOY [WP{self.current_waypoint_index}/{len(self.target_points)-1}]: "
                    f"목표=({target_point[0]:.2f}, {target_point[1]:.2f}), "
                    f"거리={distance:.2f}m, 헤딩오차={heading_error:.1f}°, "
                    f"조향={steering:.3f}"
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
    """미션 5: 도킹 스테이션 미션 (좌표 기반 LOS 가이던스)"""

    def __init__(self, thrust_scale: float = Constants.DOCK_DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        # 상태 변수
        self.docking_phase = "APPROACHING"  # APPROACHING -> REVERSING -> COMPLETED
        self.phase_start_time: Optional[float] = None
        self.is_completed = False
        
        # 도킹 포인트 정보
        self.dock_index: Optional[int] = None  # 1, 2, 또는 3
        self.dock_points: Optional[List[List[List[float]]]] = None  # [[[dock_point], [aux_point]], ...]
        self.dock_point_global: Optional[np.ndarray] = None  # 전역 좌표 도킹 포인트 [Easting, Northing]
        self.aux_point_global: Optional[np.ndarray] = None  # 전역 좌표 보조 포인트 [Easting, Northing]
        self.initial_position: Optional[np.ndarray] = None  # 미션 시작 위치 [Easting, Northing]
        
        # LOS 가이던스
        self.los_guidance = LOSGuidance()
        
        # 제어 파라미터
        self.approach_time = Constants.DOCK_APPROACH_TIME
        self.reverse_time = Constants.DOCK_REVERSE_TIME
        self.approach_speed = Constants.DOCK_APPROACH_SPEED
        self.reverse_speed = Constants.DOCK_REVERSE_SPEED
        self.dock_reach_radius = 3.0  # 도킹 포인트 도달 반경 (미터)
        
        # Body-force 명령 저장 (ROS 퍼블리시용)
        self.last_sway_force = self.last_yaw_moment = self.last_surge_velocity = 0.0

    def reset(self):
        """미션 상태 초기화"""
        self.docking_phase = "APPROACHING"
        self.phase_start_time = None
        self.is_completed = False
        self.dock_index = None
        self.dock_points = None
        self.dock_point_global = None
        self.aux_point_global = None
        self.initial_position = None

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
        """
        heading_rad = np.radians(robot_heading)
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)

        easting = robot_pos[0] + (body_x * sin_h + body_y * cos_h)
        northing = robot_pos[1] + (body_x * cos_h - body_y * sin_h)

        return np.array([easting, northing], dtype=np.float32)

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
        agent_position: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
        """
        도킹 스테이션 미션 실행 (좌표 기반 LOS 가이던스)

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력) - 사용하지 않음
            current_image: 현재 카메라 이미지 - 사용하지 않음
            logger: 로거
            raw_detections: 원본 탐지 결과 - 사용하지 않음
            mission_params: 미션 파라미터 {'dock_index': int, 'dock_points': [[[dock_point], [aux_point]], ...]}
            agent_heading: 로봇 헤딩 (도, 0=North, 90=East)
            agent_position: 로봇 현재 위치 [Easting, Northing] (world-frame)

        Returns:
            Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
                (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        # 필수 파라미터 확인
        if agent_position is None or agent_heading is None:
            if logger:
                logger.warn("DOCK_MODE: agent_position 또는 agent_heading이 없습니다. 정지")
            self._set_body_forces(0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, None

        # 미션 파라미터 처리
        if mission_params:
            # 도킹 인덱스 및 포인트 설정 (처음 한 번만)
            if 'dock_index' in mission_params and 'dock_points' in mission_params:
                new_dock_index = mission_params['dock_index']
                new_dock_points = mission_params['dock_points']
                
                # 인덱스 또는 포인트가 변경된 경우에만 업데이트
                if (self.dock_index != new_dock_index or 
                    self.dock_points != new_dock_points or
                    self.initial_position is None):
                    self.dock_index = new_dock_index
                    self.dock_points = new_dock_points
                    self.initial_position = agent_position.copy()
                    
                    # 도킹 인덱스 유효성 확인 (1-6, 보험 포함)
                    if self.dock_index < 1 or self.dock_index > 6:
                        if logger:
                            logger.error(f"DOCK_MODE: 잘못된 dock_index={self.dock_index} (1-6 범위여야 함)")
                        self._set_body_forces(0.0, 0.0, 0.0)
                        return 0.0, 0.0, 0.0, 0.0, None
                    
                    # 도킹 포인트 리스트 확인 (최소 3개, 최대 6개)
                    if (not self.dock_points or 
                        len(self.dock_points) < 3 or
                        len(self.dock_points) > 6 or
                        self.dock_index > len(self.dock_points)):
                        if logger:
                            logger.error(f"DOCK_MODE: dock_points가 올바르지 않습니다. (3-6개 필요, 현재 {len(self.dock_points) if self.dock_points else 0}개)")
                        self._set_body_forces(0.0, 0.0, 0.0)
                        return 0.0, 0.0, 0.0, 0.0, None
                    
                    # 선택된 도킹 스테이션의 포인트 가져오기 (인덱스는 1부터 시작)
                    selected_dock = self.dock_points[self.dock_index - 1]
                    if len(selected_dock) < 2:
                        if logger:
                            logger.error(f"DOCK_MODE: 도킹 스테이션 {self.dock_index}의 포인트가 올바르지 않습니다.")
                        self._set_body_forces(0.0, 0.0, 0.0)
                        return 0.0, 0.0, 0.0, 0.0, None
                    
                    # 상대 좌표를 전역 좌표로 변환
                    dock_point_rel = np.array(selected_dock[0], dtype=np.float32)  # [Easting, Northing] 상대 좌표
                    aux_point_rel = np.array(selected_dock[1], dtype=np.float32)  # [Easting, Northing] 상대 좌표
                    
                    # body-frame → world-frame 변환 (미션 시작 위치와 헤딩 기준)
                    # 상대 좌표는 미션 시작 위치를 기준으로 함
                    self.dock_point_global = self.initial_position + dock_point_rel
                    self.aux_point_global = self.initial_position + aux_point_rel
                    
                    if logger:
                        logger.info(
                            f"🎯 도킹 미션 시작: 스테이션 {self.dock_index}, "
                            f"도킹 포인트=[{self.dock_point_global[0]:.2f}, {self.dock_point_global[1]:.2f}], "
                            f"보조 포인트=[{self.aux_point_global[0]:.2f}, {self.aux_point_global[1]:.2f}]"
                        )
            
            # 제어 파라미터 업데이트
            self.approach_time = mission_params.get('dock_approach_time', self.approach_time)
            self.reverse_time = mission_params.get('dock_reverse_time', self.reverse_time)
            self.approach_speed = mission_params.get('dock_approach_speed', self.approach_speed)
            self.reverse_speed = mission_params.get('dock_reverse_speed', self.reverse_speed)
            self.dock_reach_radius = mission_params.get('dock_reach_radius', self.dock_reach_radius)

        # 도킹 포인트가 설정되지 않은 경우
        if self.dock_point_global is None or self.aux_point_global is None:
            if logger:
                logger.warn("DOCK_MODE: 도킹 포인트가 설정되지 않았습니다. 정지")
            self._set_body_forces(0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, None

        # 도킹 단계별 처리
        if self.docking_phase == "APPROACHING":
            return self._execute_approaching_phase(agent_position, agent_heading, logger)
        elif self.docking_phase == "REVERSING":
            return self._execute_reversing_phase(logger)
        else:
            # 미션 완료
            return 0.0, 0.0, 0.0, 0.0, None

    def _execute_approaching_phase(
        self, agent_position: np.ndarray, agent_heading: float, logger
    ) -> Tuple[float, float, float, float, Optional[float]]:
        """
        접근 단계: LOS 가이던스를 사용하여 도킹 포인트로 접근

        Args:
            agent_position: 로봇 현재 위치 [Easting, Northing]
            agent_heading: 로봇 현재 헤딩 (도, 0=North, 90=East)
            logger: 로거

        Returns:
            Tuple: (left_thrust, right_thrust, left_pos, right_pos, target_depth)
        """
        # 도킹 포인트까지의 거리 계산
        distance_to_dock = np.linalg.norm(agent_position - self.dock_point_global)
        
        # 도킹 포인트 도달 확인
        if distance_to_dock < self.dock_reach_radius:
            # 도킹 포인트 도달, 후진 단계로 전환
            self.docking_phase = "REVERSING"
            self.phase_start_time = time.time()
            if logger:
                logger.info(f"🛑 도킹 포인트 도달 (거리={distance_to_dock:.2f}m)! REVERSING 단계 시작")
            self._set_body_forces(0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, None

        # LOS 가이던스: 보조 포인트를 경유하여 도킹 포인트로 이동
        # 경로: 현재 위치 → 보조 포인트 → 도킹 포인트
        # 먼저 보조 포인트로 이동
        distance_to_aux = np.linalg.norm(agent_position - self.aux_point_global)
        
        if distance_to_aux > self.dock_reach_radius:
            # 보조 포인트로 LOS 가이던스
            los_target = self.los_guidance.calculate_los_point(
                agent_position, agent_position, self.aux_point_global
            )
            target_point = los_target
            target_name = "보조 포인트"
        else:
            # 보조 포인트 도달, 도킹 포인트로 LOS 가이던스
            los_target = self.los_guidance.calculate_los_point(
                agent_position, self.aux_point_global, self.dock_point_global
            )
            target_point = los_target
            target_name = "도킹 포인트"

        # 목표 방향 계산 (도)
        delta = target_point - agent_position
        target_heading_deg = np.degrees(np.arctan2(delta[0], delta[1]))  # atan2(Easting, Northing)
        
        # 헤딩 오차 계산 (도)
        heading_error = calculate_heading_error(agent_heading, target_heading_deg)
        heading_error_rad = np.radians(heading_error)
        
        # 제어 계산
        # 선속도: 거리에 따라 조절
        max_speed = self.approach_speed
        distance_factor = min(1.0, distance_to_dock / 10.0)  # 10m 이내에서 감소
        linear_velocity = max_speed * distance_factor
        
        # 각속도: 헤딩 오차에 비례
        angular_gain = 0.5
        angular_velocity = angular_gain * heading_error_rad
        angular_velocity = np.clip(angular_velocity, -1.0, 1.0)
        
        # Body forces로 변환
        # 간단한 변환: linear_velocity → surge, angular_velocity → yaw
        surge_velocity = linear_velocity
        yaw_moment = angular_velocity * 0.3  # 게인 조정
        sway_force = 0.0  # LOS 가이던스에서는 sway 사용 안함
        
        # 2-Motor Vectored Thruster Allocation
        left_thrust, right_thrust, left_pos, right_pos = body_forces_to_thruster_commands(
            surge_velocity, yaw_moment, sway_force, self.thrust_scale, use_vectored_thrusters=True
        )
        self._set_body_forces(sway_force, yaw_moment, surge_velocity)
        
        if logger:
            logger.info(
                f"Dock APPROACHING [{target_name}]: "
                f"거리={distance_to_dock:.2f}m, 헤딩오차={heading_error:.1f}°, "
                f"linear={linear_velocity:.3f}, angular={angular_velocity:.3f}, "
                f"L={left_thrust:.1f}, R={right_thrust:.1f}"
            )
        
        return left_thrust, right_thrust, left_pos, right_pos, None

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


class StopMission(BaseMissionStrategy):
    """미션 7: 정지 미션 (지정된 시간 동안 정지)"""

    def __init__(self, thrust_scale: float = Constants.DEFAULT_THRUST_SCALE):
        super().__init__(thrust_scale)
        self.stop_start_time: Optional[float] = None
        self.stop_duration: float = Constants.STOP_DEFAULT_DURATION
        self.is_completed: bool = False

    def reset(self):
        """미션 상태 초기화"""
        self.stop_start_time = None
        self.stop_duration = Constants.STOP_DEFAULT_DURATION
        self.is_completed = False

    def execute_body_forces(
        self,
        mission_params: Optional[Dict],
        logger=None,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        지정된 시간 동안 정지

        Args:
            mission_params: {'stop_duration': float} - 정지 시간 (초)
            logger: 로거

        Returns:
            Tuple[float, float, float]: (desired_speed, desired_yaw, desired_force_y)
        """
        # 정지 시작 시간 기록
        if self.stop_start_time is None:
            import time
            self.stop_start_time = time.time()
            self.is_completed = False
            
            # 파라미터에서 정지 시간 가져오기
            if mission_params:
                self.stop_duration = mission_params.get('stop_duration', Constants.STOP_DEFAULT_DURATION)
            
            if logger:
                logger.info(f"🛑 STOP 미션 시작: {self.stop_duration:.1f}초 정지")

        # 경과 시간 확인
        import time
        elapsed_time = time.time() - self.stop_start_time

        if elapsed_time >= self.stop_duration:
            if not self.is_completed:
                self.is_completed = True
                if logger:
                    logger.info(f"✅ STOP 미션 완료: {elapsed_time:.1f}초 경과")
            # 정지 완료 후에도 정지 유지
            return 0.0, 0.0, 0.0

        # 정지 중 - 모든 속도를 0으로 유지
        if logger and int(elapsed_time * 10) % 10 == 0:  # 1초마다 로그
            remaining = self.stop_duration - elapsed_time
            logger.info(f"⏸️ STOP 미션 진행 중: {elapsed_time:.1f}초 / {self.stop_duration:.1f}초 (남은 시간: {remaining:.1f}초)")

        # 정지 명령 (모든 속도 0) - 명시적으로 0 반환
        desired_speed = 0.0
        desired_yaw = 0.0
        desired_force_y = 0.0
        return desired_speed, desired_yaw, desired_force_y

    def is_mission_completed(self) -> bool:
        """미션 완료 여부 반환"""
        return self.is_completed


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
            MissionType.ROTATION: RotationMission(thrust_scale),
            MissionType.STOP: StopMission(thrust_scale)
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

    def get_circle_buoy_waypoints(self) -> List[np.ndarray]:
        """
        CIRCLE_BUOY 미션의 생성된 목표 포인트 가져오기 (시각화용)

        Returns:
            List[np.ndarray]: 목표 포인트 리스트 [[Northing, Easting], ...]
        """
        circle_mission = self.missions.get(MissionType.CIRCLE_BUOY)
        if circle_mission and hasattr(circle_mission, 'get_target_points'):
            target_points = circle_mission.get_target_points()
            if target_points and len(target_points) > 0:
                return target_points
        return []

    def get_circle_buoy_current_waypoint(self) -> Optional[np.ndarray]:
        """
        CIRCLE_BUOY 미션의 현재 목표 포인트 가져오기 (시각화용)

        Returns:
            Optional[np.ndarray]: 현재 목표 포인트 [Easting, Northing] 또는 None
        """
        circle_mission = self.missions.get(MissionType.CIRCLE_BUOY)
        if circle_mission and hasattr(circle_mission, 'target_points') and hasattr(circle_mission, 'current_waypoint_index'):
            if (circle_mission.current_waypoint_index < len(circle_mission.target_points) and 
                len(circle_mission.target_points) > 0):
                return circle_mission.target_points[circle_mission.current_waypoint_index]
        return None

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

    def is_stop_mission_completed(self) -> bool:
        """
        정지 미션 완료 여부 반환

        Returns:
            bool: 정지 미션이 완료되었는지
        """
        stop_mission = self.missions.get(MissionType.STOP)
        if stop_mission and hasattr(stop_mission, 'is_mission_completed'):
            return stop_mission.is_mission_completed()
        return False

    def get_dock_body_forces(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Dock 미션의 최근 body-force 명령 반환 (sway, yaw, surge)
        """
        dock_mission = self.missions.get(MissionType.DOCK_MODE)
        if dock_mission and hasattr(dock_mission, 'get_last_body_forces'):
            return dock_mission.get_last_body_forces()
        return None, None, None
