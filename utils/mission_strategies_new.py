#!/usr/bin/env python3
"""
미션 전략 모듈
- 4가지 미션의 제어 로직을 캡슐화
- 리팩토링: 중복 제거, 매직 넘버 상수화, 타입 힌트 보완
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Callable, Union

from .detection_system import MissionType
from .config import Constants


# ============================================================================
# 헬퍼 함수 (유틸리티)
# ============================================================================

def normalize_heading(heading: float) -> float:
    """
    헤딩을 0~360 범위로 정규화

    Args:
        heading: 헤딩 (도)

    Returns:
        float: 정규화된 헤딩 (0~360)
    """
    normalized = heading % 360
    if normalized < 0:
        normalized += 360
    return normalized


def calculate_heading_error(target_heading: float, current_heading: float) -> float:
    """
    목표 헤딩과 현재 헤딩의 최단 거리 오차 계산

    Args:
        target_heading: 목표 헤딩 (도, -180~180 또는 0~360)
        current_heading: 현재 헤딩 (도, -180~180 또는 0~360)

    Returns:
        float: 헤딩 오차 (-180~180)
    """
    # 0~360 범위로 정규화
    target = normalize_heading(target_heading)
    current = normalize_heading(current_heading)

    # 오차 계산
    error = target - current

    # -180 ~ 180 범위로 정규화
    if error > 180:
        error -= 360
    elif error < -180:
        error += 360

    return error


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

        # 마지막으로 성공한 스러스터 명령 저장 (부표 미탐지 시 사용)
        # 주의: thrust_scale로 변환된 실제 값 저장 (예: 500.0)
        self.last_known_left_cmd = 0.0
        self.last_known_right_cmd = 0.0

        # 거리 기반 명령 고정 (부표에 가까워지면 명령 고정)
        self.distance_locked = False  # 거리 임계값 도달 플래그
        # 주의: thrust_scale로 변환된 실제 값 저장 (예: 500.0)
        self.locked_left_cmd = 0.0  # 고정된 왼쪽 명령
        self.locked_right_cmd = 0.0  # 고정된 오른쪽 명령
        self.locked_left_pos = 0.0  # 고정된 왼쪽 각도
        self.locked_right_pos = 0.0  # 고정된 오른쪽 각도
        self.lock_distance_threshold = Constants.CIRCLE_LOCK_DISTANCE_THRESHOLD

        # SWAY 제어 파라미터 (부드러운 원 그리기)
        self.sway_strength = Constants.CIRCLE_SWAY_STRENGTH
        self.sway_max_angle = Constants.CIRCLE_SWAY_MAX_ANGLE

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
        self.distance_locked = False
        self.locked_left_cmd = 0.0
        self.locked_right_cmd = 0.0
        self.locked_left_pos = 0.0
        self.locked_right_pos = 0.0

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

    def calculate_thruster_allocation(
        self, sway_force: float, yaw_moment: float, surge_velocity: float
    ) -> Tuple[float, float, float, float]:
        """
        2-Motor Vectored Thruster Control Allocation (Circle 미션용)

        Args:
            sway_force: 횡방향 힘 (좌우 이동, -1~1)
            yaw_moment: 회전 모멘트 (-1~1)
            surge_velocity: 전진 속도 (0~1)

        Returns:
            Tuple[float, float, float, float]: (left_pos, left_thrust, right_pos, right_thrust)
                pos는 라디안 단위 각도
                thrust는 추력 크기
        """
        # 1. SWAY를 위한 기본 각도 (양쪽 동일)
        sway_angle = sway_force * np.radians(self.sway_max_angle)

        # 2. YAW를 위한 각도 차이 (양쪽 반대)
        yaw_angle_diff = yaw_moment * (np.pi / 6)  # 최대 ±30도

        # 3. 최종 각도 = SWAY 각도 ± YAW 각도 차이
        left_angle = sway_angle + yaw_angle_diff
        right_angle = sway_angle - yaw_angle_diff

        # 각도 제한
        left_angle = np.clip(left_angle, -np.pi / 2, np.pi / 2)
        right_angle = np.clip(right_angle, -np.pi / 2, np.pi / 2)

        # 4. Base thrust 계산 (surge_velocity 기반)
        base_thrust = surge_velocity * self.thrust_scale

        # 5. 추력 차이로 Yaw moment 추가 생성
        yaw_thrust_diff = yaw_moment * self.thrust_scale * 0.3

        # 6. 최종 추력 계산
        left_thrust = base_thrust + yaw_thrust_diff
        right_thrust = base_thrust - yaw_thrust_diff

        # 추력 제한
        max_thrust = self.thrust_scale * 1.5
        left_thrust = np.clip(left_thrust, -max_thrust, max_thrust)
        right_thrust = np.clip(right_thrust, -max_thrust, max_thrust)

        return left_angle, left_thrust, right_angle, right_thrust

    def execute(
        self,
        detected_objects: List[Dict],
        current_image: np.ndarray,
        agent_heading: float,
        mission_params: Dict,
        logger=None,
        raw_detections: Optional[List[Dict]] = None,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """
        파란색 부표 주변을 회전 (SWAY 포함)

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            current_image: 현재 카메라 이미지
            agent_heading: 로봇 헤딩 (도)
            mission_params: 미션 파라미터
            logger: 로거
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)

        Returns:
            Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos, right_pos)
        """
        # 트랙바 파라미터 업데이트 (동적 조정)
        self.base_speed = mission_params.get('circle_base_speed', self.base_speed)
        self.min_speed = mission_params.get('circle_min_speed', self.min_speed)
        self.max_turn_thrust = mission_params.get('circle_max_turn', self.max_turn_thrust)
        self.lock_distance_threshold = mission_params.get('circle_lock_distance', self.lock_distance_threshold)
        self.sway_strength = mission_params.get('circle_sway_strength', self.sway_strength)
        self.sway_max_angle = mission_params.get('circle_sway_max_angle', self.sway_max_angle)

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
            # 미션 완료 후 천천히 전진 (thruster 각도 0)
            left_thrust = Constants.CIRCLE_COMPLETION_SPEED * self.thrust_scale
            right_thrust = Constants.CIRCLE_COMPLETION_SPEED * self.thrust_scale
            return left_thrust, right_thrust, 0.0, 0.0

        # 부표 미탐지 시 이전 명령 사용
        if not blue_buoy:
            self.target_x = None
            # 거리 고정 상태이면 고정된 명령 사용, 아니면 마지막 명령 사용
            if self.distance_locked:
                left_thrust = self.locked_left_cmd
                right_thrust = self.locked_right_cmd
                left_pos = self.locked_left_pos
                right_pos = self.locked_right_pos
                if logger:
                    logger.warn(
                        f"파란색 부표 미탐지 (거리 고정 모드): 고정 명령 사용 "
                        f"L={left_thrust:.1f}, R={right_thrust:.1f}"
                    )
            else:
                left_thrust = self.last_known_left_cmd
                right_thrust = self.last_known_right_cmd
                left_pos = 0.0  # 기본값
                right_pos = 0.0
                if logger:
                    logger.warn(
                        f"파란색 부표 미탐지: 이전 명령 사용 L={left_thrust:.1f}, R={right_thrust:.1f}"
                    )
            return left_thrust, right_thrust, left_pos, right_pos

        # 미션 파라미터
        rotation_direction = mission_params.get('rotation_direction', 1)

        # 부표 측정값
        buoy_depth = blue_buoy['depth']
        buoy_x = blue_buoy['center'][0]

        # === 거리 고정 모드 체크 (이미 고정된 경우) ===
        if self.distance_locked:
            left_thrust = self.locked_left_cmd
            right_thrust = self.locked_right_cmd
            left_pos = self.locked_left_pos
            right_pos = self.locked_right_pos
            self.target_x = None  # 시각화에서 target_x 표시 안함

            # 마지막 명령도 고정된 값으로 업데이트 (부표 미탐지 시 사용)
            self.last_known_left_cmd = left_thrust
            self.last_known_right_cmd = right_thrust

            if logger:
                direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
                logger.info(
                    f"Circle [LOCKED][{data_source}]: rotation={self.total_rotation:.1f}°, "
                    f"부표 위치=({buoy_x:.1f}px), 깊이={buoy_depth:.3f}m, "
                    f"방향={direction_name}, 고정 명령 사용"
                )

            return left_thrust, right_thrust, left_pos, right_pos

        # === 일반 제어 모드 (명령 계산) ===
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

        # 스러스터를 thrust_scale로 변환
        left_thrust = (left_command / 1000.0) * self.thrust_scale
        right_thrust = (right_command / 1000.0) * self.thrust_scale
        left_pos = 0.0  # 기본 각도
        right_pos = 0.0

        # === 거리 기반 명령 고정 로직 (막 임계값을 넘었을 때) ===
        # 거리가 임계값 이하로 가까워지면 방금 계산한 명령 고정
        if not self.distance_locked and buoy_depth >= self.lock_distance_threshold:
            self.distance_locked = True
            # 방금 계산한 명령을 고정 (thrust_scale 변환된 실제 값)
            self.locked_left_cmd = left_thrust
            self.locked_right_cmd = right_thrust
            self.locked_left_pos = left_pos
            self.locked_right_pos = right_pos
            if logger:
                logger.info(
                    f"🔒 거리 임계값 도달 (depth={buoy_depth:.2f}m <= {self.lock_distance_threshold:.2f}m): "
                    f"현재 명령 고정 L={left_thrust:.1f}, R={right_thrust:.1f}"
                )

        # 마지막으로 성공한 명령 저장 (thrust_scale 변환된 실제 값 저장)
        self.last_known_left_cmd = left_thrust
        self.last_known_right_cmd = right_thrust

        if logger:
            direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
            logger.info(
                f"Circle [ROTATE][{data_source}]: rotation={self.total_rotation:.1f}°, "
                f"부표 위치=({buoy_x:.1f}px), 깊이={buoy_depth:.3f}m, "
                f"목표={target_x:.1f}px, 오차={error:.1f}px, "
                f"조향={steering_command:.3f}, 방향={direction_name}"
            )

        return left_thrust, right_thrust, left_pos, right_pos


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


class DockMission(BaseMissionStrategy):
    """미션 5: 도킹 스테이션 미션"""

    def __init__(self, thrust_scale: float = 1000.0):
        super().__init__(thrust_scale)
        # 도킹 상태 관리
        self.docking_phase = "TRACKING"  # TRACKING, APPROACHING, REVERSING
        self.phase_start_time: Optional[float] = None
        self.target_shape_label: Optional[str] = None  # 추적할 도형 라벨

        # 누적 각도 추적 (도킹 미션 시작부터의 총 각도 변화)
        self.initial_heading: Optional[float] = None  # 누적 각도 기준 (항상 0으로 고정)
        self.initial_imu_heading: Optional[float] = None  # 처음 IMU 헤딩 값 (보정 기준)
        self.accumulated_angle: float = 0.0  # 누적 각도 변화량 (부호 반대: 왼쪽=음수, 오른쪽=양수)
        self.previous_heading: Optional[float] = None  # 이전 헤딩 (각도 변화 계산용)
        self.rotation_direction: int = 1  # 현재 회전 방향 (0=왼쪽, 1=오른쪽)
        self.is_completed: bool = False

        # 제어 파라미터
        self.sway_gain = Constants.DOCK_SWAY_GAIN
        self.yaw_gain = Constants.DOCK_YAW_GAIN
        self.max_sway_thrust = Constants.DOCK_MAX_SWAY_THRUST
        self.max_yaw_thrust = Constants.DOCK_MAX_YAW_THRUST
        self.base_surge = Constants.DOCK_BASE_SURGE
        self.depth_threshold = Constants.DOCK_DEPTH_THRESHOLD
        self.sway_to_yaw_threshold = 0.5  # SWAY에서 YAW로 전환하는 depth 임계값
        self.approach_time = Constants.DOCK_APPROACH_TIME
        self.reverse_time = Constants.DOCK_REVERSE_TIME
        self.approach_speed = Constants.DOCK_APPROACH_SPEED
        self.reverse_speed = Constants.DOCK_REVERSE_SPEED

        # X 좌표 기반 SWAY 제어 파라미터
        self.center_tolerance = Constants.DOCK_CENTER_TOLERANCE  # 이미지 중앙 허용 오차 (픽셀)
        self.sway_strength = Constants.DOCK_SWAY_STRENGTH  # SWAY 강도 (0-1)

        # 최근 body-force 명령 (ROS 퍼블리시용)
        self.last_sway_force = 0.0
        self.last_yaw_moment = 0.0
        self.last_surge_velocity = 0.0

    def reset(self):
        """미션 상태 초기화"""
        self.docking_phase = "TRACKING"
        self.phase_start_time = None
        self.target_shape_label = None

        # 누적 각도 추적 초기화
        self.initial_heading = None
        self.initial_imu_heading = None
        self.accumulated_angle = 0.0
        self.previous_heading = None
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
        self.last_sway_force = float(np.clip(sway_force, -1.0, 1.0))
        self.last_yaw_moment = float(np.clip(yaw_moment, -1.0, 1.0))
        self.last_surge_velocity = float(np.clip(surge_velocity, -1.0, 1.0))

    def calculate_thruster_allocation(
        self, sway_force: float, yaw_moment: float, surge_velocity: float
    ) -> Tuple[float, float, float, float]:
        """
        2-Motor Vectored Thruster Control Allocation (역할 분리 방식)
        - 각도: Sway motion 생성 (횡방향 이동)
        - 추력 차이: Yaw moment 생성 (회전)
        → 충돌 최소화, 조화로운 제어

        Args:
            sway_force: 횡방향 힘 (좌우 이동, -1~1)
            yaw_moment: 회전 모멘트 (-1~1)
            surge_velocity: 전진 속도 (0~1)

        Returns:
            Tuple[float, float, float, float]: (left_pos, left_thrust, right_pos, right_pos)
                pos는 라디안 단위 각도
                thrust는 추력 크기
        """
        # === 각도와 추력 차이 동시 사용 제어 ===
        # 각도: SWAY + YAW 복합 제어
        # 추력 차이: YAW 추가 강화

        # 1. SWAY를 위한 기본 각도 (양쪽 동일)
        sway_angle = sway_force * (np.pi / 3)  # 최대 ±60도

        # 2. YAW를 위한 각도 차이 (양쪽 반대)
        yaw_angle_diff = yaw_moment * (np.pi / 6)  # 최대 ±30도

        # 3. 최종 각도 = SWAY 각도 ± YAW 각도 차이
        left_angle = sway_angle + yaw_angle_diff
        right_angle = sway_angle - yaw_angle_diff

        # 각도 제한
        left_angle = np.clip(left_angle, -np.pi / 2, np.pi / 2)
        right_angle = np.clip(right_angle, -np.pi / 2, np.pi / 2)

        # 2. Base thrust 계산 (surge_velocity는 이미 sway_force를 고려하여 계산됨)
        base_thrust = surge_velocity * self.thrust_scale

        # 3. Sway 전용 추가 추력 (surge가 거의 0이고 sway가 있을 때만)
        # Sway는 주로 각도로 제어하되, 필요시 최소 추가 추력 제공
        additional_thrust = 0.0
        if abs(sway_force) > 0.01 and abs(surge_velocity) < 0.01:
            # SWAY_ONLY 모드: 횡방향 이동을 위한 최소 추가 추력
            additional_thrust = abs(sway_force) * self.thrust_scale * 0.2
        # 4. 추력 차이로 Yaw moment 추가 생성
        yaw_thrust_diff = yaw_moment * self.thrust_scale * 0.5

        # 5. 최종 추력 계산 (base_thrust는 surge_velocity에 직접 비례, 추가 추력 포함)
        left_thrust = base_thrust + additional_thrust + yaw_thrust_diff
        right_thrust = base_thrust + additional_thrust - yaw_thrust_diff

        # 추력 제한 (음수 허용: 후진 가능)
        max_thrust = self.thrust_scale * 1.5
        left_thrust = np.clip(left_thrust, -max_thrust, max_thrust)
        right_thrust = np.clip(right_thrust, -max_thrust, max_thrust)

        return left_angle, left_thrust, right_angle, right_thrust

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

        # 1. SWAY 제어: X 좌표 기반 횡방향 이동 (중앙 허용 범위 적용)
        if effective_error_x != 0.0:
            sway_force = sway_direction * self.sway_strength
            sway_force = np.clip(sway_force, -1.0, 1.0)
        else:
            sway_force = 0.0

        # Accumulated angle 피드백 추가
        # accumulated_angle < 0 (음수, 왼쪽 회전 누적) → SWAY를 오른쪽으로 (+)
        # accumulated_angle > 0 (양수, 오른쪽 회전 누적) → SWAY를 왼쪽으로 (-)
        angle_feedback_gain = 0.04  # 게인 (조정 가능)
        angle_feedback = self.accumulated_angle * angle_feedback_gain
        sway_force += angle_feedback
        sway_force = np.clip(sway_force, -1.0, 1.0)

        # 2. Moment(YAW) 제어: error_x 기반 회전 (객체 x좌표를 이미지 중심으로)
        # error_x > 0: 객체가 오른쪽에 있음 → 오른쪽으로 회전 (양수 moment)
        # error_x < 0: 객체가 왼쪽에 있음 → 왼쪽으로 회전 (음수 moment)
        yaw_moment = normalized_error_x * self.yaw_gain
        yaw_moment = np.clip(yaw_moment, -1.0, 1.0)

        # 3. SURGE: 전진 속도 (에러가 작을수록 빠르게 전진)
        surge_velocity = self.base_surge * (1.0 - 6*abs(sway_force))
        surge_velocity = max(-0.0000, surge_velocity)

        control_mode = "SWAY_YAW_CONTROL"

        # 2-Motor Vectored Thruster Allocation
        left_pos, left_thrust, right_pos, right_thrust = self.calculate_thruster_allocation(
            sway_force, yaw_moment, surge_velocity
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
    """미션 6: 제자리 선회"""

    def __init__(self, thrust_scale: float = 1000.0):
        super().__init__(thrust_scale)
        self.target_angle: Optional[float] = None
        self.is_completed: bool = False
        self.stable_frames: int = 0

    def reset(self):
        self.target_angle = None
        self.is_completed = False
        self.stable_frames = 0

    def execute(
        self,
        agent_heading: Optional[float],
        mission_params: Optional[Dict],
        logger=None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        제자리에서 목표 각도까지 선회

        Args:
            agent_heading: 현재 헤딩 (deg)
            mission_params: {'desired_angle': float, ...}
        """
        if agent_heading is None or mission_params is None:
            return 0.0, 0.0

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
            return 0.0, 0.0

        rotation_cmd = np.clip(error * gain, -max_thrust, max_thrust)
        left_thrust = -rotation_cmd * self.thrust_scale
        right_thrust = rotation_cmd * self.thrust_scale

        if logger:
            logger.info(
                f"Rotation: 목표={self.target_angle:.1f}°, 오차={error:.1f}°, "
                f"명령={rotation_cmd:.3f}"
            )

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
