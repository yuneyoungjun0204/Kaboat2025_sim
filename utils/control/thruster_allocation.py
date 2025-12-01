#!/usr/bin/env python3
"""
Thruster Allocation Module
- 통일된 body force 명령을 thruster 명령으로 변환
- 2-Motor Vectored Thruster Control Allocation

Example:
    >>> from utils.thruster_allocation import body_forces_to_thruster_commands
    >>> left, right, left_pos, right_pos = body_forces_to_thruster_commands(
    ...     desired_speed=0.5,
    ...     desired_yaw=0.2,
    ...     desired_force_y=0.0
    ... )
"""

import numpy as np
from typing import Tuple
from ..core.config import Constants


def body_forces_to_thruster_commands(
    desired_speed: float,
    desired_yaw: float,
    desired_force_y: float = 0.0,
    thrust_scale: float = Constants.DEFAULT_THRUST_SCALE,
    use_vectored_thrusters: bool = True
) -> Tuple[float, float, float, float]:
    """
    Body force 명령을 thruster 명령으로 변환 (통일된 인터페이스)

    Args:
        desired_speed: 전진 속도 (-1 ~ 1, surge velocity)
        desired_yaw: 회전 모멘트 (-1 ~ 1, yaw moment)
        desired_force_y: 횡방향 힘 (-1 ~ 1, sway force)
        thrust_scale: 추력 스케일 (기본값: 1500.0)
        use_vectored_thrusters: True면 벡터 추진기 사용, False면 차동 구동

    Returns:
        Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos, right_pos)
            - left_thrust, right_thrust: 추력 크기
            - left_pos, right_pos: 추진기 각도 (라디안)
    """
    # 입력값 클리핑
    desired_speed = np.clip(desired_speed, -1.0, 1.0)
    desired_yaw = np.clip(desired_yaw, -1.0, 1.0)
    desired_force_y = np.clip(desired_force_y, -1.0, 1.0)

    if use_vectored_thrusters and abs(desired_force_y) > 0.01:
        # 벡터 추진기 모드: SWAY + YAW + SURGE
        return _vectored_thruster_allocation(
            desired_speed, desired_yaw, desired_force_y, thrust_scale
        )
    else:
        # 차동 구동 모드: SURGE + YAW만 사용
        return _differential_drive_allocation(
            desired_speed, desired_yaw, thrust_scale
        )


def _vectored_thruster_allocation(
    surge_velocity: float,
    yaw_moment: float,
    sway_force: float,
    thrust_scale: float
) -> Tuple[float, float, float, float]:
    """
    2-Motor Vectored Thruster Control Allocation
    - 각도로 SWAY motion 생성 (횡방향 이동)
    - 추력 차이로 YAW moment 생성 (회전)

    Args:
        surge_velocity: 전진 속도 (-1~1)
        yaw_moment: 회전 모멘트 (-1~1)
        sway_force: 횡방향 힘 (-1~1)
        thrust_scale: 추력 스케일

    Returns:
        Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos, right_pos)
    """
    # 1. SWAY를 위한 기본 각도 (양쪽 동일)
    sway_angle = sway_force * np.radians(Constants.DOCK_SWAY_MAX_ANGLE)

    # 2. YAW를 위한 각도 차이 (양쪽 반대)
    yaw_angle_diff = yaw_moment * np.radians(Constants.DOCK_YAW_MAX_ANGLE_DIFF)

    # 3. 최종 각도 = SWAY 각도 ± YAW 각도 차이
    left_angle = sway_angle + yaw_angle_diff
    right_angle = sway_angle - yaw_angle_diff

    # 각도 제한
    left_angle = np.clip(left_angle, -np.pi / 2, np.pi / 2)
    right_angle = np.clip(right_angle, -np.pi / 2, np.pi / 2)

    # 4. Base thrust 계산 (surge_velocity 기반)
    base_thrust = surge_velocity * thrust_scale

    # 5. SWAY 전용 추가 추력 (surge가 거의 0이고 sway가 있을 때만)
    additional_thrust = 0.0
    if abs(sway_force) > Constants.DOCK_SWAY_FORCE_THRESHOLD and \
       abs(surge_velocity) < Constants.DOCK_SURGE_VELOCITY_THRESHOLD:
        additional_thrust = abs(sway_force) * thrust_scale * Constants.DOCK_SWAY_ONLY_THRUST_COEFF

    # 6. 추력 차이로 Yaw moment 추가 생성
    yaw_thrust_diff = yaw_moment * thrust_scale * Constants.DOCK_YAW_THRUST_DIFF_COEFF

    # 7. 최종 추력 계산
    left_thrust = base_thrust + additional_thrust + yaw_thrust_diff
    right_thrust = base_thrust + additional_thrust - yaw_thrust_diff

    # 추력 제한 (음수 허용: 후진 가능)
    max_thrust = thrust_scale * Constants.DOCK_MAX_THRUST_LIMIT_COEFF
    left_thrust = np.clip(left_thrust, -max_thrust, max_thrust)
    right_thrust = np.clip(right_thrust, -max_thrust, max_thrust)

    return left_thrust, right_thrust, left_angle, right_angle


def _differential_drive_allocation(
    linear_velocity: float,
    angular_velocity: float,
    thrust_scale: float
) -> Tuple[float, float, float, float]:
    """
    차동 구동 방식 Thruster Allocation
    - 단순 differential drive (left = linear + angular, right = linear - angular)
    - 추진기 각도는 항상 0 (직진)

    Args:
        linear_velocity: 선속도 (-1~1)
        angular_velocity: 각속도 (-1~1)
        thrust_scale: 추력 스케일

    Returns:
        Tuple[float, float, float, float]: (left_thrust, right_thrust, left_pos=0, right_pos=0)
    """
    # Differential drive 제약: left=linear+angular, right=linear-angular ∈ [-1,1]
    max_angular = min(1.0 - linear_velocity, linear_velocity + 1.0)
    min_angular = max(-1.0 - linear_velocity, linear_velocity - 1.0)
    angular_velocity = np.clip(angular_velocity, min_angular, max_angular)

    # 스러스터 명령 계산
    left_cmd = linear_velocity - angular_velocity*30
    right_cmd = linear_velocity + angular_velocity*30

    # 추력 변환
    left_thrust = left_cmd * thrust_scale
    right_thrust = right_cmd * thrust_scale

    # 추력 제한
    left_thrust = np.clip(left_thrust, -thrust_scale, thrust_scale)
    right_thrust = np.clip(right_thrust, -thrust_scale, thrust_scale)

    # 추진기 각도는 0 (직진)
    left_pos = 0.0
    right_pos = 0.0

    return left_thrust, right_thrust, left_pos, right_pos


def simple_surge_yaw_to_thrusters(
    linear_velocity: float,
    angular_velocity: float,
    thrust_scale: float = Constants.DEFAULT_THRUST_SCALE
) -> Tuple[float, float]:
    """
    단순 차동 구동 변환 (각도 정보 없이)
    - 하위 호환성을 위한 헬퍼 함수

    Args:
        linear_velocity: 선속도 (-1~1)
        angular_velocity: 각속도 (-1~1)
        thrust_scale: 추력 스케일

    Returns:
        Tuple[float, float]: (left_thrust, right_thrust)
    """
    left_thrust, right_thrust, _, _ = _differential_drive_allocation(
        linear_velocity, angular_velocity, thrust_scale
    )
    return left_thrust, right_thrust
