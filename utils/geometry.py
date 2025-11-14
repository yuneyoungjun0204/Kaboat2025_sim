#!/usr/bin/env python3
"""
기하학 유틸리티 모듈
- 각도 정규화 함수
- 좌표 변환 유틸리티
- 거리 계산 함수
"""

import numpy as np
from typing import Tuple


# ============================================================================
# 각도 정규화 함수 (통합)
# ============================================================================

def normalize_angle_180(angle_deg: float) -> float:
    """
    각도를 -180~180도 범위로 정규화 (NED 좌표계 기준)

    Args:
        angle_deg: 입력 각도 (도 단위)

    Returns:
        -180~180도 범위로 정규화된 각도

    Examples:
        >>> normalize_angle_180(0)
        0.0
        >>> normalize_angle_180(270)
        -90.0
        >>> normalize_angle_180(360)
        0.0
    """
    # 먼저 0~360도 범위로 변환
    angle_deg = angle_deg % 360.0

    # 180도를 넘으면 음수로 변환
    if angle_deg > 180.0:
        angle_deg -= 360.0

    return angle_deg


def normalize_angle_360(angle_deg: float) -> float:
    """
    각도를 0~360도 범위로 정규화

    Args:
        angle_deg: 입력 각도 (도 단위)

    Returns:
        0~360도 범위로 정규화된 각도

    Examples:
        >>> normalize_angle_360(-90)
        270.0
        >>> normalize_angle_360(450)
        90.0
    """
    return angle_deg % 360.0


def normalize_angle_rad(angle_rad: float) -> float:
    """
    각도를 -π~π 범위로 정규화 (라디안)

    Args:
        angle_rad: 입력 각도 (라디안 단위)

    Returns:
        -π~π 범위로 정규화된 각도 (라디안)
    """
    # 먼저 0~2π 범위로 변환
    angle_rad = angle_rad % (2.0 * np.pi)

    # π를 넘으면 음수로 변환
    if angle_rad > np.pi:
        angle_rad -= 2.0 * np.pi

    return angle_rad


# ============================================================================
# 거리 계산 함수
# ============================================================================

def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    두 위치 간의 유클리드 거리 계산

    Args:
        pos1: 첫 번째 위치 [x, y]
        pos2: 두 번째 위치 [x, y]

    Returns:
        두 점 사이의 거리 (m)
    """
    return np.linalg.norm(pos1 - pos2)


def calculate_heading(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """
    한 점에서 다른 점으로의 방향 계산 (NED 좌표계)

    Args:
        from_pos: 시작 위치 [x, y]
        to_pos: 목표 위치 [x, y]

    Returns:
        방향 각도 (도, -180~180)
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    heading_rad = np.arctan2(dy, dx)
    heading_deg = np.degrees(heading_rad)
    return normalize_angle_180(heading_deg)


# ============================================================================
# 좌표 변환 함수
# ============================================================================

def polar_to_cartesian(range_val: float, angle_rad: float) -> Tuple[float, float]:
    """
    극좌표를 직교좌표로 변환

    Args:
        range_val: 거리 (m)
        angle_rad: 각도 (라디안)

    Returns:
        (x, y): 직교좌표
    """
    x = range_val * np.cos(angle_rad)
    y = range_val * np.sin(angle_rad)
    return x, y


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """
    직교좌표를 극좌표로 변환

    Args:
        x: X 좌표
        y: Y 좌표

    Returns:
        (range, angle): 거리와 각도 (라디안)
    """
    range_val = np.sqrt(x**2 + y**2)
    angle_rad = np.arctan2(y, x)
    return range_val, angle_rad
