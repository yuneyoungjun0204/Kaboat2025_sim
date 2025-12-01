#!/usr/bin/env python3
"""
공통 헬퍼 함수 모듈
- 중복 제거를 위한 유틸리티 함수들
- 모든 미션에서 공통적으로 사용되는 헬퍼 함수 제공

Functions:
    - normalize_heading: 헤딩 0~360 정규화
    - calculate_heading_error: 최단 거리 헤딩 오차 계산
    - find_buoy_with_fallback: 부표 탐지 (추적값 우선)
    - clip_value: 값 범위 제한
    - safe_divide: 0으로 나누기 방지
    - sanitize_value: NaN/Inf 제거
    - interpolate_linear: 선형 보간
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any


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
    부표 탐지: 추적값만 사용 (원본 측정값 사용 안 함)

    Args:
        label: 찾을 부표 라벨 ('red_cone', 'green_cone', 'blue_buoy' 등)
        detected_objects: 추적된 객체 리스트 (IMM-PDAF 출력)
        raw_detections: 원본 탐지 결과 (미사용, 호환성 유지)
        logger: 로거

    Returns:
        Tuple[Optional[Dict], str]: (부표 딕셔너리, 데이터 소스)
            데이터 소스는 'TRACKED' 또는 'NONE'
    """
    # 추적값에서만 찾기 (원본 측정값 사용 안 함)
    if detected_objects:
        for det in detected_objects:
            if det['label'] == label:
                return det, "TRACKED"

    # 추적값이 없으면 None 반환
    return None, "NONE"


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    값을 범위 내로 제한

    Args:
        value: 입력 값
        min_val: 최소값
        max_val: 최대값

    Returns:
        float: 제한된 값
    """
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    안전한 나눗셈 (0으로 나누기 방지)

    Args:
        numerator: 분자
        denominator: 분모
        default: 분모가 0일 때 반환할 값

    Returns:
        float: 나눗셈 결과 또는 기본값
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def sanitize_value(value: float, default: float = 0.0) -> float:
    """
    NaN 및 Inf 값을 안전한 값으로 변환 (성능 최적화 버전)

    Args:
        value: 입력 값
        default: NaN/Inf일 때 반환할 값

    Returns:
        float: 정상 값 또는 기본값

    Note:
        numpy.isfinite()가 isnan() + isinf()보다 빠름
    """
    return default if not np.isfinite(value) else float(value)


def interpolate_linear(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    선형 보간

    Args:
        x: 보간할 x 값
        x0, y0: 첫 번째 점
        x1, y1: 두 번째 점

    Returns:
        float: 보간된 y 값
    """
    if abs(x1 - x0) < 1e-10:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
