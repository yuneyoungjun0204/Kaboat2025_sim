"""
Depth filtering utilities for stabilizing noisy depth measurements.
Implements low-pass filters to reduce high-frequency noise in depth data.
"""

from typing import Dict, Optional
import numpy as np


class ExponentialMovingAverageFilter:
    """
    지수 이동 평균 (EMA) 저주파 필터

    y(t) = alpha * x(t) + (1 - alpha) * y(t-1)

    where:
        - x(t): 현재 측정값 (noisy input)
        - y(t): 필터링된 출력값 (smoothed output)
        - alpha: smoothing factor (0 < alpha <= 1)
            - alpha = 1: 필터링 없음 (원본 값 사용)
            - alpha → 0: 강한 필터링 (느린 반응)
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: smoothing factor (0 < alpha <= 1)
                   권장값: 0.2 ~ 0.4 (depth 안정화용)
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")

        self.alpha = alpha
        self.filtered_value: Optional[float] = None

    def update(self, measurement: float) -> float:
        """
        새로운 측정값으로 필터 업데이트

        Args:
            measurement: 현재 depth 측정값

        Returns:
            필터링된 depth 값
        """
        if self.filtered_value is None:
            # 첫 측정값은 그대로 사용
            self.filtered_value = measurement
        else:
            # EMA 공식 적용
            self.filtered_value = self.alpha * measurement + (1 - self.alpha) * self.filtered_value

        return self.filtered_value

    def get_value(self) -> Optional[float]:
        """현재 필터링된 값 반환"""
        return self.filtered_value

    def reset(self) -> None:
        """필터 초기화"""
        self.filtered_value = None

    def is_initialized(self) -> bool:
        """필터가 초기화되었는지 확인"""
        return self.filtered_value is not None


class SpatialDepthSmoother:
    """
    공간적 depth smoothing
    중심점 주변 영역의 depth 값들을 평균하여 단일 픽셀 노이즈 제거
    """

    @staticmethod
    def smooth_depth_at_point(
        depth_map: np.ndarray,
        cx: int,
        cy: int,
        kernel_size: int = 5,
        method: str = 'median'
    ) -> float:
        """
        특정 좌표 주변 영역의 depth 값을 평활화

        Args:
            depth_map: 전체 depth map (2D numpy array)
            cx: 중심 x 좌표
            cy: 중심 y 좌표
            kernel_size: 평활화 커널 크기 (홀수 권장, default=5)
            method: 'mean' (평균) 또는 'median' (중앙값)
                   median이 outlier에 더 robust

        Returns:
            평활화된 depth 값
        """
        h, w = depth_map.shape

        # 커널 반경 계산
        radius = kernel_size // 2

        # 경계 처리
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)

        # 주변 영역 추출
        region = depth_map[y_min:y_max, x_min:x_max]

        # 평활화 방법 선택
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        else:
            raise ValueError(f"Unknown smoothing method: {method}")


class DepthFilterManager:
    """
    객체별 depth 필터 관리자
    각 track_id에 대해 독립적인 EMA 필터 유지
    """

    def __init__(self, alpha: float = 0.3, timeout: int = 30):
        """
        Args:
            alpha: EMA smoothing factor
            timeout: 필터 타임아웃 (프레임 수)
                    이 프레임 동안 업데이트 없으면 필터 제거
        """
        self.alpha = alpha
        self.timeout = timeout

        # track_id -> EMA filter
        self.filters: Dict[int, ExponentialMovingAverageFilter] = {}

        # track_id -> 마지막 업데이트 프레임 카운터
        self.last_update: Dict[int, int] = {}

        # 전역 프레임 카운터
        self.frame_count = 0

    def update(self, track_id: int, depth: float) -> float:
        """
        특정 track의 depth 값 업데이트 및 필터링

        Args:
            track_id: 객체 추적 ID
            depth: 현재 depth 측정값

        Returns:
            필터링된 depth 값
        """
        # 새로운 track이면 필터 생성
        if track_id not in self.filters:
            self.filters[track_id] = ExponentialMovingAverageFilter(alpha=self.alpha)

        # 필터 업데이트
        filtered_depth = self.filters[track_id].update(depth)

        # 마지막 업데이트 시간 기록
        self.last_update[track_id] = self.frame_count

        return filtered_depth

    def get_filtered_depth(self, track_id: int) -> Optional[float]:
        """특정 track의 필터링된 depth 값 조회"""
        if track_id in self.filters:
            return self.filters[track_id].get_value()
        return None

    def remove_track(self, track_id: int) -> None:
        """특정 track의 필터 제거"""
        if track_id in self.filters:
            del self.filters[track_id]
        if track_id in self.last_update:
            del self.last_update[track_id]

    def cleanup_old_tracks(self) -> None:
        """타임아웃된 track들의 필터 정리"""
        expired_tracks = [
            track_id for track_id, last_frame in self.last_update.items()
            if self.frame_count - last_frame > self.timeout
        ]

        for track_id in expired_tracks:
            self.remove_track(track_id)

    def increment_frame(self) -> None:
        """프레임 카운터 증가 및 자동 cleanup"""
        self.frame_count += 1

        # 주기적으로 cleanup (매 30 프레임마다)
        if self.frame_count % 30 == 0:
            self.cleanup_old_tracks()

    def reset(self) -> None:
        """모든 필터 초기화"""
        self.filters.clear()
        self.last_update.clear()
        self.frame_count = 0


# 편의 함수들
def create_ema_filter(alpha: float = 0.3) -> ExponentialMovingAverageFilter:
    """EMA 필터 생성 팩토리 함수"""
    return ExponentialMovingAverageFilter(alpha=alpha)


def smooth_depth_spatially(
    depth_map: np.ndarray,
    x: int,
    y: int,
    kernel_size: int = 5
) -> float:
    """공간적 depth smoothing 편의 함수"""
    return SpatialDepthSmoother.smooth_depth_at_point(
        depth_map, x, y, kernel_size, method='median'
    )
