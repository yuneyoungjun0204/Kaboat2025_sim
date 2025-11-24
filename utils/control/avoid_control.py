#!/usr/bin/env python3
"""
장애물 회피 제어 모듈
- LOS (Line of Sight) guidance 기반 경로 추종
- 장애물 감지 및 회피 판단
- ONNX 모델과 직접 제어 통합

좌표계 규칙:
- 모든 위치는 [x, y] 형태 (UTM 좌표: x=Easting, y=Northing)
- agent_heading: 0-360도 (0도=북쪽, 90도=동쪽, 180도=남쪽, 270도=서쪽)
- LiDAR 각도: -100~100도 (0도=정면, 양수=좌측, 음수=우측, 로봇 기준)
"""

import numpy as np
import math
from typing import Tuple, Optional, List
from ..core.config import Constants


class LOSGuidance:
    """LOS (Line of Sight) Guidance 시스템"""

    def __init__(self, delta=None, lookahead_min=None, lookahead_max=None, lookahead_factor=None):
        """
        Args:
            delta: 수직 오프셋 (미터) - None이면 config에서 로드
            lookahead_min: 최소 look-ahead 거리 (미터) - None이면 config에서 로드
            lookahead_max: 최대 look-ahead 거리 (미터) - None이면 config에서 로드
            lookahead_factor: look-ahead 거리 계산 계수 - None이면 config에서 로드
        """
        ac = Constants.AvoidControl
        self.delta = delta if delta is not None else ac.LOS_DELTA
        self.lookahead_min = lookahead_min if lookahead_min is not None else ac.LOS_LOOKAHEAD_MIN
        self.lookahead_max = lookahead_max if lookahead_max is not None else ac.LOS_LOOKAHEAD_MAX
        self.lookahead_factor = lookahead_factor if lookahead_factor is not None else ac.LOS_LOOKAHEAD_FACTOR

    def update_delta(self, delta: float):
        """LOS delta 동적 업데이트"""
        self.delta = delta

    def calculate_crosstrack_error(self, current_pos: np.ndarray, line_start: np.ndarray,
                                   line_end: np.ndarray) -> float:
        """
        현재 위치에서 직선까지의 crosstrack error 계산

        Args:
            current_pos: 현재 위치 [x, y]
            line_start: 직선 시작점 [x, y]
            line_end: 직선 끝점 [x, y]

        Returns:
            crosstrack error (양수=우측, 음수=좌측)
        """
        line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        line_length = np.linalg.norm(line_vec)

        if line_length < 0.01:  # 거의 같은 점
            return 0.0

        point_vec = np.array([current_pos[0] - line_start[0], current_pos[1] - line_start[1]])
        line_unit = line_vec / line_length
        crosstrack_error = np.linalg.norm(point_vec - np.dot(point_vec, line_unit) * line_unit)

        # 부호 결정 (외적 사용)
        cross_product = np.cross(line_unit, point_vec)
        if cross_product < 0:
            crosstrack_error = -crosstrack_error

        return crosstrack_error

    def calculate_adaptive_lookahead(self, crosstrack_error: float) -> float:
        """crosstrack error에 따른 adaptive look-ahead distance 계산"""
        abs_crosstrack_error = abs(crosstrack_error)
        # Crosstrack error가 클수록 lookahead를 짧게
        adaptive_lookahead = self.lookahead_max / (1.0 + abs_crosstrack_error * 0.05)
        return np.clip(adaptive_lookahead, self.lookahead_min, self.lookahead_max)

    def calculate_los_point(self, current_pos: np.ndarray, waypoint_start: np.ndarray,
                           waypoint_end: np.ndarray) -> np.ndarray:
        """
        LOS guidance를 사용한 look-ahead point 계산

        Args:
            current_pos: 현재 위치 [x, y]
            waypoint_start: 경로 시작점 [x, y]
            waypoint_end: 경로 끝점 [x, y]

        Returns:
            LOS target 위치 [x, y]
        """
        # Crosstrack error 계산
        crosstrack_error = self.calculate_crosstrack_error(current_pos, waypoint_start, waypoint_end)
        adaptive_lookahead = self.calculate_adaptive_lookahead(crosstrack_error)

        # 경로 벡터 계산
        line_vec = np.array([waypoint_end[0] - waypoint_start[0], waypoint_end[1] - waypoint_start[1]])
        line_length = np.linalg.norm(line_vec)

        if line_length < 0.01:
            return waypoint_end

        line_unit = line_vec / line_length

        # 현재 위치를 경로에 투영
        point_vec = np.array([current_pos[0] - waypoint_start[0], current_pos[1] - waypoint_start[1]])
        projection_length = np.dot(point_vec, line_unit)
        projection_point = np.array(waypoint_start) + projection_length * line_unit

        # Lookahead point 계산 (경로 방향으로 전진)
        los_point = projection_point + adaptive_lookahead * line_unit

        # Crosstrack error 보정 (경로로 부드럽게 복귀)
        perpendicular_unit = np.array([-line_unit[1], line_unit[0]])  # 90도 회전 (좌측 방향)
        perpendicular_offset = crosstrack_error * self.delta / 10.0
        los_point += perpendicular_offset * perpendicular_unit

        return los_point


class ObstacleDetector:
    """장애물 감지 시스템"""

    def __init__(self, boat_width=None, boat_height=None, max_lidar_distance=None, 
                 obstacle_count_threshold=None, obstacle_count_threshold_off=None):
        """
        Args:
            boat_width: 배 폭 (미터) - None이면 config에서 로드
            boat_height: 배 높이 (미터) - None이면 config에서 로드
            max_lidar_distance: LiDAR 최대 거리 (미터) - None이면 config에서 로드
            obstacle_count_threshold: ONNX 모드 전환을 위한 최소 장애물 감지 개수 - None이면 config에서 로드
        """
        ac = Constants.AvoidControl
        self.boat_width = boat_width if boat_width is not None else ac.OBSTACLE_BOAT_WIDTH
        self.boat_height = boat_height if boat_height is not None else ac.OBSTACLE_BOAT_HEIGHT
        self.max_lidar_distance = max_lidar_distance if max_lidar_distance is not None else Constants.MAX_LIDAR_DISTANCE
        self.obstacle_count_threshold = obstacle_count_threshold if obstacle_count_threshold is not None else ac.OBSTACLE_COUNT_THRESHOLD
        self.obstacle_count_threshold_off = obstacle_count_threshold_off if obstacle_count_threshold_off is not None else ac.OBSTACLE_COUNT_THRESHOLD_OFF

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """각도를 -π ~ π 범위로 정규화"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def calculate_goal_psi(self, current_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """
        목표 방향각 계산 (북쪽 기준, 시계방향)

        Args:
            current_pos: 현재 위치 [North, East] (NED 좌표계)
            target_pos: 목표 위치 [North, East]

        Returns:
            방향각 (라디안, 북쪽=0, 동쪽=π/2, 시계방향)
        """
        # NED 좌표계: [North, East]
        d_north = target_pos[0] - current_pos[0]  # North 차이
        d_east = target_pos[1] - current_pos[1]   # East 차이
        # atan2(East, North) = 북쪽 기준 시계방향 각도
        return np.arctan2(d_north, d_east)

    def calculate_distance(self, current_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """목적지와 현재 위치 간의 거리 계산"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        return min(self.boat_height, np.sqrt(dx**2 + dy**2))

    def calculate_range_theta(self, distance: float) -> float:
        """배 폭을 고려한 탐색 각도 범위 계산"""
        if distance < 0.1:
            return np.pi / 4  # 기본 45도
        return np.arctan2(self.boat_width / 2.0, distance)

    def check_obstacles(self, current_pos: np.ndarray, los_target: np.ndarray,
                       agent_heading: float, lidar_distances: np.ndarray,
                       get_lidar_distance_func) -> Tuple[bool, List[float]]:
        """
        LOS target으로 가는 경로에 장애물이 있는지 확인

        Args:
            current_pos: 현재 위치 [x, y] (UTM)
            los_target: LOS 목표 위치 [x, y] (UTM)
            agent_heading: 로봇 방향 (0-360도, 북쪽=0)
            lidar_distances: LiDAR 거리 배열
            get_lidar_distance_func: LiDAR 거리 조회 함수

        Returns:
            (장애물 존재 여부, 체크 영역 점들 [x1, y1, x2, y2, ...])
        """
        L = self.calculate_distance(current_pos, los_target)
        goal_psi = self.calculate_goal_psi(current_pos, los_target)  # 목표 방향 (북쪽 기준)
        current_psi = np.radians(agent_heading)  # 현재 방향 (북쪽 기준)

        # LiDAR 각도로 변환 (로봇 정면 기준)
        relative_angle = goal_psi - current_psi
        relative_angle = self.normalize_angle(relative_angle)

        range_theta = self.calculate_range_theta(L)

        check_area_points = []
        obstacle_count = 0  # 장애물 개수 카운팅

        # 1. LOS target으로 가는 경로 체크
        # 상대 각도를 중심으로 배 폭만큼의 범위를 체크
        num_rays = 181  # -90 ~ 90도
        # for i in range(-90, 91):
        #     # 로봇 기준 각도 (LiDAR 좌표계)
        #     lidar_angle_deg = np.degrees(relative_angle) + i

        #     # 탐색 거리 결정
        #     if abs(np.radians(i)) <= range_theta:
        #         search_distance = L  # 경로 중심부는 목표까지 거리
        #     else:
        #         # 경로 양옆은 배 폭 기준
        #         angle_rad = abs(np.radians(i))
        #         if angle_rad > 0.01:
        #             search_distance = (self.boat_width / 2.0) / np.sin(angle_rad)
        #             search_distance = min(search_distance, L)
        #         else:
        #             search_distance = L

            # # 체크 영역 점 계산 (NED 좌표계)
            # world_angle = current_psi + np.radians(lidar_angle_deg)
            # # current_pos = [North, East]
            # check_north = current_pos[0] + search_distance * np.cos(world_angle)  # North
            # check_east = current_pos[1] + search_distance * np.sin(world_angle)   # East
            # # 발행 형식: [East, North] (trajectory_viz에서 순서를 바꿔서 받음)
            # check_area_points.extend([check_east, check_north])

            # # LiDAR 거리 조회
            # lidar_distance = get_lidar_distance_func(lidar_angle_deg)
            # if lidar_distance > self.max_lidar_distance or lidar_distance < 0.0 or np.isinf(lidar_distance):
            #     lidar_distance = self.max_lidar_distance

            # # 장애물 감지 (개수 카운팅)
            # if lidar_distance < search_distance:
            #     obstacle_count += 1

        # 2. 정면 긴급 장애물 검사
        L_front = Constants.AvoidControl.L_FRONT
        range_theta_front = self.calculate_range_theta(L_front)

        for i in range(-90, 91):
            lidar_angle_deg = float(i)

            # 탐색 거리 결정
            if abs(np.radians(i)) <= range_theta_front:
                search_distance = L_front
            else:
                angle_rad = abs(np.radians(i))
                if angle_rad > 0.01:
                    search_distance = (self.boat_width / 2.0) / np.sin(angle_rad)
                    search_distance = min(search_distance, L_front)
                else:
                    search_distance = L_front

            # 체크 영역 점 계산 (NED 좌표계) - 정면 검사 영역도 시각화에 추가
            world_angle = current_psi + np.radians(lidar_angle_deg)
            # current_pos = [North, East]
            check_north = current_pos[0] + search_distance * np.cos(world_angle)  # North
            check_east = current_pos[1] + search_distance * np.sin(world_angle)   # East
            # 발행 형식: [East, North] (trajectory_viz에서 순서를 바꿔서 받음)
            check_area_points.extend([check_north, check_east])

            # LiDAR 거리 조회
            lidar_distance = get_lidar_distance_func(lidar_angle_deg)
            if lidar_distance > self.max_lidar_distance or lidar_distance < 0.0 or np.isinf(lidar_distance):
                lidar_distance = self.max_lidar_distance

            # 장애물 감지 (개수 카운팅)
            if lidar_distance < search_distance:
                obstacle_count += 1

        # 장애물 개수 반환 (히스테리시스 로직은 상위에서 처리)
        return obstacle_count, check_area_points


class DirectController:
    """직접 제어 시스템 (장애물이 없을 때)"""

    @staticmethod
    def calculate_heading_diff(current_pos: np.ndarray, target_pos: np.ndarray,
                              agent_heading: float) -> float:
        """
        목적지로 향하는 헤딩 차이 계산

        Args:
            current_pos: 현재 위치 [x, y] (x=Easting, y=Northing)
            target_pos: 목표 위치 [x, y]
            agent_heading: 현재 헤딩 (0-360도, 북쪽=0, 시계방향)

        Returns:
            헤딩 차이 (라디안, 양수=좌회전 필요, 음수=우회전 필요)
        """
        dx = target_pos[0] - current_pos[0]  # Easting 차이 (동쪽 방향)
        dy = target_pos[1] - current_pos[1]  # Northing 차이 (북쪽 방향)

        # 목표 방향 계산 (북쪽 기준, atan2(동쪽, 북쪽))
        target_heading_rad = np.arctan2(dx, dy)
        current_heading_rad = np.radians(agent_heading)

        # 헤딩 차이 계산 (-π ~ π)
        heading_diff_rad = target_heading_rad - current_heading_rad
        heading_diff_rad = np.arctan2(np.sin(heading_diff_rad), np.cos(heading_diff_rad))

        return -heading_diff_rad

    @staticmethod
    def calculate_control(current_pos: np.ndarray, los_target: np.ndarray,
                         agent_heading: float) -> Tuple[float, float]:
        """
        LOS target을 향한 직접제어 명령 계산

        Args:
            current_pos: 현재 위치 [x, y]
            los_target: LOS 목표 위치 [x, y]
            agent_heading: 현재 헤딩 (0-360도)

        Returns:
            (linear_velocity, angular_velocity)
        """
        heading_diff_rad = DirectController.calculate_heading_diff(current_pos, los_target, agent_heading)
        distance_to_los = np.sqrt((los_target[0] - current_pos[0])**2 +
                                 (los_target[1] - current_pos[1])**2)

        # 각속도: 헤딩 차이에 비례
        ac = Constants.AvoidControl
        angular_velocity = np.clip(heading_diff_rad / np.pi, -ac.ANGULAR_VELOCITY_CLIP, ac.ANGULAR_VELOCITY_CLIP)

        # 선속도: 거리에 따라 조절
        if distance_to_los > ac.DIRECT_SPEED_FAR_DISTANCE:
            linear_velocity = ac.DIRECT_SPEED_FAR
        elif distance_to_los > ac.DIRECT_SPEED_MID_DISTANCE:
            linear_velocity = ac.DIRECT_SPEED_MID
        else:
            linear_velocity = ac.DIRECT_SPEED_NEAR

        # 회전 중에는 속도 감소
        linear_velocity = linear_velocity * (1.0 - abs(angular_velocity) * 0.3)
        linear_velocity = np.clip(linear_velocity, 0.1, 1.0)

        return linear_velocity, angular_velocity


class LowPassFilter:
    """1차 저주파 필터"""

    def __init__(self, alpha=None):
        """
        Args:
            alpha: 필터 계수 (0~1, 낮을수록 더 부드러움) - None이면 config에서 로드
        """
        self.alpha = alpha if alpha is not None else Constants.AvoidControl.LPF_ALPHA
        self.filtered_linear_velocity = 0.0
        self.filtered_angular_velocity = 0.0
        self.filtered_left_thrust = 0.0
        self.filtered_right_thrust = 0.0

    def apply(self, new_value: float, current_filtered_value: float) -> float:
        """1차 저주파 필터 적용"""
        return self.alpha * new_value + (1 - self.alpha) * current_filtered_value

    def filter_control(self, linear_velocity: float, angular_velocity: float) -> Tuple[float, float]:
        """제어 명령에 필터 적용"""
        self.filtered_linear_velocity = self.apply(linear_velocity, self.filtered_linear_velocity)
        self.filtered_angular_velocity = self.apply(angular_velocity, self.filtered_angular_velocity)
        return self.filtered_linear_velocity, self.filtered_angular_velocity

    def filter_thrust(self, left_thrust: float, right_thrust: float) -> Tuple[float, float]:
        """스러스터 명령에 필터 적용"""
        self.filtered_left_thrust = self.apply(left_thrust, self.filtered_left_thrust)
        self.filtered_right_thrust = self.apply(right_thrust, self.filtered_right_thrust)
        return self.filtered_left_thrust, self.filtered_right_thrust


class AvoidanceController:
    """
    통합 장애물 회피 제어 시스템
    - LOS guidance, 장애물 감지, 직접 제어를 통합
    """

    def __init__(self, boat_width=None, boat_height=None, max_lidar_distance=None,
                 los_delta=None, los_lookahead_min=None, los_lookahead_max=None,
                 filter_alpha=None, obstacle_count_threshold=None):
        """
        Args:
            boat_width: 배 폭 (미터) - None이면 config에서 로드
            boat_height: 배 높이 (미터) - None이면 config에서 로드
            max_lidar_distance: LiDAR 최대 거리 (미터) - None이면 config에서 로드
            los_delta: LOS 수직 오프셋 (미터) - None이면 config에서 로드
            los_lookahead_min: 최소 look-ahead 거리 (미터) - None이면 config에서 로드
            los_lookahead_max: 최대 look-ahead 거리 (미터) - None이면 config에서 로드
            filter_alpha: 필터 계수 - None이면 config에서 로드
            obstacle_count_threshold: ONNX 모드 전환을 위한 최소 장애물 감지 개수 - None이면 config에서 로드

        Note:
            모든 파라미터가 None이면 Constants.AvoidControl에서 기본값을 로드합니다.
        """
        ac = Constants.AvoidControl
        self.los_guidance = LOSGuidance(los_delta, los_lookahead_min, los_lookahead_max)
        # ObstacleDetector에 threshold_off도 전달
        obstacle_count_threshold_off = ac.OBSTACLE_COUNT_THRESHOLD_OFF
        self.obstacle_detector = ObstacleDetector(
            boat_width, boat_height, max_lidar_distance, 
            obstacle_count_threshold, obstacle_count_threshold_off
        )
        self.low_pass_filter = LowPassFilter(filter_alpha)
        
        # 히스테리시스: 현재 제어 모드 상태 저장 (채터링 방지)
        self.current_mode_is_onnx = False  # False=DIRECT, True=ONNX

    def update_los_delta(self, los_delta: float):
        """LOS delta 동적 업데이트"""
        self.los_guidance.update_delta(los_delta)

    def get_los_target(self, current_pos: np.ndarray, waypoints: List,
                      current_target_index: int) -> np.ndarray:
        """
        LOS target 계산
        - 현재 위치에서 다음 웨이포인트로 가는 LOS guidance 적용

        Args:
            current_pos: 현재 위치 [x, y]
            waypoints: 웨이포인트 리스트 [[x1, y1], [x2, y2], ...]
            current_target_index: 현재 목표 웨이포인트 인덱스

        Returns:
            LOS target 위치 [x, y]
        """
        if len(waypoints) == 0:
            return current_pos

        # 유효한 인덱스 확인
        if current_target_index >= len(waypoints):
            current_target_index = len(waypoints) - 1

        # 경로 시작점: 항상 현재 위치
        waypoint_start = current_pos

        # 경로 끝점: 목표 웨이포인트
        waypoint_end = np.array(waypoints[current_target_index])

        # LOS point 계산
        los_target = self.los_guidance.calculate_los_point(
            current_pos, waypoint_start, waypoint_end
        )

        return los_target

    def check_obstacles_and_get_control(self, current_pos: np.ndarray, los_target: np.ndarray,
                                       agent_heading: float, lidar_distances: np.ndarray,
                                       get_lidar_distance_func, onnx_control_func) -> Tuple[bool, float, float, List[float]]:
        """
        장애물을 확인하고 제어 명령 계산 (히스테리시스 적용)

        Returns:
            (use_direct_control, linear_velocity, angular_velocity, check_area_points)
        """
        # 장애물 검사 (장애물 개수 반환)
        obstacle_count, check_area_points = self.obstacle_detector.check_obstacles(
            current_pos, los_target, agent_heading, lidar_distances, get_lidar_distance_func
        )

        # 히스테리시스 로직: 채터링 방지
        # ONNX 모드로 전환: 장애물 개수 >= threshold
        # DIRECT 모드로 전환: 장애물 개수 < threshold_off
        if self.current_mode_is_onnx:
            # 현재 ONNX 모드: threshold_off 이하로 내려가야 DIRECT로 전환
            if obstacle_count < self.obstacle_detector.obstacle_count_threshold_off:
                self.current_mode_is_onnx = False
        else:
            # 현재 DIRECT 모드: threshold 이상 올라가야 ONNX로 전환
            if obstacle_count >= self.obstacle_detector.obstacle_count_threshold:
                self.current_mode_is_onnx = True

        # 제어 모드에 따라 명령 계산
        if self.current_mode_is_onnx:
            # ONNX 모델 사용
            linear_velocity, angular_velocity = onnx_control_func()
            use_direct_control = False
        else:
            # 직접 제어 (LOS guidance)
            linear_velocity, angular_velocity = DirectController.calculate_control(
                current_pos, los_target, agent_heading
            )
            use_direct_control = True

        return use_direct_control, linear_velocity, angular_velocity, check_area_points

    def apply_filters(self, linear_velocity: float, angular_velocity: float) -> Tuple[float, float]:
        """제어 명령에 필터 적용"""
        return self.low_pass_filter.filter_control(linear_velocity, angular_velocity)

    def apply_thrust_filters(self, left_thrust: float, right_thrust: float) -> Tuple[float, float]:
        """스러스터 명령에 필터 적용"""
        return self.low_pass_filter.filter_thrust(left_thrust, right_thrust)
