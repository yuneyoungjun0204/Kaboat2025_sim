#!/usr/bin/env python3
"""
미션 전략 모듈
- 4가지 미션의 제어 로직을 캡슐화
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Callable
from .detection_system import MissionType


class BaseMissionStrategy:
    """미션 전략 베이스 클래스"""

    def __init__(self, thrust_scale: float = 1000.0):
        self.thrust_scale = thrust_scale

    def execute(self, **kwargs) -> Tuple[float, float]:
        """
        미션 실행

        Returns:
            (left_thrust, right_thrust)
        """
        raise NotImplementedError

    def reset(self):
        """미션 상태 초기화"""
        pass


class PassBetweenBuoysMission(BaseMissionStrategy):
    """미션 1: 부표 사이 지나가기"""

    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, **kwargs) -> Tuple[float, float]:
        """빨간색/초록색 고깔 부표 사이로 지나가기"""
        # 빨간색/초록색 부표 찾기
        red_buoy = None
        green_buoy = None

        for det in detected_objects:
            if det['label'] == 'red_cone':
                red_buoy = det
            elif det['label'] == 'green_cone':
                green_buoy = det

        # 두 부표가 모두 탐지된 경우
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
            steering_gain = 0.003
            forward_speed = 0.5

            steering = error * steering_gain
            steering = np.clip(steering, -0.3, 0.3)

            # 스러스터 명령 계산
            left_thrust = (forward_speed + steering) * self.thrust_scale
            right_thrust = (forward_speed - steering) * self.thrust_scale

            if logger:
                logger.info(
                    f"Pass Buoys: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )
        else:
            # 부표 미탐지 시 천천히 전진
            left_thrust = 0.2 * self.thrust_scale
            right_thrust = 0.2 * self.thrust_scale
            if logger:
                logger.warn("부표 미탐지: 천천히 전진")

        return left_thrust, right_thrust


class CircleBuoyMission(BaseMissionStrategy):
    """미션 2: 부표 주변 회전"""

    def __init__(self, thrust_scale: float = 1000.0):
        super().__init__(thrust_scale)
        self.circle_start_time = None
        self.circle_initial_heading = None
        self.total_rotation = 0.0
        self.previous_heading = None
        self.circling_started = False

    def reset(self):
        """미션 상태 초기화"""
        self.circle_start_time = None
        self.circle_initial_heading = None
        self.total_rotation = 0.0
        self.previous_heading = None
        self.circling_started = False

    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                agent_heading: float, mission_params: Dict, logger=None, **kwargs) -> Tuple[float, float]:
        """파란색 부표 주변을 회전"""
        # 파란색 부표 찾기
        blue_buoy = None
        for det in detected_objects:
            if det['label'] == 'blue_buoy':
                blue_buoy = det
                break

        if not blue_buoy:
            # 부표 미탐지 시 정지
            if logger:
                logger.warn("파란색 부표 미탐지: 정지")
            return 0.0, 0.0

        # 회전 시작 시간 기록
        if self.circle_start_time is None:
            self.circle_start_time = time.time()
            self.circle_initial_heading = agent_heading
            self.previous_heading = agent_heading
            self.total_rotation = 0.0

        # 누적 회전 각도 계산
        heading_diff = agent_heading - self.previous_heading

        # 각도 차이 정규화 (-180 ~ 180)
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360

        self.total_rotation += abs(heading_diff)
        self.previous_heading = agent_heading

        # 360도 회전 완료 확인
        if self.total_rotation >= 350:  # 약간의 여유
            if logger:
                logger.info("부표 회전 완료!")
            return 0.0, 0.0

        # 미션 파라미터
        circle_radius = mission_params.get('circle_radius', 10.0)  # 목표 반경 (미터)
        rotation_direction = mission_params.get('rotation_direction', 1)  # 1=반시계, -1=시계

        # 부표 측정값
        buoy_depth = blue_buoy['depth']  # 부표까지 거리 (미터)
        buoy_x = blue_buoy['center'][0]
        image_center_x = current_image.shape[1] / 2
        lateral_error = buoy_x - image_center_x  # 양수 = 부표가 오른쪽

        # 반경 오차 계산
        radius_error = circle_radius - buoy_depth

        # 두 단계 제어
        if not self.circling_started:
            # ===== 단계 1: 접근 단계 (목표 반경으로 접근) =====
            if abs(radius_error) < 2.0:  # 목표 반경 ±2m 이내
                self.circling_started = True
                if logger:
                    logger.info(f"선회 시작! (depth={buoy_depth:.1f}m, target={circle_radius:.1f}m)")

            # 반경 오차에 비례한 전진/후진 속도
            forward_speed = 0.5 * np.tanh(radius_error / 5.0)
            forward_speed = np.clip(forward_speed, -0.3, 0.7)

            # 부표를 중앙에 유지하면서 접근
            centering_gain = 0.002
            angular_speed = lateral_error * centering_gain

            mode = "APPROACH"
        else:
            # ===== 단계 2: 선회 단계 (반경 유지하며 회전) =====
            # 기본 전진 속도
            forward_speed = 0.4

            # 반경 유지를 위한 속도 조정
            radius_correction = 0.1 * radius_error / circle_radius
            forward_speed += radius_correction
            forward_speed = np.clip(forward_speed, 0.2, 0.6)

            # 회전 속도
            turn_rate = 0.3 * rotation_direction

            # 부표를 시야에 유지하기 위한 조정
            centering_gain = 0.002
            centering_adjustment = lateral_error * centering_gain

            angular_speed = turn_rate + centering_adjustment

            mode = "CIRCLE"

        # 각속도 제한
        angular_speed = np.clip(angular_speed, -0.5, 0.5)

        # 급선회 시 속도 감소
        forward_speed = forward_speed * (1.0 - abs(angular_speed) * 0.2)

        # 스러스터 명령 계산
        left_thrust = (forward_speed + angular_speed) * self.thrust_scale
        right_thrust = (forward_speed - angular_speed) * self.thrust_scale

        # 스러스터 제한
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)

        if logger:
            logger.info(
                f"Circle [{mode}]: rotation={self.total_rotation:.1f}°, "
                f"depth={buoy_depth:.1f}m, radius_err={radius_error:.1f}m, "
                f"lateral_err={lateral_error:.1f}px"
            )

        return left_thrust, right_thrust


class WaypointFollowMission(BaseMissionStrategy):
    """미션 3: 웨이포인트 추종"""

    def execute(self, agent_position: np.ndarray, agent_heading: float,
                target_waypoint: Dict, logger=None, **kwargs) -> Tuple[float, float]:
        """단순 웨이포인트 추종"""
        # 목표 웨이포인트
        target_pos = np.array([target_waypoint['x'], target_waypoint['y']], dtype=np.float32)

        # 목표까지의 벡터
        delta = target_pos - agent_position
        distance = np.linalg.norm(delta)

        if distance < 1.0:
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
        steering_gain = 0.01
        forward_speed = 0.5

        steering = heading_error * steering_gain
        steering = np.clip(steering, -0.5, 0.5)

        # 스러스터 명령
        left_thrust = (forward_speed + steering) * self.thrust_scale
        right_thrust = (forward_speed - steering) * self.thrust_scale

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

    def execute(self, agent_position: np.ndarray, agent_heading: float,
                waypoints: List[Dict], current_waypoint_index: int,
                lidar_distances: np.ndarray, get_lidar_distance_func: Callable,
                get_onnx_control_func: Callable, logger=None, **kwargs) -> Tuple[float, float]:
        """ONNX 모델 + 알고리즘 하이브리드 장애물 회피"""
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
                f"Obstacle Avoid [{mode}]: linear={filtered_linear:.3f}, angular={filtered_angular:.3f}"
            )

        return left_thrust, right_thrust

    def _calculate_thruster_commands(self, linear_velocity: float, angular_velocity: float) -> Tuple[float, float]:
        """스러스터 명령 계산"""
        forward_thrust = linear_velocity * self.thrust_scale
        turn_thrust = angular_velocity * self.thrust_scale
        left_thrust = forward_thrust + turn_thrust
        right_thrust = forward_thrust - turn_thrust
        left_thrust = np.clip(left_thrust, -self.thrust_scale, self.thrust_scale)
        right_thrust = np.clip(right_thrust, -self.thrust_scale, self.thrust_scale)
        return left_thrust, right_thrust


class MissionManager:
    """미션 관리 시스템"""

    def __init__(self, thrust_scale: float = 1000.0, avoidance_controller=None):
        self.thrust_scale = thrust_scale
        self.missions = {
            MissionType.PASS_BETWEEN_BUOYS: PassBetweenBuoysMission(thrust_scale),
            MissionType.CIRCLE_BUOY: CircleBuoyMission(thrust_scale),
            MissionType.WAYPOINT_FOLLOW: WaypointFollowMission(thrust_scale),
            MissionType.OBSTACLE_AVOID: ObstacleAvoidMission(thrust_scale, avoidance_controller)
        }
        self.current_mission = None

    def set_mission(self, mission_type: MissionType):
        """현재 미션 설정 및 초기화"""
        if mission_type != self.current_mission:
            # 이전 미션 리셋
            if self.current_mission and self.current_mission in self.missions:
                self.missions[self.current_mission].reset()
            self.current_mission = mission_type

    def update_thrust_scale(self, thrust_scale: float):
        """모든 미션의 thrust_scale 업데이트"""
        self.thrust_scale = thrust_scale
        for mission in self.missions.values():
            mission.thrust_scale = thrust_scale

    def execute_mission(self, mission_type: MissionType, **kwargs) -> Tuple[float, float]:
        """미션 실행"""
        mission = self.missions.get(mission_type)
        if mission:
            return mission.execute(**kwargs)
        return 0.0, 0.0
