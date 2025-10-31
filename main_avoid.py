#!/usr/bin/env python3
"""
VRX 장애물 회피 전용 제어 시스템
- OBSTACLE_AVOID 미션 전용 (탐지/트래킹 시스템 제외)
- ONNX + LOS guidance 하이브리드 제어
- Main_MCP.py 기반 간소화 버전
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np

from utils import Constants, VRXSystemFactory
from utils.detection_system import MissionType
from utils.mission_control import WaypointTransitionHandler, ObstacleAvoidExecutor


class VRXObstacleAvoidController(Node):
    """장애물 회피 전용 제어 노드"""

    # 클릭 웨이포인트 처리 모드
    CLICK_MODE_INSERT_NEXT = "insert_next"  # 현재 목표 다음에 삽입
    CLICK_MODE_GO_IMMEDIATE = "go_immediate"  # 즉시 클릭 지점으로 이동

    def __init__(self, click_mode: str = CLICK_MODE_GO_IMMEDIATE):
        """
        Args:
            click_mode: 클릭 웨이포인트 처리 모드
                - "insert_next": 현재 목표 완료 후 클릭 지점으로 이동
                - "go_immediate": 클릭 즉시 방향 전환하여 이동 (기본값)
        """
        super().__init__('vrx_obstacle_avoid_controller')
        self._log("VRX 장애물 회피 시스템 초기화", header=True)

        self.click_mode = click_mode

        # 컴포넌트 초기화 (최소 구성)
        factory = VRXSystemFactory(self, None, self.get_logger())
        c = factory.create_minimal_components()

        self.sensors = c['sensor_handler']
        self.waypoint_mgr = c['waypoint_manager']
        self.mission_mgr = c['mission_manager']
        self.mission_mgr.set_mission(MissionType.OBSTACLE_AVOID)

        # ROS2 통신 설정
        from utils.ros_communication import ROSCommunicationManager
        self.ros = ROSCommunicationManager(self)
        self.ros.setup_subscribers({
            'lidar': self.sensors.lidar_callback,
            'gps': self.sensors.gps_callback,
            'imu': self.sensors.imu_callback,
            'waypoint': lambda msg: self.sensors.waypoint_callback(msg, self._add_waypoint)
        })
        self.ros.setup_publishers()

        # 실행자 설정
        self.wp_handler = WaypointTransitionHandler(
            self.waypoint_mgr, self.mission_mgr, self.get_logger()
        )
        self.avoid_executor = ObstacleAvoidExecutor(
            c['avoidance_controller'], c['onnx_controller'],
            self.waypoint_mgr, c['mission_executor'],
            self.sensors, self.ros, self.get_logger()
        )

        # 상태
        self.completed = False
        self.loop_count = 0

        # 타이머 (10Hz)
        self.create_timer(Constants.MAIN_LOOP_PERIOD, self._control_loop)

        # 모드 안내
        mode_desc = "즉시 이동" if click_mode == self.CLICK_MODE_GO_IMMEDIATE else "다음 목표"
        self._log(f"🖱️  클릭 모드: {mode_desc}")
        self._log("초기화 완료!", header=True)

    def _add_waypoint(self, msg: Point):
        """
        웨이포인트 추가 (클릭 지점 우선 처리)

        모드에 따라 동작이 달라집니다:
        - insert_next: 현재 목표 완료 후 클릭 지점으로 이동
        - go_immediate: 즉시 클릭 지점을 현재 목표로 설정 (기본)
        """
        new_wp = {
            'x': msg.x, 'y': msg.y,
            'mission_type': MissionType.OBSTACLE_AVOID,
            'radius': Constants.DEFAULT_WAYPOINT_RADIUS,
            'params': {}
        }

        # 웨이포인트가 없으면 첫 목표로 추가
        if not self.waypoint_mgr.waypoints:
            self.waypoint_mgr.waypoints.append(new_wp)
            self.waypoint_mgr.current_waypoint_index = 0
            self._log(f"🎯 첫 웨이포인트 설정: ({msg.y:.1f}, {msg.x:.1f})")
            self.completed = False
            return

        # 모드에 따라 처리
        if self.click_mode == self.CLICK_MODE_GO_IMMEDIATE:
            # 즉시 이동: 현재 웨이포인트를 클릭 지점으로 교체
            insert_idx = self.waypoint_mgr.current_waypoint_index + 1
            self.waypoint_mgr.waypoints.insert(insert_idx, new_wp)
            self.waypoint_mgr.current_waypoint_index = insert_idx
            self._log(f"🎯 즉시 이동 설정 [{insert_idx}]: ({msg.y:.1f}, {msg.x:.1f})")
        else:
            # 다음에 삽입: 현재 목표 완료 후 이동
            insert_idx = self.waypoint_mgr.current_waypoint_index + 1
            self.waypoint_mgr.waypoints.insert(insert_idx, new_wp)
            self._log(f"🎯 다음 목표 삽입 [{insert_idx}]: ({msg.y:.1f}, {msg.x:.1f})")

    def _control_loop(self):
        """제어 루프 (10Hz)"""
        self.loop_count += 1

        # 웨이포인트 없으면 정지
        if not self.waypoint_mgr.get_current_waypoint():
            if not self.completed and self.waypoint_mgr.get_total_waypoints() > 0:
                self.completed = True
                self._log("모든 웨이포인트 완료!")
            self.ros.publish_thrust_commands(0.0, 0.0)
            return

        # 센서 데이터 대기
        if not self.sensors.has_valid_data():
            if self.loop_count % 100 == 0:
                self.get_logger().warn("센서 데이터 대기 중...")
            return

        # 웨이포인트 전환 확인
        if self.wp_handler.check_and_transition(self.sensors.agent_position):
            self.completed = True
            self.ros.publish_thrust_commands(0.0, 0.0)
            return

        # 장애물 회피 실행
        linear, angular = self.avoid_executor.execute(
            self.sensors.agent_position,
            self.sensors.agent_heading,
            self.sensors.lidar_distances,
            self.sensors.get_lidar_distance_at_angle_degrees
        )

        # 스러스터 계산 및 발행
        left, right = self._calc_thrust(linear, angular)
        left, right = self.avoid_executor.avoidance_controller.apply_thrust_filters(left, right)
        self.ros.publish_thrust_commands(left, right)

        # 주기적 상태 로그
        if self.loop_count % 100 == 0:
            wp = self.waypoint_mgr.get_current_waypoint()
            dist = np.linalg.norm(self.sensors.agent_position - [wp['x'], wp['y']])
            self._log(
                f"[{self.waypoint_mgr.get_waypoint_index()}/{self.waypoint_mgr.get_total_waypoints()}] "
                f"거리: {dist:.1f}m | v={linear:.2f}, ω={angular:.2f}"
            )

    def _calc_thrust(self, linear: float, angular: float) -> tuple:
        """선속도/각속도 → 스러스터"""
        fwd = linear * 2000
        turn = angular * 2000
        return (np.clip(fwd + turn, -2000, 2000), np.clip(fwd - turn, -2000, 2000))

    def _log(self, msg: str, header: bool = False):
        """로그 헬퍼"""
        if header:
            self.get_logger().info("=" * 80)
        self.get_logger().info(msg)
        if header:
            self.get_logger().info("=" * 80)

    def destroy_node(self):
        """종료"""
        self.ros.publish_thrust_commands(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(VRXObstacleAvoidController())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
