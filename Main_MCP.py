#!/usr/bin/env python3
"""
VRX 통합 미션 제어 시스템 (Main Mission Control Platform)
- NanoOWL 기반 객체 탐지 + MiDaS 깊이 필터링
- 웨이포인트 기반 미션 전환
- 4가지 미션: 부표 사이 지나가기 → 부표 회전 → 웨이포인트 추종 → 장애물 회피

리팩토링됨: 제어 로직을 MissionLoopExecutor로 분리하여 간소화
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from utils.config import Constants
from utils.system_factory import VRXSystemFactory	
from utils.detection_system import MissionType
from utils.mission_control import (
    MissionLoopExecutor,
    WaypointTransitionHandler,
    ObstacleAvoidExecutor
)


class VRXMissionController(Node):
    """VRX 통합 미션 제어 노드 (간소화됨)"""

    def __init__(self):
        super().__init__('vrx_mission_controller')
        self._log_header("VRX 통합 미션 제어 시스템 초기화")

        # 기본 설정
        self.bridge = CvBridge()

        # 🚀 Factory를 사용한 컴포넌트 초기화
        factory = VRXSystemFactory(self, self.bridge, self.get_logger())
        components = factory.create_all_components()

        # 컴포넌트 할당
        self._assign_components(components)

        # ROS2 통신 설정
        self._setup_ros_communication()

        # 🎯 미션 제어 실행자 생성
        self._setup_mission_executors()

        # 타이머 설정
        self.timer = self.create_timer(
            Constants.MAIN_LOOP_PERIOD,
            self.loop_executor.execute_loop
        )

        self.get_logger().info("✓ 초기화 완료!")
        self.get_logger().info("=" * 80)

    def _assign_components(self, components: dict):
        """컴포넌트 할당"""
        # depth_estimator는 detection_system 내부에서만 사용되므로 별도 보관 불필요 (성능 최적화)
        self.detection_system = components['detection_system']
        self.sensor_handler = components['sensor_handler']
        self.avoidance_controller = components['avoidance_controller']
        self.mission_manager = components['mission_manager']
        self.waypoint_manager = components['waypoint_manager']
        self.mission_executor = components['mission_executor']
        self.visualization = components['visualization']
        self.param_manager = components['param_manager']
        self.onnx_controller = components['onnx_controller']
        self.tracker = components['tracker']

    def _setup_ros_communication(self):
        """ROS2 통신 설정"""
        from utils.ros_communication import ROSCommunicationManager

        self.get_logger().info("ROS2 통신 설정 중...")
        self.ros_comm = ROSCommunicationManager(self)

        # 서브스크라이버 설정
        self.ros_comm.setup_subscribers({
            'image': self.sensor_handler.image_callback,
            'lidar': self.sensor_handler.lidar_callback,
            'gps': self.sensor_handler.gps_callback,
            'imu': self.sensor_handler.imu_callback,
            'waypoint': self._waypoint_callback_wrapper
        })

        # 퍼블리셔 설정
        self.ros_comm.setup_publishers()

        self.get_logger().info("✓ ROS2 통신 설정 완료")

    def _setup_mission_executors(self):
        """미션 실행자들 설정"""
        # 웨이포인트 전환 핸들러
        self.waypoint_transition_handler = WaypointTransitionHandler(
            self.waypoint_manager,
            self.mission_manager,
            self.get_logger()
        )

        # 장애물 회피 실행자
        self.obstacle_avoid_executor = ObstacleAvoidExecutor(
            self.avoidance_controller,
            self.onnx_controller,
            self.waypoint_manager,
            self.mission_executor,
            self.sensor_handler,
            self.ros_comm,
            self.get_logger()
        )

        # 메인 루프 실행자
        self.loop_executor = MissionLoopExecutor(
            self.waypoint_manager,
            self.mission_manager,
            self.mission_executor,
            self.detection_system,
            self.tracker,
            self.param_manager,
            self.sensor_handler,
            self.visualization,
            self.ros_comm,
            self.obstacle_avoid_executor,
            self.waypoint_transition_handler,
            self.get_logger()
        )

        # depth_estimator는 시각화에 사용하지 않으므로 전달 불필요 (성능 최적화)
        self.sensor_handler.bridge = self.bridge

    def _waypoint_callback_wrapper(self, msg: Point):
        """웨이포인트 콜백 래퍼"""
        self.sensor_handler.waypoint_callback(msg, self._add_waypoint_from_click)

    def _add_waypoint_from_click(self, msg: Point):
        """클릭한 위치에서 웨이포인트 추가"""
        mission_sequence = [
            MissionType.OBSTACLE_AVOID,
            MissionType.OBSTACLE_AVOID,
            MissionType.PASS_BETWEEN_BUOYS,
            MissionType.OBSTACLE_AVOID
        ]

        waypoint_count = self.waypoint_manager.get_total_waypoints()
        mission_type = mission_sequence[min(waypoint_count, len(mission_sequence) - 1)]

        params = {}
        if mission_type == MissionType.CIRCLE_BUOY:
            params['rotation_direction'] = 1
            params['circle_radius'] = 15.0

        self.waypoint_manager.add_waypoint(
            msg.y, msg.x, mission_type,
            Constants.DEFAULT_WAYPOINT_RADIUS,
            params
        )
        self.get_logger().info(
            f"웨이포인트 추가: {mission_type.name} at ({msg.y:.1f}, {msg.x:.1f})"
        )

    def _log_header(self, message: str):
        """헤더 로그 출력"""
        self.get_logger().info("=" * 80)
        self.get_logger().info(message)
        self.get_logger().info("=" * 80)

    def destroy_node(self):
        """노드 종료"""
        self.ros_comm.publish_thrust_commands(0.0, 0.0)
        self.visualization.cleanup()
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = VRXMissionController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
