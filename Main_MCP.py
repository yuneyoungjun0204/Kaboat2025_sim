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

from utils import Constants, VRXSystemFactory
from utils.detection.detection_system_optimized import MissionType
from utils.mission.mission_control import (
    MissionLoopExecutor,
    WaypointTransitionHandler,
    ObstacleAvoidExecutor
)
from utils.core.jetson_optimizer import setup_jetson


class VRXMissionController(Node):
    """VRX 통합 미션 제어 노드 (간소화됨)"""

    def __init__(self):
        super().__init__('vrx_mission_controller')
        self._log_header("VRX 통합 미션 제어 시스템 초기화")

        # 🚀 Jetson 최적화 비활성화 (시작 시간 단축)
        #setup_jetson()

        # 기본 설정
        self.bridge = CvBridge()

        # 🚀 Factory를 사용한 컴포넌트 초기화 (탐지 시스템은 즉시 로딩)
        factory = VRXSystemFactory(self, self.bridge, self.get_logger())
        components = factory.create_all_components(lazy_load=True)  # 탐지가 안 되므로 즉시 로딩
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

        # GPS 기준점 정보 발행 (trajectory_viz 동기화용)
        self.ros_comm.publish_gps_reference()

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
        from utils.communication.ros_communication import ROSCommunicationManager

        self.get_logger().info("ROS2 통신 설정 중...")
        self.ros_comm = ROSCommunicationManager(self)

        # 기본 서브스크라이버 콜백
        callbacks = {
            'image': self.sensor_handler.image_callback,
            'lidar': self.sensor_handler.lidar_callback,
            'waypoint': self._waypoint_callback_wrapper
        }

        # PX4 모드일 때 픽스호크 센서 사용, 아니면 기존 센서 사용
        if Constants.PX4.ENABLED:
            # PX4 센서 구독 (GPS, IMU를 픽스호크에서 받음)
            callbacks['px4_global_position'] = self.sensor_handler.px4_global_position_callback
            callbacks['px4_local_position'] = self.sensor_handler.px4_local_position_callback
            callbacks['livox_imu'] = self.sensor_handler.livox_imu_callback
            self.get_logger().info("PX4 모드: VehicleGlobalPosition, VehicleLocalPosition, Livox IMU 구독 활성화")
        else:
            # 기존 센서 구독
            callbacks['gps'] = self.sensor_handler.gps_callback
            callbacks['imu'] = self.sensor_handler.imu_callback
            self.get_logger().info("시뮬레이터 모드: 기존 GPS/IMU 구독 활성화")

        # 서브스크라이버 설정
        self.ros_comm.setup_subscribers(callbacks)

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
            MissionType.DOCK_MODE,
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

        current_position = None
        if hasattr(self.sensor_handler, 'agent_position'):
            agent_pos = self.sensor_handler.agent_position
            if agent_pos is not None and len(agent_pos) >= 2:
                current_position = (float(agent_pos[0]), float(agent_pos[1]))

        self.waypoint_manager.add_waypoint(
            msg.y, msg.x, mission_type,
            Constants.DEFAULT_WAYPOINT_RADIUS,
            params,
            current_position=current_position
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