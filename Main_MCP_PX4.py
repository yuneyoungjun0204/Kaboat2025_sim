#!/usr/bin/env python3
"""
VRX 통합 미션 제어 시스템 - PX4 버전
- PX4/Pixhawk와 연동하여 실제 하드웨어에서 동작
- 기존 Main_MCP.py의 기능을 유지하면서 PX4 센서/제어 통합

실행 방법:
1. PX4 브릿지 노드 실행 (px4_mission_msg_pub_node.py)
2. 이 노드 실행 (Main_MCP_PX4.py)

토픽 구조:
- 센서 입력: PX4 토픽 (/fmu/out/*)
- 제어 출력: PX4 브릿지 토픽 (/px4_bridge/*)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np

# PX4 메시지
try:
    from px4_msgs.msg import VehicleGlobalPosition, VehicleLocalPosition
    PX4_MSGS_AVAILABLE = True
except ImportError:
    PX4_MSGS_AVAILABLE = False
    print("Warning: px4_msgs not found. PX4 sensor subscription disabled.")

from utils import Constants, VRXSystemFactory
from utils.detection_system import MissionType
from utils.mission_control import (
    MissionLoopExecutor,
    WaypointTransitionHandler,
    ObstacleAvoidExecutor
)
from utils.px4_adapter import PX4SensorAdapter, CoordinateConverter


class VRXMissionControllerPX4(Node):
    """VRX 통합 미션 제어 노드 - PX4 버전"""

    def __init__(self):
        super().__init__('vrx_mission_controller_px4')
        self._log_header("VRX 통합 미션 제어 시스템 (PX4) 초기화")

        # 기본 설정
        self.bridge = CvBridge()

        # PX4 어댑터 초기화
        self.px4_adapter = PX4SensorAdapter()
        self.coord_converter = CoordinateConverter()

        # Factory를 사용한 컴포넌트 초기화
        factory = VRXSystemFactory(self, self.bridge, self.get_logger())
        components = factory.create_all_components()

        # 컴포넌트 할당
        self._assign_components(components)

        # ROS2 통신 설정 (기존 + PX4)
        self._setup_ros_communication()

        # PX4 센서 구독 설정
        self._setup_px4_subscribers()

        # 미션 제어 실행자 생성
        self._setup_mission_executors()

        # 타이머 설정
        self.timer = self.create_timer(
            Constants.MAIN_LOOP_PERIOD,
            self.loop_executor.execute_loop
        )

        self.get_logger().info("PX4 mode: ENABLED")
        self.get_logger().info("Initialization complete!")
        self.get_logger().info("=" * 80)

    def _assign_components(self, components: dict):
        """컴포넌트 할당"""
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

        self.get_logger().info("Setting up ROS2 communication...")
        self.ros_comm = ROSCommunicationManager(self)

        # 기존 센서 서브스크라이버 (카메라, 웨이포인트 등)
        # GPS/IMU는 PX4에서 받으므로 제외
        callbacks = {
            'image': self.sensor_handler.image_callback,
            'waypoint': self._waypoint_callback_wrapper
        }

        # LiDAR는 별도로 설정 (PointCloud2 또는 LaserScan)
        self.ros_comm.setup_subscribers(callbacks)

        # 퍼블리셔 설정
        self.ros_comm.setup_publishers()

        self.get_logger().info("ROS2 communication setup complete")

    def _setup_px4_subscribers(self):
        """PX4 센서 구독 설정"""
        if not PX4_MSGS_AVAILABLE:
            self.get_logger().warn("PX4 messages not available. Skipping PX4 sensor setup.")
            return

        # PX4 QoS 프로파일
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Livox LiDAR QoS
        livox_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # VehicleGlobalPosition (GPS)
        self.create_subscription(
            VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self._px4_global_position_callback,
            px4_qos
        )

        # VehicleLocalPosition (heading, velocity)
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self._px4_local_position_callback,
            px4_qos
        )

        # Livox LiDAR (PointCloud2)
        self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self._px4_lidar_callback,
            livox_qos
        )

        self.get_logger().info("PX4 sensor subscribers created")

    def _px4_global_position_callback(self, msg):
        """PX4 Global Position 콜백"""
        # NED 좌표로 변환
        ned_pos = self.px4_adapter.process_global_position(msg.lat, msg.lon, msg.alt)

        # sensor_handler에 위치 업데이트
        # UTM 좌표계로 변환 (기존 시스템과 호환)
        self.sensor_handler.agent_position = np.array(
            [ned_pos.east, ned_pos.north],  # [x, y] = [East, North]
            dtype=np.float32
        )

    def _px4_local_position_callback(self, msg):
        """PX4 Local Position 콜백"""
        # heading과 velocity 업데이트
        self.px4_adapter.process_local_position(msg.heading, msg.vx, msg.vy, msg.vz)

        # sensor_handler에 헤딩 업데이트 (도 단위)
        self.sensor_handler.agent_heading = self.px4_adapter.get_heading_deg()

        # angular velocity 업데이트 (도/초)
        # PX4에서는 직접적인 angular velocity가 없으므로 추정 필요
        # 현재는 0으로 설정 (IMU에서 별도로 받을 수 있음)
        # self.sensor_handler.angular_velocity_y = 0.0

    def _px4_lidar_callback(self, msg):
        """PX4 LiDAR (PointCloud2) 콜백"""
        try:
            # PointCloud2에서 점 추출
            points_generator = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
            points = np.array(
                [[p[0], p[1], p[2]] for p in points_generator],
                dtype=np.float32
            )

            if points.shape[0] == 0:
                return

            # LaserScan 형식으로 변환
            ranges = self.px4_adapter.convert_pointcloud_to_laserscan(
                points,
                angle_min=-np.pi,
                angle_max=np.pi,
                angle_increment=np.radians(1.0),
                range_min=0.1,
                range_max=Constants.MAX_LIDAR_DISTANCE,
                height_min=-0.5,
                height_max=2.0
            )

            # LiDAR 각도 범위에 맞게 자르기 (-100 ~ +100도)
            full_size = len(ranges)
            center = full_size // 2
            half_range = Constants.LIDAR_ARRAY_SIZE // 2

            start_idx = center - half_range
            end_idx = start_idx + Constants.LIDAR_ARRAY_SIZE

            if start_idx >= 0 and end_idx <= full_size:
                self.sensor_handler.lidar_distances = ranges[start_idx:end_idx]
            else:
                # 범위 초과 시 전체 사용
                self.sensor_handler.lidar_distances = ranges[:Constants.LIDAR_ARRAY_SIZE]

        except Exception as e:
            self.get_logger().error(f"LiDAR callback error: {e}")

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

        self.waypoint_manager.add_waypoint(
            msg.y, msg.x, mission_type,
            Constants.DEFAULT_WAYPOINT_RADIUS,
            params
        )
        self.get_logger().info(
            f"Waypoint added: {mission_type.name} at ({msg.y:.1f}, {msg.x:.1f})"
        )

    def _log_header(self, message: str):
        """헤더 로그 출력"""
        self.get_logger().info("=" * 80)
        self.get_logger().info(message)
        self.get_logger().info("=" * 80)

    def destroy_node(self):
        """노드 종료"""
        # 정지 명령 발행
        self.ros_comm.publish_thrust_commands(0.0, 0.0)
        if Constants.PX4.ENABLED:
            self.ros_comm.publish_px4_velocity_command(0.0, 0.0)

        self.visualization.cleanup()
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = VRXMissionControllerPX4()

        # MultiThreadedExecutor 사용 (PX4 콜백 처리)
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard Interrupt received. Shutting down...')
        finally:
            executor.shutdown()
            node.destroy_node()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
