#!/usr/bin/env python3
"""
ROS2 통신 모듈
- 퍼블리셔/서브스크라이버 설정 및 관리
"""

from typing import Dict, List, Callable, Any
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Float64MultiArray, String, Bool, Int32
from .config import Constants
from rclpy.qos import QoSProfile, ReliabilityPolicy
# PX4 메시지 (optional)
try:
    from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition, VehicleGlobalPosition
    PX4_MSGS_AVAILABLE = True
except ImportError:
    PX4_MSGS_AVAILABLE = False
    VehicleOdometry = None
    VehicleLocalPosition = None
    VehicleGlobalPosition = None


class ROSCommunicationManager:
    """
    ROS2 통신 관리 시스템

    ROS2 퍼블리셔와 서브스크라이버를 중앙에서 관리하고,
    토픽 이름을 config.py에서 가져와 일관성을 유지합니다.

    Attributes:
        node: ROS2 노드 인스턴스
        publishers: 퍼블리셔 딕셔너리
        subscribers: 서브스크라이버 딕셔너리
    """

    def __init__(self, node: Node) -> None:
        """
        Args:
            node: ROS2 노드 인스턴스
        """
        self.node: Node = node
        self.publishers: Dict[str, Any] = {}
        self.subscribers: Dict[str, Any] = {}

    def setup_subscribers(self, callbacks: Dict[str, Callable]) -> None:
        """
        ROS2 구독자 설정

        Args:
            callbacks: {topic_name: callback_function} 형태의 딕셔너리
                지원하는 topic_name: 'image', 'lidar', 'gps', 'imu', 'waypoint'
        """
        # 이미지 구독
        if 'image' in callbacks:
            self.subscribers['image'] = self.node.create_subscription(
                Image,
                Constants.Topics.CAMERA_IMAGE,
                callbacks['image'],
                Constants.QueueSizes.SENSOR
            )

        # LiDAR 구독 (선택적)
        if 'lidar' in callbacks:
            try:
                lidar_qos_profile = QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST
                )

                self.subscribers['lidar'] = self.node.create_subscription(
                    LaserScan,
                    Constants.Topics.LIDAR_SCAN,
                    callbacks['lidar'],
                    lidar_qos_profile
                )
            except Exception as e:
                self.node.get_logger().warn(f"⚠️ LiDAR 구독 실패: {e}. 계속 진행합니다.")

        # GPS 구독
        if 'gps' in callbacks:
            self.subscribers['gps'] = self.node.create_subscription(
                NavSatFix,
                Constants.Topics.GPS_FIX,
                callbacks['gps'],
                Constants.QueueSizes.SENSOR
            )

        # IMU 구독
        if 'imu' in callbacks:
            self.subscribers['imu'] = self.node.create_subscription(
                Imu,
                Constants.Topics.IMU_DATA,
                callbacks['imu'],
                Constants.QueueSizes.SENSOR
            )

        # 웨이포인트 구독
        if 'waypoint' in callbacks:
            self.subscribers['waypoint'] = self.node.create_subscription(
                Point,
                Constants.Topics.WAYPOINT,
                callbacks['waypoint'],
                Constants.QueueSizes.DEFAULT
            )

        # PX4 Odometry 구독 (각속도, heading 등)
        if 'px4_odometry' in callbacks and Constants.PX4.ENABLED and PX4_MSGS_AVAILABLE:
            # PX4 QoS 설정
            px4_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.subscribers['px4_odometry'] = self.node.create_subscription(
                VehicleOdometry,
                Constants.Topics.PX4_VEHICLE_ODOMETRY,
                callbacks['px4_odometry'],
                px4_qos
            )

        # PX4 VehicleLocalPosition 구독 (NED 위치, 속도, 가속도, heading)
        if 'px4_local_position' in callbacks and Constants.PX4.ENABLED and PX4_MSGS_AVAILABLE:
            px4_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.subscribers['px4_local_position'] = self.node.create_subscription(
                VehicleLocalPosition,
                Constants.Topics.PX4_VEHICLE_LOCAL_POSITION_SUB,
                callbacks['px4_local_position'],
                px4_qos
            )

        # Livox LiDAR IMU 구독 (각속도 데이터)
        if 'livox_imu' in callbacks:
            livox_imu_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.subscribers['livox_imu'] = self.node.create_subscription(
                Imu,
                Constants.Topics.LIVOX_IMU,
                callbacks['livox_imu'],
                livox_imu_qos
            )

        # PX4 VehicleGlobalPosition 구독 (GPS 대체)
        if 'px4_global_position' in callbacks and Constants.PX4.ENABLED and PX4_MSGS_AVAILABLE:
            px4_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.subscribers['px4_global_position'] = self.node.create_subscription(
                VehicleGlobalPosition,
                Constants.Topics.PX4_VEHICLE_GLOBAL_POSITION,
                callbacks['px4_global_position'],
                px4_qos
            )

    def setup_publishers(self) -> None:
        """
        ROS2 발행자 설정

        자동으로 모든 필요한 퍼블리셔를 생성합니다:
        - left_thrust, right_thrust: 스러스터 제어
        - mission_status: 미션 상태
        - detections: 객체 탐지 결과
        - viz_image: 시각화 이미지
        - control_output: 제어 출력값
        - control_mode: 제어 모드
        - obstacle_check_area: 장애물 체크 영역
        - los_target: LOS 타겟 위치
        """
        self.publishers['left_thrust'] = self.node.create_publisher(
            Float64, Constants.Topics.LEFT_THRUST, Constants.QueueSizes.CONTROL
        )
        self.publishers['right_thrust'] = self.node.create_publisher(
            Float64, Constants.Topics.RIGHT_THRUST, Constants.QueueSizes.CONTROL
        )
        self.publishers['left_pos'] = self.node.create_publisher(
            Float64, Constants.Topics.LEFT_POS, Constants.QueueSizes.CONTROL
        )
        self.publishers['right_pos'] = self.node.create_publisher(
            Float64, Constants.Topics.RIGHT_POS, Constants.QueueSizes.CONTROL
        )
        self.publishers['mission_status'] = self.node.create_publisher(
            String, Constants.Topics.MISSION_STATUS, Constants.QueueSizes.STATUS
        )
        self.publishers['detections'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.DETECTIONS, Constants.QueueSizes.STATUS
        )
        self.publishers['detection_depths'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.DETECTION_DEPTHS, Constants.QueueSizes.STATUS
        )
        self.publishers['viz_image'] = self.node.create_publisher(
            Image, Constants.Topics.VISUALIZATION, Constants.QueueSizes.DEFAULT
        )
        # trajectory_viz용 제어 출력값 및 모드 정보
        self.publishers['control_output'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.CONTROL_OUTPUT, Constants.QueueSizes.STATUS
        )
        self.publishers['control_mode'] = self.node.create_publisher(
            String, Constants.Topics.CONTROL_MODE, Constants.QueueSizes.STATUS
        )
        self.publishers['obstacle_check_area'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.OBSTACLE_CHECK_AREA, Constants.QueueSizes.STATUS
        )
        self.publishers['los_target'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.LOS_TARGET, Constants.QueueSizes.STATUS
        )
        self.publishers['target_depth'] = self.node.create_publisher(
            Float64, Constants.Topics.TARGET_DEPTH, Constants.QueueSizes.STATUS
        )
        # 통합 제어 명령 퍼블리셔 (배 독립적)
        self.publishers['desired_speed'] = self.node.create_publisher(
            Float64, Constants.Topics.DESIRED_SPEED, Constants.QueueSizes.CONTROL
        )
        self.publishers['desired_moment'] = self.node.create_publisher(
            Float64, Constants.Topics.DESIRED_MOMENT, Constants.QueueSizes.CONTROL
        )
        self.publishers['desired_force_y'] = self.node.create_publisher(
            Float64, Constants.Topics.DESIRED_FORCE_Y, Constants.QueueSizes.CONTROL
        )

        # 장애물 회피용 퍼블리셔
        self.publishers['avoid_yaw'] = self.node.create_publisher(
            Int32, Constants.Topics.AVOID_YAW, Constants.QueueSizes.CONTROL
        )

        # PX4 브릿지 퍼블리셔 (항상 활성화)
        self.publishers['px4_velocity_yaw'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.PX4_VELOCITY_YAW_CMD, Constants.QueueSizes.CONTROL
        )
        self.publishers['px4_position_error'] = self.node.create_publisher(
            Float64MultiArray, Constants.Topics.PX4_POSITION_ERROR, Constants.QueueSizes.CONTROL
        )
        self.publishers['px4_control_flag'] = self.node.create_publisher(
            Bool, Constants.Topics.PX4_CONTROL_FLAG, Constants.QueueSizes.CONTROL
        )

    def publish_thrust_commands(self, left_thrust: float, right_thrust: float) -> None:
        """
        스러스터 명령 발행

        Args:
            left_thrust: 좌측 스러스터 명령 (-2000 ~ 2000)
            right_thrust: 우측 스러스터 명령 (-2000 ~ 2000)
        """
        left_msg = Float64()
        left_msg.data = float(left_thrust)
        self.publishers['left_thrust'].publish(left_msg)

        right_msg = Float64()
        right_msg.data = float(right_thrust)
        self.publishers['right_thrust'].publish(right_msg)

    def publish_mission_status(self, mission_name: str, waypoint_index: int, total_waypoints: int) -> None:
        """
        미션 상태 발행

        Args:
            mission_name: 현재 미션 이름
            waypoint_index: 현재 웨이포인트 인덱스
            total_waypoints: 전체 웨이포인트 개수
        """
        status_msg = String()
        status_msg.data = f"Mission: {mission_name}, Waypoint: {waypoint_index}/{total_waypoints}"
        self.publishers['mission_status'].publish(status_msg)

    def publish_detections(self, detections: List[Dict[str, Any]]) -> None:
        """
        탐지 결과 발행

        Args:
            detections: 탐지 결과 리스트
                각 항목: {'label': str, 'confidence': float, 'bbox': [x1, y1, x2, y2], 'depth': float}
        """
        msg = Float64MultiArray()
        data = [float(len(detections))]

        # 라벨 ID 매핑
        label_to_id = {
            "red_cone": 0,
            "green_cone": 1,
            "blue_buoy": 2
        }

        for det in detections:
            label_id = label_to_id.get(det['label'], -1)
            data.extend([
                float(label_id),
                float(det.get('confidence', 0.0)),
                float(det['bbox'][0]),
                float(det['bbox'][1]),
                float(det['bbox'][2]),
                float(det['bbox'][3]),
                float(det['depth'])
            ])

        msg.data = data
        self.publishers['detections'].publish(msg)

    def publish_detection_depths(self, detections: List[Dict[str, Any]]) -> None:
        """
        탐지된 객체들의 depth 정보 발행

        Args:
            detections: 탐지 결과 리스트
                각 항목: {'label': str, 'depth': float, 'center': [x, y]}

        발행 포맷:
            [count, label_id_1, depth_1, center_x_1, center_y_1, label_id_2, depth_2, ...]
        """
        msg = Float64MultiArray()
        data = [float(len(detections))]

        # 라벨 ID 매핑
        label_to_id = {
            "red_cone": 0,
            "green_cone": 1,
            "blue_buoy": 2,
            "red_circle": 3,
            "blue_circle": 4,
            "green_circle": 5,
            "red_triangle": 6,
            "blue_triangle": 7,
            "green_triangle": 8,
            "red_cross": 9,
            "blue_cross": 10,
            "green_cross": 11
        }

        for det in detections:
            label_id = label_to_id.get(det.get('label', ''), -1)
            depth = det.get('depth', 0.0)
            center = det.get('center', [0.0, 0.0])

            data.extend([
                float(label_id),
                float(depth),
                float(center[0]),
                float(center[1])
            ])

        msg.data = data
        self.publishers['detection_depths'].publish(msg)

    def publish_control_output(self, linear_velocity: float, angular_velocity: float) -> None:
        """
        제어 출력값 발행 (ONNX 모델 또는 직접 제어)

        Args:
            linear_velocity: 선속도 (m/s)
            angular_velocity: 각속도 (rad/s)
        """
        msg = Float64MultiArray()
        msg.data = [float(linear_velocity), float(angular_velocity)]
        self.publishers['control_output'].publish(msg)

    def publish_control_mode(self, mode: str) -> None:
        """
        제어 모드 발행

        Args:
            mode: 제어 모드
                - "ONNX_MODEL": ONNX 모델 사용
                - "DIRECT_CONTROL": 직접 제어 (LOS 기반)
                - "BUOY_MISSION": 부표 미션 제어
                - "CIRCLE_MISSION": 부표 회전 미션 제어
        """
        msg = String()
        msg.data = mode
        self.publishers['control_mode'].publish(msg)

    def publish_obstacle_check_area(self, area_points: List[tuple]) -> None:
        """
        장애물 체크 영역 발행

        Args:
            area_points: 체크 영역 점들 [(x1, y1), (x2, y2), ...]
                각 점은 UTM 좌표계 (미터 단위)
        """
        msg = Float64MultiArray()
        data = []
        for point in area_points:
            data.extend([float(point[0]), float(point[1])])
        msg.data = data
        self.publishers['obstacle_check_area'].publish(msg)

    def publish_los_target(self, target_x: float, target_y: float) -> None:
        """
        LOS (Line-of-Sight) 타겟 위치 발행

        Args:
            target_x: 타겟 X 좌표 (UTM Easting, 미터)
            target_y: 타겟 Y 좌표 (UTM Northing, 미터)
        """
        msg = Float64MultiArray()
        msg.data = [float(target_x), float(target_y)]
        self.publishers['los_target'].publish(msg)

    def publish_avoid_yaw(self, value: int) -> None:
        """
        장애물 회피 yaw 상태 발행 (ONNX 모드가 아닐 때 0, 1 번갈아 발행)

        Args:
            value: 0 또는 1
        """
        from std_msgs.msg import Int32
        msg = Int32()
        msg.data = value
        self.publishers['avoid_yaw'].publish(msg)

    def publish_thruster_positions(self, left_pos: float, right_pos: float) -> None:
        """
        스러스터 각도 명령 발행

        Args:
            left_pos: 좌측 스러스터 각도 (라디안, -π/2 ~ π/2)
            right_pos: 우측 스러스터 각도 (라디안, -π/2 ~ π/2)
        """
        left_msg = Float64()
        left_msg.data = float(left_pos)
        self.publishers['left_pos'].publish(left_msg)

        right_msg = Float64()
        right_msg.data = float(right_pos)
        self.publishers['right_pos'].publish(right_msg)

    def publish_target_depth(self, depth: float) -> None:
        """
        목표 객체의 깊이 값 발행 (Dock_mode용)

        Args:
            depth: 목표 객체까지의 깊이 (0-1 스케일, 1=가까움, 0=멀리)
        """
        msg = Float64()
        msg.data = float(depth)
        self.publishers['target_depth'].publish(msg)

    def publish_desired_control(self, desired_speed: float, desired_moment: float,
                               desired_force_y: float, current_yaw: float = 0.0,
                               is_dock_mode: bool = False) -> None:
        """
        통합 제어 명령 발행 (배 독립적) + PX4 브릿지 명령 동시 발행

        Args:
            desired_speed: Surge velocity (-1~1)
            desired_moment: Yaw moment (-1~1)
            desired_force_y: Sway force (-1~1)
            current_yaw: 현재 yaw 각도 (rad), PX4 목표 yaw 계산용
            is_dock_mode: 도킹 미션 여부 (True일 때 position_error 발행)
        """
        # VRX 토픽 발행
        speed_msg = Float64()
        speed_msg.data = float(desired_speed)
        self.publishers['desired_speed'].publish(speed_msg)

        moment_msg = Float64()
        moment_msg.data = float(desired_moment)
        self.publishers['desired_moment'].publish(moment_msg)

        force_y_msg = Float64()
        force_y_msg.data = float(desired_force_y)
        self.publishers['desired_force_y'].publish(force_y_msg)

        # PX4 브릿지 명령 발행 (항상 발행)
        import numpy as np
        import math

        # desired_speed → velocity (m/s)
        velocity = float(desired_speed) * Constants.PX4.VELOCITY_SCALE
        velocity = np.clip(velocity, Constants.PX4.MIN_VELOCITY, Constants.PX4.MAX_VELOCITY)

        # desired_moment → 목표 yaw 계산 (현재 yaw + moment * scale)
        yaw_delta = float(desired_moment) * Constants.PX4.YAW_RATE_SCALE * 0.02  # dt ≈ 0.02s (50Hz)
        target_yaw = current_yaw + yaw_delta
        # [-pi, pi] 범위로 정규화
        while target_yaw > math.pi:
            target_yaw -= 2 * math.pi
        while target_yaw < -math.pi:
            target_yaw += 2 * math.pi

        # 도킹 미션: position_error만 발행 (control_flag=True 유지)
        if is_dock_mode:
            # desired_speed → x_error (m)
            x_error = float(desired_speed) * 1.0  # 1.0m 스케일
            # desired_force_y → y_error (m)
            y_error = float(desired_force_y) * 1.0  # 1.0m 스케일
            self.publish_px4_position_command(x_error, y_error)
            # 위치 제어 모드이므로 velocity/yaw 명령은 발행하지 않음
        else:
            # 일반 모드: velocity/yaw만 발행
            self.publish_px4_velocity_command(velocity, target_yaw)

    # =========================================================================
    # PX4 브릿지 퍼블리셔 메서드
    # =========================================================================

    def publish_px4_velocity_command(self, velocity: float, yaw: float) -> None:
        """
        PX4 속도/yaw 명령 발행 (항상 발행)

        Args:
            velocity: 전진 속도 (m/s)
            yaw: 목표 yaw (rad)
        """
        msg = Float64MultiArray()
        msg.data = [float(velocity), float(yaw)]
        self.publishers['px4_velocity_yaw'].publish(msg)

        # 제어 플래그: 속도 제어 모드
        flag_msg = Bool()
        flag_msg.data = False
        self.publishers['px4_control_flag'].publish(flag_msg)

    def publish_px4_position_command(self, x_error: float, y_error: float) -> None:
        """
        PX4 위치 오차 명령 발행 (항상 발행)

        Args:
            x_error: X 방향 위치 오차 (m)
            y_error: Y 방향 위치 오차 (m)
        """
        msg = Float64MultiArray()
        msg.data = [float(x_error), float(y_error)]
        self.publishers['px4_position_error'].publish(msg)

        # 제어 플래그: 위치 제어 모드
        flag_msg = Bool()
        flag_msg.data = True
        self.publishers['px4_control_flag'].publish(flag_msg)

    def publish_px4_commands(
        self,
        velocity: float = None,
        yaw: float = None,
        x_error: float = None,
        y_error: float = None,
        use_position_control: bool = False
    ) -> None:
        """
        PX4 통합 명령 발행 (항상 발행)

        Args:
            velocity: 전진 속도 (m/s), 속도 제어 시 사용
            yaw: 목표 yaw (rad)
            x_error: X 위치 오차 (m), 위치 제어 시 사용
            y_error: Y 위치 오차 (m), 위치 제어 시 사용
            use_position_control: True=위치제어, False=속도제어
        """
        if use_position_control:
            self.publish_px4_position_command(
                x_error if x_error is not None else 0.0,
                y_error if y_error is not None else 0.0
            )
        else:
            self.publish_px4_velocity_command(
                velocity if velocity is not None else 0.0,
                yaw if yaw is not None else 0.0
            )
