#!/usr/bin/env python3
"""
ROS2 통신 모듈
- 퍼블리셔/서브스크라이버 설정 및 관리
"""

from typing import Dict, List, Callable, Any
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Float64MultiArray, String
from .config import Constants


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

        # LiDAR 구독
        if 'lidar' in callbacks:
            self.subscribers['lidar'] = self.node.create_subscription(
                LaserScan,
                Constants.Topics.LIDAR_SCAN,
                callbacks['lidar'],
                Constants.QueueSizes.SENSOR
            )

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
