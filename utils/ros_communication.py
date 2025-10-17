#!/usr/bin/env python3
"""
ROS2 통신 모듈
- 퍼블리셔/서브스크라이버 설정 및 관리
"""

from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Float64MultiArray, String


class ROSCommunicationManager:
    """ROS2 통신 관리 시스템"""

    def __init__(self, node: Node):
        """
        Args:
            node: ROS2 노드 인스턴스
        """
        self.node = node
        self.publishers = {}
        self.subscribers = {}

    def setup_subscribers(self, callbacks: dict):
        """
        ROS2 구독자 설정

        Args:
            callbacks: {topic_name: callback_function} 형태의 딕셔너리
        """
        # 이미지 구독
        if 'image' in callbacks:
            self.subscribers['image'] = self.node.create_subscription(
                Image,
                '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
                callbacks['image'],
                10
            )

        # LiDAR 구독
        if 'lidar' in callbacks:
            self.subscribers['lidar'] = self.node.create_subscription(
                LaserScan,
                '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
                callbacks['lidar'],
                10
            )

        # GPS 구독
        if 'gps' in callbacks:
            self.subscribers['gps'] = self.node.create_subscription(
                NavSatFix,
                '/wamv/sensors/gps/gps/fix',
                callbacks['gps'],
                10
            )

        # IMU 구독
        if 'imu' in callbacks:
            self.subscribers['imu'] = self.node.create_subscription(
                Imu,
                '/wamv/sensors/imu/imu/data',
                callbacks['imu'],
                10
            )

        # 웨이포인트 구독
        if 'waypoint' in callbacks:
            self.subscribers['waypoint'] = self.node.create_subscription(
                Point,
                '/vrx/waypoint',
                callbacks['waypoint'],
                10
            )

    def setup_publishers(self):
        """ROS2 발행자 설정"""
        self.publishers['left_thrust'] = self.node.create_publisher(
            Float64, '/wamv/thrusters/left/thrust', 10
        )
        self.publishers['right_thrust'] = self.node.create_publisher(
            Float64, '/wamv/thrusters/right/thrust', 10
        )
        self.publishers['mission_status'] = self.node.create_publisher(
            String, '/vrx/mission_status', 10
        )
        self.publishers['detections'] = self.node.create_publisher(
            Float64MultiArray, '/vrx/detections', 10
        )
        self.publishers['viz_image'] = self.node.create_publisher(
            Image, '/vrx/visualization', 10
        )

    def publish_thrust_commands(self, left_thrust: float, right_thrust: float):
        """스러스터 명령 발행"""
        left_msg = Float64()
        left_msg.data = float(left_thrust)
        self.publishers['left_thrust'].publish(left_msg)

        right_msg = Float64()
        right_msg.data = float(right_thrust)
        self.publishers['right_thrust'].publish(right_msg)

    def publish_mission_status(self, mission_name: str, waypoint_index: int, total_waypoints: int):
        """미션 상태 발행"""
        status_msg = String()
        status_msg.data = f"Mission: {mission_name}, Waypoint: {waypoint_index}/{total_waypoints}"
        self.publishers['mission_status'].publish(status_msg)

    def publish_detections(self, detections: list):
        """
        탐지 결과 발행

        Args:
            detections: 탐지 결과 리스트
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
