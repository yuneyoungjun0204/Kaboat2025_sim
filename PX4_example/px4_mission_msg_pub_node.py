#!/usr/bin/env python3
"""
PX4 미션 메시지 퍼블리셔 노드
- 기존 시스템(Main_MCP.py)의 출력을 PX4 명령으로 변환
- OffboardControlMode와 TrajectorySetpoint를 PX4로 전송

토픽 구조:
  - 입력: /px4_bridge/velocity_yaw_cmd, /px4_bridge/position_error, /px4_bridge/control_flag
  - 출력: /fmu/in/offboard_control_mode, /fmu/in/trajectory_setpoint
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
from std_msgs.msg import Bool, Float64MultiArray
import numpy as np
from threading import Lock
import math


class MissionMsgPublisherforPx4(Node):
    """
    PX4 Offboard 제어 명령을 전송하는 ROS 2 노드

    기존 제어 시스템의 출력을 구독하여 PX4 형식으로 변환 후 발행
    """

    # --- Constants ---
    # PX4 uORB Topic Names (출력)
    TOPIC_OFFBOARD_CONTROL_MODE = '/fmu/in/offboard_control_mode'
    TOPIC_TRAJECTORY_SETPOINT = '/fmu/in/trajectory_setpoint'

    # 브릿지 토픽 (입력) - 기존 시스템에서 발행
    TOPIC_VELOCITY_YAW_CMD = '/px4_bridge/velocity_yaw_cmd'   # [속도(m/s), yaw(rad)]
    TOPIC_POSITION_ERROR = '/px4_bridge/position_error'       # [x_error(m), y_error(m)]
    TOPIC_CONTROL_FLAG = '/px4_bridge/control_flag'           # Bool (True=위치제어)

    def __init__(self):
        """노드 초기화: 퍼블리셔, 서브스크라이버, 타이머 설정"""
        super().__init__('mission_msg_pub_px4')

        # --- Parameter Declaration ---
        self.declare_parameter('timer_period', 0.02)  # [sec] (Default: 50Hz)
        self.declare_parameter('max_velocity', 2.0)   # [m/s] 최대 전진 속도
        self.declare_parameter('max_yaw_rate', 1.0)   # [rad/s] 최대 yaw rate

        # Get parameter values
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.max_velocity = self.get_parameter('max_velocity').get_parameter_value().double_value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').get_parameter_value().double_value

        # PX4 QoS profile (RELIABLE + TRANSIENT_LOCAL)
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Lock for thread safety
        self.data_lock = Lock()

        # Data initialization
        with self.data_lock:
            self.ctrl_flag = False      # False=속도제어, True=위치제어
            self.velocity = 0.0         # 목표 전진 속도 (m/s)
            self.yaw = 0.0              # 목표 yaw (rad)
            self.x_error = 0.0          # X 위치 오차 (m)
            self.y_error = 0.0          # Y 위치 오차 (m)

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, self.TOPIC_OFFBOARD_CONTROL_MODE, px4_qos)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, self.TOPIC_TRAJECTORY_SETPOINT, px4_qos)

        # Create subscribers
        self.create_subscription(
            Bool, self.TOPIC_CONTROL_FLAG, self.control_flag_callback, 1)
        self.create_subscription(
            Float64MultiArray, self.TOPIC_VELOCITY_YAW_CMD, self.velocity_yaw_callback, 1)
        self.create_subscription(
            Float64MultiArray, self.TOPIC_POSITION_ERROR, self.position_error_callback, 1)

        # Timer for periodic message publishing
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"PX4 Mission Message Publisher initialized at {1.0/timer_period:.1f} Hz\n"
            f"  - Max velocity: {self.max_velocity} m/s\n"
            f"  - Max yaw rate: {self.max_yaw_rate} rad/s"
        )

    # --- Subscriber Callbacks ---
    def control_flag_callback(self, msg: Bool):
        """제어 플래그 콜백 (True=위치제어, False=속도제어)"""
        with self.data_lock:
            self.ctrl_flag = msg.data

    def velocity_yaw_callback(self, msg: Float64MultiArray):
        """속도/yaw 명령 콜백"""
        with self.data_lock:
            if len(msg.data) >= 2:
                self.velocity = msg.data[0]  # m/s
                self.yaw = msg.data[1]       # rad

    def position_error_callback(self, msg: Float64MultiArray):
        """위치 오차 콜백"""
        with self.data_lock:
            if len(msg.data) >= 2:
                self.x_error = msg.data[0]   # m
                self.y_error = msg.data[1]   # m

    # --- Publisher Functions ---
    def publish_offboard_control_mode(self, timestamp: int):
        """OffboardControlMode 메시지 발행"""
        msg = OffboardControlMode()

        with self.data_lock:
            use_position = self.ctrl_flag

        # 속도 제어 모드 (기본)
        msg.position = use_position       # 위치 제어 사용 여부
        msg.velocity = not use_position   # 속도 제어 사용 여부
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = timestamp

        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, timestamp: int):
        """TrajectorySetpoint 메시지 발행"""
        msg = TrajectorySetpoint()

        with self.data_lock:
            velocity = self.velocity
            yaw = self.yaw
            x_error = self.x_error
            y_error = self.y_error
            use_position = self.ctrl_flag

        if use_position:
            # 위치 제어 모드: 위치 오차를 setpoint로 사용
            msg.position = [float(x_error), float(y_error), 0.0]
            msg.velocity = [float('nan'), float('nan'), float('nan')]
        else:
            # 속도 제어 모드: 속도를 setpoint로 사용
            # USV의 경우 x가 전진 방향
            msg.position = [float('nan'), float('nan'), float('nan')]
            msg.velocity = [float(velocity), 0.0, 0.0]  # [vx, vy, vz]

        msg.yaw = float(yaw)
        msg.timestamp = timestamp

        self.trajectory_setpoint_publisher.publish(msg)

    def timer_callback(self):
        """주기적 제어 메시지 발행"""
        current_time_us = int(self.get_clock().now().nanoseconds / 1000)
        self.publish_offboard_control_mode(current_time_us)
        self.publish_trajectory_setpoint(current_time_us)


def main(args=None):
    rclpy.init(args=args)
    node = MissionMsgPublisherforPx4()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        node.get_logger().error(f"Executor failed: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
