#!/usr/bin/env python3
"""
PX4 Offboard Bridge Node
========================
현재 시스템의 제어 명령을 PX4 Pixhawk Offboard 제어 메시지로 변환하는 브리지 노드

구독 토픽:
  - /px4_bridge/control_flag (Bool): 제어 모드 (True=위치, False=속도)
  - /px4_bridge/velocity_yaw_cmd (Float64MultiArray): [velocity, yaw]
  - /px4_bridge/position_error (Float64MultiArray): [x_error, y_error]

발행 토픽:
  - /fmu/in/offboard_control_mode (OffboardControlMode): Offboard 제어 모드
  - /fmu/in/trajectory_setpoint (TrajectorySetpoint): 궤적 설정점

참고:
  - PX4 공식 문서: https://docs.px4.io/main/en/ros2/offboard_control.html
  - NED 좌표계 사용 (North-East-Down)
  - 50Hz 이상 퍼블리시 필요 (권장)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from threading import Lock
import numpy as np

from std_msgs.msg import Bool, Float64MultiArray
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition


class PX4OffboardBridge(Node):
    """
    현재 제어 시스템과 PX4 Pixhawk 간의 브리지 노드

    제어 모드:
      1. 속도 제어 모드 (control_flag = False):
         - velocity를 사용하여 전진 속도 제어
         - yaw를 사용하여 방향 제어

      2. 위치 제어 모드 (control_flag = True):
         - position_error를 사용하여 상대 위치 제어
         - 현재 위치 기준 오차를 보상
    """

    # 상수
    PUBLISH_RATE_HZ = 50.0  # PX4는 최소 2Hz, 권장 50Hz 이상
    NAN = float('nan')

    def __init__(self):
        super().__init__('px4_offboard_bridge')

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )


        # === QoS 프로파일 설정 ===
        # PX4는 RELIABLE + TRANSIENT_LOCAL 사용
        self.px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 일반 토픽용 QoS (RELIABLE)
        self.standard_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # === 데이터 보호용 Lock ===
        self.data_lock = Lock()

        # === 제어 상태 변수 ===
        with self.data_lock:
            # 제어 모드 (True=위치, False=속도)
            self.use_position_control = False

            # 속도 제어 명령
            self.velocity_cmd = 0.0  # 전진 속도 (m/s)
            self.yaw_cmd = 0.0       # 목표 yaw (rad)

            # 위치 제어 명령
            self.x_error = 0.0  # X 방향 위치 오차 (m, NED: North)
            self.y_error = 0.0  # Y 방향 위치 오차 (m, NED: East)

            # 명령 수신 플래그
            self.command_received = False
            self.yaw=0.0

        # === 구독자 설정 ===
        self.create_subscription(
            Bool,
            '/px4_bridge/control_flag',
            self.control_flag_callback,
            self.standard_qos
        )

        self.create_subscription(
            Float64MultiArray,
            '/px4_bridge/velocity_yaw_cmd',
            self.velocity_yaw_callback,
            self.standard_qos
        )

        self.create_subscription(
            Float64MultiArray,
            '/px4_bridge/position_error',
            self.position_error_callback,
            self.standard_qos
        )

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.yaw_calback,
            self.qos_profile
        )

        # === 퍼블리셔 설정 ===
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            self.px4_qos
        )

        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            self.px4_qos
        )
        # self.trajectory_setpoint_pub = self.create_publisher(
        #     TrajectorySetpoint,
        #     '/fmu/out/trajectory_setpoint',
        #     self.px4_qos
        # )

        # === 타이머 설정 (50Hz) ===
        timer_period = 1.0 / self.PUBLISH_RATE_HZ
        self.timer = self.create_timer(timer_period, self.publish_control_commands)

        self.get_logger().info('=' * 60)
        self.get_logger().info('PX4 Offboard Bridge 시작')
        self.get_logger().info(f'발행 주기: {self.PUBLISH_RATE_HZ} Hz')
        self.get_logger().info('구독 토픽:')
        self.get_logger().info('  - /px4_bridge/control_flag')
        self.get_logger().info('  - /px4_bridge/velocity_yaw_cmd')
        self.get_logger().info('  - /px4_bridge/position_error')
        self.get_logger().info('발행 토픽:')
        self.get_logger().info('  - /fmu/in/offboard_control_mode')
        self.get_logger().info('  - /fmu/in/trajectory_setpoint')
        self.get_logger().info('=' * 60)

    # =========================================================================
    # 콜백 함수들
    # =========================================================================

    def control_flag_callback(self, msg: Bool):
        """
        제어 모드 플래그 콜백

        Args:
            msg.data: True=위치제어, False=속도제어
        """
        with self.data_lock:
            prev_mode = self.use_position_control
            self.use_position_control = msg.data

            # 모드 변경 시 로그
            if prev_mode != self.use_position_control:
                mode_name = "위치 제어" if self.use_position_control else "속도 제어"
                self.get_logger().info(f'제어 모드 변경: {mode_name}')

    def velocity_yaw_callback(self, msg: Float64MultiArray):
        """
        속도/Yaw 명령 콜백

        Args:
            msg.data: [velocity (m/s), yaw (rad)]
        """
        if len(msg.data) < 2:
            self.get_logger().warn('velocity_yaw_cmd: 데이터 길이 부족')
            return

        with self.data_lock:
            self.velocity_cmd = float(msg.data[0])
            self.yaw_cmd = float(msg.data[1])
            self.command_received = True

    def position_error_callback(self, msg: Float64MultiArray):
        """
        위치 오차 명령 콜백

        Args:
            msg.data: [x_error (m), y_error (m)]
        """
        if len(msg.data) < 2:
            self.get_logger().warn('position_error: 데이터 길이 부족')
            return

        with self.data_lock:
            self.x_error = float(msg.data[0])
            self.y_error = float(msg.data[1])
            self.command_received = True

    # =========================================================================
    # 메인 퍼블리시 함수
    # =========================================================================


    def yaw_calback(self, msg: VehicleLocalPosition):
        self.yaw=msg.heading

    # =========================================================================
    # 메인 퍼블리시 함수
    # =========================================================================

    def publish_control_commands(self):
        """
        PX4 Offboard 제어 명령 발행 (50Hz)

        항상 OffboardControlMode와 TrajectorySetpoint를 쌍으로 발행해야 함
        """
        # 현재 시간 (마이크로초)
        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        with self.data_lock:
            use_position = self.use_position_control

            # 속도 제어 명령
            velocity = self.velocity_cmd
            yaw = self.yaw_cmd

            # 위치 제어 명령
            x_err = self.x_error
            y_err = self.y_error

            cmd_received = self.command_received

        # 명령이 수신되지 않았으면 기본값으로 발행 (PX4 연결 유지)
        if not cmd_received:
            self.publish_offboard_control_mode(timestamp, position=False, velocity=True)
            self.publish_velocity_setpoint(timestamp, 0.0, 0.0)
            return

        # === 제어 모드에 따라 메시지 발행 ===
        if use_position:
            # 위치 제어 모드
            self.publish_offboard_control_mode(timestamp, position=True, velocity=False)
            self.publish_position_setpoint(timestamp, x_err, y_err, yaw)
        else:
            # 속도 제어 모드
            self.publish_offboard_control_mode(timestamp, position=False, velocity=True)
            self.publish_velocity_setpoint(timestamp, velocity, yaw)

    # =========================================================================
    # PX4 메시지 발행 함수들
    # =========================================================================

    def publish_offboard_control_mode(
        self,
        timestamp: int,
        position: bool = False,
        velocity: bool = False
    ):
        """
        OffboardControlMode 메시지 발행

        Args:
            timestamp: 타임스탬프 (마이크로초)
            position: 위치 제어 활성화
            velocity: 속도 제어 활성화
        """
        msg = OffboardControlMode()
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = timestamp

        self.offboard_control_mode_pub.publish(msg)

    def publish_velocity_setpoint(self, timestamp: int, velocity: float, yaw: float):
        """
        속도 제어 setpoint 발행

        Args:
            timestamp: 타임스탬프 (마이크로초)
            velocity: 전진 속도 (m/s)
            yaw: 목표 yaw 각도 (rad)

        Note:
            - NED 좌표계 사용
            - velocity는 North 방향 속도로 변환
            - PX4는 body frame 속도를 자동으로 처리
        """
        msg = TrajectorySetpoint()

        # 위치는 사용하지 않음 (NaN)RELIABILITY
        # velocity는 전진 속도, yaw 방향을 고려하여 NED로 변환
        vx = velocity 
        vy = 0.0
        vz = 0.0                     # Down 방향 속도 (수상정은 0)

        msg.velocity = [vx, 0.0, 0.0]

        # 가속도는 사용하지 않음 (NaN)
        msg.acceleration = [0.0, 0.0, 0.0]

        # Yaw 설정
        msg.yaw = 1000*yaw+self.yaw  # [-π, π]
        msg.yawspeed = 0.0

        msg.timestamp = timestamp

        self.trajectory_setpoint_pub.publish(msg)

        # 로그 (1초마다)
        self.get_logger().info(
            f'[속도 제어] v={velocity:.2f} m/s, yaw={np.degrees(yaw):.1f}°, '
            f'vx={vx:.2f}, vy={vy:.2f}',
            throttle_duration_sec=1.0
        )

    def publish_position_setpoint(
        self,
        timestamp: int,
        x_error: float,
        y_error: float,
        yaw: float
    ):
        """
        위치 제어 setpoint 발행

        Args:
            timestamp: 타임스탬프 (마이크로초)
            x_error: X 방향 위치 오차 (m, NED: North)
            y_error: Y 방향 위치 오차 (m, NED: East)
            yaw: 목표 yaw 각도 (rad)

        Note:
            - NED 좌표계 사용
            - x_error, y_error는 현재 위치 기준 상대 오차
            - PX4는 현재 위치를 기준으로 목표 위치를 계산
        """
        msg = TrajectorySetpoint()

        # 위치 설정 (NED 프레임, 상대 위치)
        # 오차를 목표 위치로 사용 (현재 위치 + 오차 = 목표 위치)
        msg.position = [x_error, y_error, 0.0]  # [North, East, Down]

        # 속도는 feedforward로 사용 (여기서는 사용 안 함)
        # msg.velocity = [self.NAN, self.NAN, self.NAN]
        msg.velocity = [0.0, 0.0, 0.0]

        # 가속도는 사용하지 않음
        msg.acceleration = [0.0, 0.0, 0.0]

        # Yaw 설정
        msg.yaw = yaw
        msg.yawspeed = 0.0

        msg.timestamp = timestamp

        self.trajectory_setpoint_pub.publish(msg)

        # 로그 (1초마다)
        self.get_logger().info(
            f'[위치 제어] x_err={x_error:.2f}m, y_err={y_error:.2f}m, '
            f'yaw={np.degrees(yaw):.1f}°',
            throttle_duration_sec=1.0
        )


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = PX4OffboardBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt - 종료 중...')
    except Exception as e:
        print(f'오류 발생: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
