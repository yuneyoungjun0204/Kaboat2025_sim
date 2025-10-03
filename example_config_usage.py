#!/usr/bin/env python3
"""
ConfigManager 사용 예제
- main 파일에서 어떻게 사용하는지 보여주는 샘플 코드
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, NavSatFix
from std_msgs.msg import Float64
from utils import ConfigManager, get_config


class ExampleNodeWithConfig(Node):
    """ConfigManager를 사용하는 예제 노드"""

    def __init__(self):
        super().__init__('example_node_with_config')

        # ==================== 방법 1: ConfigManager 직접 생성 ====================
        self.config = ConfigManager()

        # 또는
        # ==================== 방법 2: 전역 싱글톤 사용 (권장) ====================
        # self.config = get_config()

        # ==================== 토픽 설정 ====================
        # 하드코딩 대신 Config에서 가져오기
        lidar_topic = self.config.get_sensor_topic('lidar')
        gps_topic = self.config.get_sensor_topic('gps')
        left_thrust_topic = self.config.get_actuator_topic('thrusters', 'left')

        # QoS도 Config에서
        qos = self.config.get_qos('sensor_data')

        # 서브스크라이버 생성
        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidar_topic,  # Config에서 가져온 토픽명
            self.lidar_callback,
            qos
        )

        self.gps_sub = self.create_subscription(
            NavSatFix,
            gps_topic,
            self.gps_callback,
            qos
        )

        # 퍼블리셔 생성
        self.left_thrust_pub = self.create_publisher(
            Float64,
            left_thrust_topic,
            qos
        )

        # ==================== 파라미터 설정 ====================
        # 제어 파라미터 가져오기
        control_params = self.config.get_control_params()
        self.thrust_scale = control_params.get('thrust_scale', 800)
        self.v_scale = control_params.get('v_scale', 1.0)
        self.w_scale = control_params.get('w_scale', -1.0)

        # 모델 경로 가져오기
        self.model_path = self.config.get_model_path()

        # 미션 파라미터 가져오기
        gate_params = self.config.get_mission_params('gate')
        self.gate_threshold = gate_params.get('completion_threshold', 15.0)
        self.gate_kp = gate_params.get('kp_heading', 2.0)

        # 센서 파라미터 가져오기
        lidar_params = self.config.get_sensor_params('lidar')
        self.max_lidar_distance = lidar_params.get('max_distance', 100.0)

        # 타이머 설정
        control_period = self.config.get_timer_period('control_update')
        self.timer = self.create_timer(control_period, self.timer_callback)

        # 설정 출력 (디버깅)
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'📋 LiDAR Topic: {lidar_topic}')
        self.get_logger().info(f'📋 GPS Topic: {gps_topic}')
        self.get_logger().info(f'🚀 Thrust Scale: {self.thrust_scale}')
        self.get_logger().info(f'🤖 Model Path: {self.model_path}')
        self.get_logger().info(f'⏱️  Control Period: {control_period}s')
        self.get_logger().info('=' * 50)

    def lidar_callback(self, msg):
        """LiDAR 콜백"""
        pass

    def gps_callback(self, msg):
        """GPS 콜백"""
        pass

    def timer_callback(self):
        """타이머 콜백"""
        pass


# ==================== 사용 패턴 요약 ====================
"""
1. 토픽 가져오기:
   - config.get_sensor_topic('lidar')
   - config.get_actuator_topic('thrusters', 'left')
   - config.get_vrx_topic('waypoint')

2. 파라미터 가져오기:
   - config.get_control_params()
   - config.get_mission_params('gate')
   - config.get_param('control', 'thrust_scale')

3. 특수 메서드:
   - config.get_model_path()
   - config.get_timer_period('control_update')
   - config.get_qos('sensor_data')

4. 변경 사항 적용:
   - config/topics.yaml 수정
   - config/mission_config.yaml 수정
   - 코드 수정 불필요!
"""


def main(args=None):
    rclpy.init(args=args)
    node = ExampleNodeWithConfig()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
