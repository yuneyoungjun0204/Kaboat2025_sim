#!/usr/bin/env python3
"""
VRX 로봇 궤적 시각화 (리팩토링됨)
- 깔끔하고 효율적인 구조
- PlotManager: matplotlib 관리
- VizCallbackHandler: ROS2 콜백 처리
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray, String
import numpy as np
from collections import deque
from typing import Optional

from utils import Constants, SensorDataManager
from utils.viz_components import PlotManager, VizCallbackHandler


class TrajectoryVizNode(Node):
    """VRX 로봇 궤적 시각화 노드 (간소화됨)"""

    def __init__(self):
        super().__init__('trajectory_viz_node')

        # 센서 및 콜백 관리
        self.sensor_manager = SensorDataManager()
        self.callback_handler = VizCallbackHandler(self.sensor_manager, self.get_logger())

        # 플롯 관리자
        self.plot_manager = PlotManager(self.get_logger())
        self.plot_manager.setup()

        # 히스토리
        self.position_history = deque(maxlen=Constants.Visualization.POSITION_HISTORY_MAXLEN)
        self.heading_history = deque(maxlen=Constants.Visualization.HEADING_HISTORY_MAXLEN)

        # 축 범위
        self.axis_initialized = False
        self.heading_offset = 0.0

        # 웨이포인트
        self.waypoints = []
        self.current_waypoint: Optional[list] = None

        # ROS2 설정
        self._setup_ros()

        # 타이머
        self.timer = self.create_timer(Constants.Visualization.UPDATE_RATE, self.update_plot)

        self.get_logger().info('🗺️ VRX 로봇 궤적 시각화 시작!')
        self.get_logger().info('🖱️  궤적 플롯에서 클릭하여 웨이포인트 설정!')

    def _setup_ros(self):
        """ROS2 구독자 및 퍼블리셔 설정"""
        # 서브스크라이버
        subscriptions = [
            (NavSatFix, Constants.Topics.GPS_FIX, self.gps_callback),
            (Imu, Constants.Topics.IMU_DATA, self.imu_callback),
            (LaserScan, Constants.Topics.LIDAR_SCAN, self.lidar_callback),
            (Float64MultiArray, Constants.Topics.CONTROL_OUTPUT, self.control_output_callback),
            (String, Constants.Topics.CONTROL_MODE, self.control_mode_callback),
            (Float64MultiArray, Constants.Topics.LOS_TARGET, self.los_target_callback),
            (Float64MultiArray, Constants.Topics.OBSTACLE_CHECK_AREA, self.obstacle_check_area_callback),
            (Float64MultiArray, Constants.Topics.GOAL_CHECK_AREAS, self.goal_check_callback),
        ]

        for msg_type, topic, callback in subscriptions:
            self.create_subscription(msg_type, topic, callback, Constants.QueueSizes.DEFAULT)

        # 퍼블리셔
        self.waypoint_pub = self.create_publisher(
            Point, Constants.Topics.WAYPOINT, Constants.QueueSizes.DEFAULT
        )

        # 마우스 클릭 이벤트
        self.plot_manager.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        """마우스 클릭으로 웨이포인트 설정"""
        if event.inaxes == self.plot_manager.ax1 and event.button == 1:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.waypoints.append([x, y])
                self.current_waypoint = [x, y]

                # ROS2 발행
                msg = Point(x=float(x), y=float(y), z=0.0)
                self.waypoint_pub.publish(msg)

                self.get_logger().info(f'🎯 웨이포인트: ({x:.1f}, {y:.1f})')

    # ============================================================================
    # 콜백 함수들 (간소화됨)
    # ============================================================================

    def gps_callback(self, msg):
        """GPS 데이터 콜백"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is None:
            return

        utm_x, utm_y = gps_data['utm_x'], gps_data['utm_y']

        # 축 초기화 (첫 GPS 데이터 기준)
        if not self.axis_initialized:
            self.axis_initialized = True
            margin_x = Constants.Visualization.AXIS_MARGIN_X
            margin_y = Constants.Visualization.AXIS_MARGIN_Y
            self.plot_manager.ax1.set_xlim(-margin_x, margin_x)
            self.plot_manager.ax1.set_ylim(-margin_y, margin_y)
            self.get_logger().info(f'축 범위 설정: X=±{margin_x}m, Y=±{margin_y}m')

        self.position_history.append([utm_x, utm_y])

    def imu_callback(self, msg):
        """IMU 데이터 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.heading_history.append(imu_data['yaw_degrees'])

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백"""
        self.sensor_manager.process_lidar_data(msg)

    def control_output_callback(self, msg):
        """제어 출력값 콜백"""
        self.callback_handler.process_control_output(msg)

    def control_mode_callback(self, msg):
        """제어 모드 콜백"""
        self.callback_handler.process_control_mode(msg)

    def los_target_callback(self, msg):
        """LOS target 콜백"""
        self.callback_handler.process_los_target(msg)

    def obstacle_check_area_callback(self, msg):
        """장애물 체크 영역 콜백"""
        self.callback_handler.process_obstacle_check_area(msg)

    def goal_check_callback(self, msg):
        """goal_check 영역 콜백"""
        self.callback_handler.process_goal_check(msg)

    # ============================================================================
    # 플롯 업데이트
    # ============================================================================

    def update_plot(self):
        """플롯 업데이트 (메인 함수)"""
        try:
            # 동적 요소 제거
            self.plot_manager.clear_dynamic_elements()

            # 데이터가 충분하지 않으면 스킵
            if len(self.position_history) < 2:
                return

            # 현재 상태 가져오기
            current_pos = self.position_history[-1]
            current_heading = self.heading_history[-1] if self.heading_history else None

            # 1. 궤적 업데이트
            target_heading = self.callback_handler.get_target_heading(current_heading) if current_heading else None
            self.plot_manager.update_trajectory(
                self.position_history,
                current_heading + self.heading_offset if current_heading else None,
                target_heading + self.heading_offset if target_heading else None
            )

            # 2. LiDAR 업데이트
            lidar_x, lidar_y = self.sensor_manager.get_lidar_cartesian()
            if len(lidar_x) > 0:
                self.plot_manager.update_lidar(lidar_x, lidar_y, target_heading)
                self._update_lidar_on_trajectory(current_pos, current_heading)

            # 3. 웨이포인트 업데이트
            self.plot_manager.update_waypoints(self.waypoints, self.current_waypoint)

            # 4. LOS target 업데이트
            if self.callback_handler.current_los_target:
                self.plot_manager.update_los_target(
                    self.callback_handler.current_los_target,
                    current_pos
                )

            # 5. 제어 출력값 업데이트
            mode = self.callback_handler.get_display_mode()
            self.plot_manager.update_control_output(
                self.callback_handler.linear_velocity,
                self.callback_handler.angular_velocity,
                mode
            )

            # 화면 갱신
            self.plot_manager.draw()

        except Exception as e:
            self.get_logger().error(f'플롯 업데이트 오류: {e}')

    def _update_lidar_on_trajectory(self, current_pos, current_heading):
        """궤적 플롯에 LiDAR 데이터 추가"""
        if current_heading is None:
            return

        lidar_x, lidar_y = self.sensor_manager.get_lidar_cartesian()
        if len(lidar_x) == 0:
            return

        # 좌표 변환
        heading_rad = np.radians(current_heading + self.heading_offset)
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)

        # LiDAR 좌표계 → UTM 좌표계 변환 (90도 회전 포함)
        rotated_x = lidar_y
        rotated_y = -lidar_x

        utm_x = current_pos[0] + (rotated_x * cos_h - rotated_y * sin_h)
        utm_y = current_pos[1] + (rotated_x * sin_h + rotated_y * cos_h)

        # 플롯
        lidar_traj, = self.plot_manager.ax1.plot(
            utm_x, utm_y, 'r.', markersize=2, alpha=0.6
        )
        self.plot_manager.dynamic_elements.append(lidar_traj)

    def destroy_node(self):
        """노드 종료"""
        import matplotlib.pyplot as plt
        plt.close('all')
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = TrajectoryVizNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
