#!/usr/bin/env python3
"""
VRX 로봇 통합 시각화 (좌표계 완전 통일)
- 모든 좌표계를 NED(North-East-Down)로 통일
- 명확하고 직관적인 시각화
- 장애물 검사 영역 scatter 표시
- 성능 최적화
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray, String
import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Circle, FancyArrow

from utils import Constants, SensorDataManager
from utils.sensor_callbacks import LidarFilter
from utils.waypoint_manager import gps_to_local
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# PX4 메시지 (optional)
try:
    from px4_msgs.msg import VehicleGlobalPosition, VehicleLocalPosition
    PX4_MSGS_AVAILABLE = True
except ImportError:
    PX4_MSGS_AVAILABLE = False
    VehicleGlobalPosition = None
    VehicleLocalPosition = None


class CoordinateTransformer:
    """좌표 변환 유틸리티 - 모든 좌표를 NED로 통일"""

    @staticmethod
    def gps_to_ned(gps_data: dict) -> np.ndarray:
        """GPS 데이터를 NED 좌표로 변환

        Args:
            gps_data: {'utm_x': North, 'utm_y': East}

        Returns:
            [North, East] 배열
        """
        return np.array([gps_data['utm_x'], gps_data['utm_y']])

    @staticmethod
    def lidar_to_ned(lidar_x: np.ndarray, lidar_y: np.ndarray,
                     robot_pos: np.ndarray, heading_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """LiDAR 데이터를 전역 NED 좌표로 변환

        Args:
            lidar_x, lidar_y: LiDAR 센서 좌표계 데이터
            robot_pos: 로봇 위치 [North, East]
            heading_deg: 로봇 헤딩 (도)

        Returns:
            (north_coords, east_coords): NED 좌표
        """
        # LiDAR 좌표계를 로봇 중심으로 회전 (90도 보정 포함)
        heading_rad = np.radians(heading_deg)
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)

        # LiDAR → Robot frame (90도 회전)
        rotated_x = lidar_y
        rotated_y = -lidar_x

        # Robot frame → Global NED
        north = robot_pos[0] + (rotated_x * cos_h - rotated_y * sin_h)
        east = robot_pos[1] + (rotated_x * sin_h + rotated_y * cos_h)

        return north, east

    @staticmethod
    def heading_to_arrow(pos: np.ndarray, heading_deg: float,
                        length: float = 15.0) -> dict:
        """헤딩을 화살표 파라미터로 변환

        Args:
            pos: 위치 [North, East]
            heading_deg: 헤딩 (도)
            length: 화살표 길이

        Returns:
            arrow 함수용 파라미터 딕셔너리
        """
        heading_rad = np.radians(heading_deg)
        dx = length * np.cos(heading_rad)
        dy = length * np.sin(heading_rad)

        return {
            'x': pos[0], 'y': pos[1],
            'dx': dx, 'dy': dy,
            'head_width': 3.0, 'head_length': 3.0,
            'linewidth': 3, 'alpha': 0.8
        }


class UnifiedPlotManager:
    """통합 플롯 매니저 - NED 좌표계 사용"""

    def __init__(self, logger):
        self.logger = logger
        self.fig: Optional[Figure] = None
        self.ax_main: Optional[plt.Axes] = None  # 메인 플롯 (궤적 + LiDAR)
        self.ax_lidar_polar: Optional[plt.Axes] = None  # LiDAR 극좌표
        self.ax_control: Optional[plt.Axes] = None  # 제어 출력

        # 플롯 요소들
        self.trajectory_line = None
        self.robot_marker = None
        self.lidar_scatter = None
        self.waypoint_markers = None

        # 제어 출력 요소들
        self.linear_bar = None
        self.angular_bar = None
        self.linear_text = None
        self.angular_text = None
        self.mode_text = None

        # 동적 요소들 (매 프레임마다 재생성)
        self.dynamic_elements = []

        # 좌표 변환기
        self.transformer = CoordinateTransformer()

    def setup(self):
        """matplotlib 초기 설정"""
        self.logger.info("🎨 통합 시각화 설정 중...")

        # Figure 생성
        self.fig = plt.figure(figsize=(18, 9))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[3, 2, 1], height_ratios=[4, 1])

        self.ax_main = self.fig.add_subplot(gs[0, 0])  # 메인 플롯
        self.ax_lidar_polar = self.fig.add_subplot(gs[0, 1], projection='polar')  # LiDAR 극좌표
        self.ax_control = self.fig.add_subplot(gs[0, 2])  # 제어 출력

        self.fig.suptitle(
            'VRX Unified Visualization (NED Coordinate System)',
            fontsize=18, fontweight='bold'
        )

        self._setup_main_plot()
        self._setup_lidar_polar_plot()
        self._setup_control_plot()

        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.logger.info("✓ 시각화 설정 완료")

    def _setup_main_plot(self):
        """메인 플롯 초기화 (NED: North=X, East=Y)"""
        self.ax_main.set_title('Robot Trajectory & Environment (NED)', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('North (m)', fontsize=12)
        self.ax_main.set_ylabel('East (m)', fontsize=12)
        self.ax_main.grid(True, alpha=0.4, linestyle='--')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlim(-100, 100)
        self.ax_main.set_ylim(-100, 100)

        # 배경색
        self.ax_main.set_facecolor('#f0f0f0')

        # 플롯 요소들
        self.trajectory_line, = self.ax_main.plot(
            [], [], 'b-', linewidth=2.5, label='Trajectory', alpha=0.8
        )
        self.robot_marker, = self.ax_main.plot(
            [], [], 'ro', markersize=14, label='Robot', markeredgecolor='darkred', markeredgewidth=2
        )
        self.lidar_scatter = self.ax_main.scatter(
            [], [], c='red', marker='.', s=8, alpha=0.5, label='LiDAR Obstacles'
        )
        self.waypoint_markers, = self.ax_main.plot(
            [], [], 'go', markersize=10, label='Waypoints', markeredgecolor='darkgreen', markeredgewidth=2
        )

        # 범례용 더미
        self.ax_main.plot([], [], 'r-', linewidth=3, label='Current Heading')
        self.ax_main.plot([], [], 'lime', linewidth=3, label='Target Heading')
        self.ax_main.plot([], [], 'mD', markersize=10, label='LOS Target')
        self.ax_main.scatter([], [], c='orange', marker='x', s=60, linewidths=2, label='Obstacle Check Points')
        self.ax_main.fill([], [], color='purple', alpha=0.3, label='Goal Check Zone')

        self.ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)

    def _setup_lidar_polar_plot(self):
        """LiDAR 극좌표 플롯 초기화 (NED 좌표계)"""
        self.ax_lidar_polar.set_title('LiDAR Polar View (NED)', fontsize=14, fontweight='bold')
        self.ax_lidar_polar.set_theta_zero_location('N')  # North = 0도
        self.ax_lidar_polar.set_theta_direction(-1)  # 시계방향 (NED)
        self.ax_lidar_polar.set_ylim(0, Constants.Visualization.LIDAR_MAX_RANGE)
        self.ax_lidar_polar.grid(True, alpha=0.4)

    def _setup_control_plot(self):
        """제어 출력 플롯 초기화"""
        self.ax_control.set_title('Control Output', fontsize=14, fontweight='bold')
        self.ax_control.set_xlim(0, 1)
        self.ax_control.set_ylim(-1.5, 1.5)
        self.ax_control.set_xlabel('Velocity', fontsize=10)
        self.ax_control.grid(True, alpha=0.3)
        self.ax_control.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 바 차트
        self.linear_bar = self.ax_control.barh(
            0.3, 0, height=0.3, color='blue', alpha=0.7, label='Linear'
        )
        self.angular_bar = self.ax_control.barh(
            -0.3, 0, height=0.3, color='red', alpha=0.7, label='Angular'
        )

        # 텍스트
        self.linear_text = self.ax_control.text(
            0.02, 0.3, '0.000', fontsize=10, va='center', fontweight='bold'
        )
        self.angular_text = self.ax_control.text(
            0.02, -0.3, '0.000', fontsize=10, va='center', fontweight='bold'
        )
        self.mode_text = self.ax_control.text(
            0.5, -1.2, 'Mode: UNKNOWN', fontsize=11, va='center', ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9),
            fontweight='bold'
        )

        self.ax_control.legend(fontsize=10, loc='upper right')

    def clear_dynamic_elements(self):
        """동적 요소들 제거"""
        for element in self.dynamic_elements:
            try:
                element.remove()
            except:
                pass
        self.dynamic_elements.clear()

        # collections와 patches 제거
        for collection in self.ax_main.collections[:]:
            if collection != self.lidar_scatter:
                try:
                    collection.remove()
                except:
                    pass

        for patch in self.ax_main.patches[:]:
            try:
                patch.remove()
            except:
                pass

        # LiDAR 극좌표 정리
        for collection in self.ax_lidar_polar.collections[:]:
            try:
                collection.remove()
            except:
                pass

    def update_trajectory(self, positions: deque, current_heading: Optional[float] = None,
                         target_heading: Optional[float] = None):
        """궤적 업데이트

        Args:
            positions: [[North, East], ...] deque
            current_heading: 현재 헤딩 (도)
            target_heading: 목표 헤딩 (도)
        """
        if len(positions) < 2:
            return

        pos_array = np.array(positions)

        # 궤적 그리기 (NED: X=North, Y=East)
        self.trajectory_line.set_data(pos_array[:, 0], pos_array[:, 1])
        self.robot_marker.set_data([pos_array[-1, 0]], [pos_array[-1, 1]])

        robot_pos = pos_array[-1]

        # 현재 헤딩 화살표
        if current_heading is not None:
            arrow_params = self.transformer.heading_to_arrow(robot_pos, current_heading, 15.0)
            arrow = self.ax_main.arrow(**arrow_params, fc='red', ec='darkred', zorder=10)
            self.dynamic_elements.append(arrow)

        # 목표 헤딩 화살표
        if target_heading is not None:
            arrow_params = self.transformer.heading_to_arrow(robot_pos, target_heading, 18.0)
            arrow = self.ax_main.arrow(**arrow_params, fc='lime', ec='green', zorder=9)
            self.dynamic_elements.append(arrow)

    def update_lidar(self, lidar_x: np.ndarray, lidar_y: np.ndarray,
                     robot_pos: np.ndarray, heading: float, target_heading: float = None):
        """LiDAR 데이터 업데이트

        Args:
            lidar_x, lidar_y: LiDAR 센서 좌표
            robot_pos: 로봇 위치 [North, East]
            heading: 로봇 헤딩 (도)
        """
        if len(lidar_x) == 0:
            return

        # 전역 NED 좌표로 변환
        north, east = self.transformer.lidar_to_ned(lidar_x, lidar_y, robot_pos, heading)

        # 메인 플롯에 LiDAR 표시
        self.lidar_scatter.set_offsets(np.column_stack([north, east]))

        # 극좌표 플롯 업데이트
        ranges = np.sqrt(lidar_x**2 + lidar_y**2)
        angles = np.arctan2(lidar_y, lidar_x)

        scatter = self.ax_lidar_polar.scatter(
            angles, ranges, c='red', marker='.', s=15, alpha=0.6
        )
        self.dynamic_elements.append(scatter)
        # 현재 헤딩 (파란색 화살표)
        max_range = Constants.Visualization.LIDAR_MAX_RANGE
        current_arrow = self.ax_lidar_polar.annotate(
            '', xy=(target_heading+0, max_range * 0.9),
            xytext=(target_heading+0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=3)
        )
        self.dynamic_elements.append(current_arrow)




    def update_obstacle_check_area(self, area_points: List[List[float]]):
        """장애물 검사 영역 시각화 (Scatter)

        Args:
            area_points: [[North, East], ...] 리스트
        """
        if not area_points or len(area_points) < 1:
            return

        area_array = np.array(area_points)

        # 점들만 간단하게 표시
        scatter = self.ax_main.scatter(
            area_array[:, 0], area_array[:, 1],
            c='orange', marker='x', s=60, alpha=0.7, linewidths=2,
            zorder=5
        )
        self.dynamic_elements.append(scatter)

    def update_goal_check_areas(self, goal_areas: List[dict]):
        """Goal 체크 영역 시각화

        Args:
            goal_areas: [{'corners': [[North, East], ...]}, ...]
        """
        for area in goal_areas:
            if 'corners' in area and len(area['corners']) >= 4:
                corners = np.array(area['corners'])

                poly = Polygon(
                    corners, closed=True,
                    facecolor='purple', edgecolor='darkviolet',
                    alpha=0.3, linewidth=2.5, zorder=4
                )
                self.ax_main.add_patch(poly)
                self.dynamic_elements.append(poly)

    def update_los_target(self, los_target: List[float], robot_pos: List[float]):
        """LOS target 시각화

        Args:
            los_target: [North, East]
            robot_pos: [North, East]
        """
        # 마커
        marker = self.ax_main.scatter(
            [los_target[0]], [los_target[1]],
            c='magenta', marker='D', s=150, alpha=0.9,
            edgecolors='darkmagenta', linewidths=2, zorder=8
        )
        self.dynamic_elements.append(marker)

        # 연결선
        line, = self.ax_main.plot(
            [robot_pos[0], los_target[0]],
            [robot_pos[1], los_target[1]],
            'm--', alpha=0.7, linewidth=2.5, zorder=7
        )
        self.dynamic_elements.append(line)

    def update_waypoints(self, waypoints: List[List[float]], current: Optional[List[float]] = None):
        """웨이포인트 업데이트

        Args:
            waypoints: [[North, East], ...]
            current: 현재 타겟 웨이포인트 [North, East]
        """
        if len(waypoints) > 0:
            wp_array = np.array(waypoints)
            self.waypoint_markers.set_data(wp_array[:, 0], wp_array[:, 1])

        if current:
            marker = self.ax_main.plot(
                [current[0]], [current[1]], 'gs', markersize=14,
                markeredgecolor='black', markeredgewidth=2.5, zorder=9
            )[0]
            self.dynamic_elements.append(marker)

    def update_control_output(self, linear_vel: float, angular_vel: float, mode: str):
        """제어 출력 업데이트"""
        # 바 크기 조정
        linear_width = max(0, min(1, abs(linear_vel)))
        angular_width = max(0, min(1, abs(angular_vel)))

        self.linear_bar[0].set_width(linear_width)
        self.angular_bar[0].set_width(angular_width)

        # 색상 변경
        self.linear_bar[0].set_color('blue' if linear_vel > 0 else 'red' if linear_vel < 0 else 'gray')
        self.angular_bar[0].set_color('green' if angular_vel > 0 else 'orange' if angular_vel < 0 else 'gray')

        # 텍스트 업데이트
        self.linear_text.set_text(f'{linear_vel:+.3f}')
        self.angular_text.set_text(f'{angular_vel:+.3f}')

        # 모드 업데이트
        mode_colors = {
            "DIRECT_CONTROL": "lightgreen", "ONNX_MODEL": "lightblue",
            "ONNX": "lightblue", "DIRECT": "lightgreen",
            "STOP": "lightcoral", "REACHED": "lightyellow"
        }
        color = mode_colors.get(mode, "lightgray")

        self.mode_text.set_text(f'Mode: {mode}')
        self.mode_text.set_bbox(
            dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9)
        )

    def draw(self):
        """화면 갱신"""
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except:
            pass


class UnifiedVizNode(Node):
    """통합 시각화 노드"""

    def __init__(self):
        super().__init__('unified_viz_node')

        # 매니저들
        self.sensor_manager = SensorDataManager()
        self.plot_manager = UnifiedPlotManager(self.get_logger())
        self.transformer = CoordinateTransformer()

        # LiDAR 필터 초기화 (sensor_callbacks와 동일한 설정)
        self.lidar_filter = None
        self.lidar_cartesian_x = np.array([])
        self.lidar_cartesian_y = np.array([])
        if Constants.LIDAR_FILTER_ENABLED:
            self.lidar_filter = LidarFilter(
                min_valid_distance=Constants.LIDAR_MIN_VALID_DISTANCE,
                max_valid_distance=Constants.LIDAR_MAX_VALID_DISTANCE,
                median_window=Constants.LIDAR_MEDIAN_FILTER_WINDOW,
                temporal_alpha=Constants.LIDAR_TEMPORAL_FILTER_ALPHA,
                array_size=Constants.LIDAR_ARRAY_SIZE
            )
            self.get_logger().info(
                f"✓ LiDAR 필터 활성화 - Median: {Constants.LIDAR_MEDIAN_FILTER_WINDOW}, "
                f"Temporal: {Constants.LIDAR_TEMPORAL_FILTER_ALPHA}, "
                f"Scale: {Constants.LIDAR_SCALE_FACTOR}"
            )

        # 데이터 히스토리 (NED 좌표)
        self.position_history = deque(maxlen=Constants.Visualization.POSITION_HISTORY_MAXLEN)
        self.heading_history = deque(maxlen=Constants.Visualization.HEADING_HISTORY_MAXLEN)

        # 상태 변수들
        self.current_position: Optional[np.ndarray] = None
        self.current_heading: Optional[float] = None
        self.axis_initialized = False

        # PX4 모드용 초기 위치 (GPS 기준점)
        self.initial_lat: Optional[float] = None
        self.initial_lon: Optional[float] = None

        # 웨이포인트
        self.waypoints = []
        self.current_waypoint: Optional[List[float]] = None

        # 콜백 데이터
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.control_mode = "UNKNOWN"
        self.los_target: Optional[List[float]] = None
        self.obstacle_check_area: List[List[float]] = []
        self.goal_check_areas: List[dict] = []

        # 플롯 설정 (ROS보다 먼저)
        self.plot_manager.setup()

        # ROS2 설정
        self._setup_ros()

        # 타이머
        self.timer = self.create_timer(Constants.Visualization.UPDATE_RATE, self.update_plot)

        self.get_logger().info('🚀 통합 시각화 시작!')
        self.get_logger().info('📍 클릭하여 웨이포인트 설정!')

    def _setup_ros(self):
        """ROS2 구독자 및 퍼블리셔 설정"""
        # 디버그: PX4 설정 상태 출력
        self.get_logger().info(f"PX4.ENABLED: {Constants.PX4.ENABLED}, PX4_MSGS_AVAILABLE: {PX4_MSGS_AVAILABLE}")

        # LiDAR 센서를 위한 BEST_EFFORT QoS 프로파일 정의
        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # PX4 모드에 따라 GPS/IMU 구독 설정
        if Constants.PX4.ENABLED and PX4_MSGS_AVAILABLE:
            # PX4 QoS 프로파일
            px4_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            # PX4 토픽 구독
            self.create_subscription(
                VehicleGlobalPosition,
                Constants.Topics.PX4_VEHICLE_GLOBAL_POSITION,
                self.px4_global_position_callback,
                px4_qos
            )
            self.create_subscription(
                VehicleLocalPosition,
                Constants.Topics.PX4_VEHICLE_LOCAL_POSITION,
                self.px4_local_position_callback,
                px4_qos
            )
            self.get_logger().info("PX4 모드: VehicleGlobalPosition, VehicleLocalPosition 구독")
        else:
            # 기존 시뮬레이터 토픽 구독
            self.create_subscription(NavSatFix, Constants.Topics.GPS_FIX, self.gps_callback, 10)
            self.create_subscription(Imu, Constants.Topics.IMU_DATA, self.imu_callback, 10)
            self.get_logger().info("시뮬레이터 모드: 기존 GPS/IMU 구독")

        # LiDAR 구독
        self.create_subscription(LaserScan, Constants.Topics.LIDAR_SCAN, self.lidar_callback, qos_sensor_data)

        # 공통 구독
        self.create_subscription(Float64MultiArray, Constants.Topics.CONTROL_OUTPUT, self.control_callback, 10)
        self.create_subscription(String, Constants.Topics.CONTROL_MODE, self.mode_callback, 10)
        self.create_subscription(Float64MultiArray, Constants.Topics.LOS_TARGET, self.los_callback, 10)
        self.create_subscription(Float64MultiArray, Constants.Topics.OBSTACLE_CHECK_AREA, self.obstacle_callback, 10)
        self.create_subscription(Float64MultiArray, Constants.Topics.GOAL_CHECK_AREAS, self.goal_callback, 10)

        # 퍼블리셔
        self.waypoint_pub = self.create_publisher(Point, Constants.Topics.WAYPOINT, 10)

        # 마우스 클릭 이벤트
        self.plot_manager.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        """마우스 클릭으로 웨이포인트 설정"""
        if event.inaxes == self.plot_manager.ax_main and event.button == 1:
            north, east = event.xdata, event.ydata
            if north is not None and east is not None:
                # 저장: [North, East]
                self.waypoints.append([north, east])
                self.current_waypoint = [north, east]

                # ROS2 publish: Point(x=North, y=East)
                msg = Point(x=float(north), y=float(east), z=0.0)
                self.waypoint_pub.publish(msg)

                self.get_logger().info(f'🎯 웨이포인트: N={north:.1f}m, E={east:.1f}m')

    # ============================================================================
    # 콜백 함수들
    # ============================================================================

    def gps_callback(self, msg):
        """GPS 데이터 콜백"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is None:
            return

        # NED 좌표로 변환
        position = self.transformer.gps_to_ned(gps_data)
        self.current_position = position
        self.position_history.append(position)

        # 축 초기화
        if not self.axis_initialized:
            self.axis_initialized = True
            margin_x = Constants.Visualization.AXIS_MARGIN_X
            margin_y = Constants.Visualization.AXIS_MARGIN_Y
            self.plot_manager.ax_main.set_xlim(-margin_x, margin_x)
            self.plot_manager.ax_main.set_ylim(-margin_y, margin_y)
            self.get_logger().info(f'📐 축 범위: N=±{margin_x}m, E=±{margin_y}m')

    def imu_callback(self, msg):
        """IMU 데이터 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.current_heading = imu_data['yaw_degrees']
        self.heading_history.append(self.current_heading)

    def px4_global_position_callback(self, msg):
        """PX4 VehicleGlobalPosition 콜백 - GPS 위치"""
        # 첫 번째 위치를 기준점으로 설정
        if self.initial_lat is None:
            self.initial_lat = msg.lat
            self.initial_lon = msg.lon
            self.get_logger().info(
                f"PX4 GPS 초기 위치 설정: lat={msg.lat:.8f}, lon={msg.lon:.8f}"
            )
            self.get_logger().info("✓ px4_global_position_callback 첫 호출 성공")

        # 로컬 좌표로 변환
        x_local, y_local = gps_to_local(msg.lat, msg.lon, self.initial_lat, self.initial_lon)
        position = np.array([x_local, y_local])
        self.current_position = position
        self.position_history.append(position)

        # 축 초기화
        if not self.axis_initialized:
            self.axis_initialized = True
            margin_x = Constants.Visualization.AXIS_MARGIN_X
            margin_y = Constants.Visualization.AXIS_MARGIN_Y
            self.plot_manager.ax_main.set_xlim(-margin_x, margin_x)
            self.plot_manager.ax_main.set_ylim(-margin_y, margin_y)
            self.get_logger().info(f'📐 축 범위: N=±{margin_x}m, E=±{margin_y}m')

    def px4_local_position_callback(self, msg):
        """PX4 VehicleLocalPosition 콜백 - heading"""
        # heading 항상 업데이트 (heading_good_for_control과 관계없이)
        # 라디안 → 도 변환 및 -180~180 정규화
        heading_deg = np.degrees(msg.heading)
        while heading_deg > 180:
            heading_deg -= 360
        while heading_deg < -180:
            heading_deg += 360

        # 첫 호출 시 로깅
        if self.current_heading is None:
            self.get_logger().info(f"✓ px4_local_position_callback 첫 호출 성공: heading={heading_deg:.1f}°")

        self.current_heading = heading_deg
        self.heading_history.append(self.current_heading)

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 (전처리 포함)"""
        # 첫 호출 시 로깅
        if not hasattr(self, '_lidar_callback_logged'):
            self._lidar_callback_logged = True
            self.get_logger().info(f"✓ lidar_callback 첫 호출 성공: {len(msg.ranges)} points")

        # sensor_callbacks.py와 동일한 전처리 로직 적용
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        raw_ranges = np.full(
            Constants.LIDAR_ARRAY_SIZE,
            Constants.MAX_LIDAR_DISTANCE,
            dtype=np.float32
        )

        # 1. 원본 데이터 변환 및 스케일 적용
        for i in range(len(ranges)):
            angle_rad = angle_min + i * angle_increment
            angle_deg = np.degrees(angle_rad)

            if Constants.LIDAR_ANGLE_RANGE[0] <= angle_deg <= Constants.LIDAR_ANGLE_RANGE[1]:
                distance = ranges[i]
                if np.isinf(distance) or np.isnan(distance) or distance >= Constants.MAX_LIDAR_DISTANCE:
                    distance = Constants.MAX_LIDAR_DISTANCE
                else:
                    # LIDAR_SCALE_FACTOR 적용
                    distance = distance * Constants.LIDAR_SCALE_FACTOR

                idx = int(angle_deg + 100)
                idx = max(0, min(Constants.LIDAR_ARRAY_SIZE - 1, idx))
                raw_ranges[idx] = distance

        # 2. 필터 적용 (활성화된 경우)
        if self.lidar_filter is not None:
            filtered_ranges = self.lidar_filter.filter(raw_ranges)
        else:
            filtered_ranges = raw_ranges

        # 3. 직교좌표로 변환 (시각화용)
        # 각도 배열 생성 (-100도 ~ +100도)
        angles_deg = np.arange(101, -100, -1)  # 201개
        angles_rad = np.radians(angles_deg)

        # 유효한 데이터만 선택 (최대 거리가 아닌 것)
        valid_mask = filtered_ranges < Constants.MAX_LIDAR_DISTANCE
        valid_ranges = filtered_ranges[valid_mask]
        valid_angles = angles_rad[valid_mask]

        # 직교좌표 변환
        if len(valid_ranges) > 0:
            self.lidar_cartesian_y = valid_ranges * np.cos(valid_angles)
            self.lidar_cartesian_x = valid_ranges * np.sin(valid_angles)
        else:
            self.lidar_cartesian_y = np.array([])
            self.lidar_cartesian_x = np.array([])

    def control_callback(self, msg):
        """제어 출력 콜백"""
        # 첫 호출 시 로깅
        if not hasattr(self, '_control_callback_logged'):
            self._control_callback_logged = True
            self.get_logger().info(f"✓ control_callback 첫 호출 성공")

        if len(msg.data) >= 2:
            self.linear_velocity = float(msg.data[0])
            self.angular_velocity = float(msg.data[1])

    def mode_callback(self, msg):
        """제어 모드 콜백"""
        # 첫 호출 시 로깅
        if not hasattr(self, '_mode_callback_logged'):
            self._mode_callback_logged = True
            self.get_logger().info(f"✓ mode_callback 첫 호출 성공: {msg.data}")

        self.control_mode = msg.data

    def los_callback(self, msg):
        """LOS target 콜백"""
        if len(msg.data) >= 2:
            self.los_target = [msg.data[1], msg.data[0]]  # [North, East]
        else:
            self.los_target = None

    def obstacle_callback(self, msg):
        """장애물 체크 영역 콜백"""
        self.obstacle_check_area = [
            [msg.data[i + 1], msg.data[i]]  # [North, East]
            for i in range(0, len(msg.data) - 1, 2)
        ]

    def goal_callback(self, msg):
        """Goal 체크 영역 콜백"""
        if len(msg.data) < 9 or msg.data[0] != 4.0:
            self.goal_check_areas = []
            return

        corners = [
            [msg.data[i], msg.data[i+1]]  # [North, East]
            for i in range(1, len(msg.data) - 1, 2)
        ]

        if len(corners) >= 4:
            self.goal_check_areas = [{'corners': corners}]
        else:
            self.goal_check_areas = []

    # ============================================================================
    # 플롯 업데이트
    # ============================================================================

    def update_plot(self):
        """플롯 업데이트 메인 함수"""
        try:
            # 동적 요소 제거
            self.plot_manager.clear_dynamic_elements()

            # 데이터 부족시 스킵
            if len(self.position_history) < 2 or self.current_position is None:
                return

            # 목표 헤딩 계산
            target_heading = None
            if self.current_heading is not None and abs(self.angular_velocity) > 0.01:
                target_heading = self.current_heading - (self.angular_velocity * 60.0)

            # 1. 궤적 업데이트
            self.plot_manager.update_trajectory(
                self.position_history,
                self.current_heading,
                target_heading
            )

            # 2. LiDAR 업데이트 (전처리된 데이터 사용)
            if self.current_heading is not None:
                # 전처리된 LiDAR 데이터 사용
                if len(self.lidar_cartesian_x) > 0:
                    self.plot_manager.update_lidar(
                        self.lidar_cartesian_y, self.lidar_cartesian_x,
                        [self.current_position[0],self.current_position[1]],
                        self.current_heading,
                        (self.angular_velocity * 1)
                    )

            # 3. 장애물 검사 영역
            if self.obstacle_check_area:
                self.plot_manager.update_obstacle_check_area(self.obstacle_check_area)

            # 4. Goal 체크 영역
            if self.goal_check_areas:
                self.plot_manager.update_goal_check_areas(self.goal_check_areas)

            # 5. LOS target
            if self.los_target:
                self.plot_manager.update_los_target(self.los_target, self.current_position)

            # 6. 웨이포인트
            self.plot_manager.update_waypoints(self.waypoints, self.current_waypoint)

            # 7. 제어 출력
            self.plot_manager.update_control_output(
                self.linear_velocity,
                self.angular_velocity,
                self.control_mode
            )

            # 화면 갱신
            self.plot_manager.draw()

        except Exception as e:
            self.get_logger().error(f'플롯 업데이트 오류: {e}')

    def destroy_node(self):
        """노드 종료"""
        plt.close('all')
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)

    try:
        node = UnifiedVizNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
