#!/usr/bin/env python3
"""
시각화 컴포넌트 모듈
- PlotManager: matplotlib 플롯 관리
- VizCallbackHandler: ROS2 콜백 처리
- VizUtils: 유틸리티 함수들
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from collections import deque

from .config import Constants
from .sensor_preprocessing import SensorDataManager


class VizUtils:
    """시각화 유틸리티 함수들"""

    @staticmethod
    def calculate_path_corners(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        width: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        경로의 네 모서리와 체크 포인트 계산

        Args:
            start_pos: 시작 위치 [x, y]
            end_pos: 끝 위치 [x, y]
            width: 경로 너비

        Returns:
            (corners, check_points): 모서리 점들, 체크 포인트들
        """
        if len(start_pos) < 2 or len(end_pos) < 2:
            return [], []

        distance = np.linalg.norm(end_pos - start_pos)
        if distance < 0.1:
            return [], []

        # 방향 벡터 및 수직 벡터
        direction = (end_pos - start_pos) / distance
        perp = np.array([-direction[1], direction[0]])

        # 네 모서리
        half_width = width / 2.0
        corners = [
            start_pos + half_width * perp,
            end_pos + half_width * perp,
            end_pos - half_width * perp,
            start_pos - half_width * perp,
            start_pos + half_width * perp
        ]

        # 체크 포인트들
        num_points = max(int(distance / 5.0), 5)
        check_points = [
            start_pos + t * (end_pos - start_pos)
            for t in np.linspace(0, 1, num_points + 1)
        ]

        return corners, check_points

    @staticmethod
    def create_arrow_params(
        pos: np.ndarray,
        heading_deg: float,
        length: float,
        color: str = 'red'
    ) -> Dict[str, Any]:
        """
        화살표 생성을 위한 파라미터 생성

        Args:
            pos: 위치 [x, y]
            heading_deg: 헤딩 (도)
            length: 화살표 길이
            color: 화살표 색상

        Returns:
            arrow 함수에 전달할 파라미터 딕셔너리
        """
        heading_rad = np.radians(heading_deg)
        dx = length * np.cos(heading_rad)
        dy = length * np.sin(heading_rad)

        return {
            'x': pos[0], 'y': pos[1],
            'dx': dx, 'dy': dy,
            'head_width': 3.0, 'head_length': 3.0,
            'fc': color, 'ec': color,
            'alpha': 0.8, 'linewidth': 3
        }

    @staticmethod
    def transform_to_robot_frame(
        points: np.ndarray,
        robot_pos: np.ndarray,
        heading_deg: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        전역 좌표를 로봇 중심 좌표로 변환

        Args:
            points: 전역 좌표 점들 [N x 2]
            robot_pos: 로봇 위치 [x, y]
            heading_deg: 로봇 헤딩 (도)

        Returns:
            (rel_x, rel_y): 로봇 중심 좌표
        """
        rel_x = points[:, 0] - robot_pos[0]
        rel_y = points[:, 1] - robot_pos[1]

        if abs(heading_deg) > 0.01:
            heading_rad = np.radians(heading_deg)
            cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
            rot_x = rel_x * cos_h + rel_y * sin_h
            rot_y = -rel_x * sin_h + rel_y * cos_h
            return rot_x, rot_y

        return rel_x, rel_y


class PlotManager:
    """
    matplotlib 플롯 관리자

    3개의 subplot을 관리:
    - ax1: 궤적 플롯
    - ax2: LiDAR 플롯
    - ax3: 제어 출력값 플롯
    """

    def __init__(self, logger):
        """
        Args:
            logger: ROS2 logger
        """
        self.logger = logger
        self.fig: Optional[Figure] = None
        self.ax1: Optional[Axes] = None  # 궤적
        self.ax2: Optional[Axes] = None  # LiDAR
        self.ax3: Optional[Axes] = None  # 제어 출력

        # 플롯 요소들
        self.trajectory_line = None
        self.current_pos = None
        self.lidar_points = None
        self.robot_center = None
        self.linear_bar = None
        self.angular_bar = None
        self.linear_text = None
        self.angular_text = None
        self.mode_text = None
        self.waypoint_markers = None

        # 동적 플롯 요소들 (매번 재생성)
        self.dynamic_elements: List[Any] = []

    def setup(self) -> Figure:
        """matplotlib 초기 설정"""
        self.logger.info("matplotlib 설정 중...")

        # Figure 생성
        self.fig = plt.figure(figsize=Constants.Visualization.FIGURE_SIZE)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[4, 1])

        self.ax1 = self.fig.add_subplot(gs[0, 0])  # 궤적
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # LiDAR
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # 제어

        self.fig.suptitle(
            'VRX Robot Trajectory & LiDAR (NED Coordinate)',
            fontsize=16, fontweight='bold'
        )

        self._setup_trajectory_plot()
        self._setup_lidar_plot()
        self._setup_control_plot()

        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.logger.info("✓ matplotlib 설정 완료")
        return self.fig

    def _setup_trajectory_plot(self):
        """궤적 플롯 초기화 (NED 좌표계)"""
        self.ax1.set_title('Robot Position Trajectory & Heading (NED)', fontsize=14)
        self.ax1.set_xlabel('North (m)', fontsize=12)
        self.ax1.set_ylabel('East (m)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlim(-100, 100)
        self.ax1.set_ylim(-100, 100)

        # 플롯 요소들
        self.trajectory_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Trajectory')
        self.current_pos, = self.ax1.plot([], [], 'ro', markersize=12, label='Current Position')
        self.waypoint_markers, = self.ax1.plot([], [], 'bo', markersize=8, label='Waypoints')

        # 범례용 더미
        self.ax1.plot([], [], 'r-', linewidth=3, label='IMU Heading')
        self.ax1.plot([], [], 'g-', linewidth=3, label='Target Heading')
        self.ax1.plot([], [], 'rD', markersize=8, label='LOS Target')
        self.ax1.scatter([], [], c='orange', marker='.', s=30, alpha=0.6, label='Obstacle Check Area')
        self.ax1.fill([], [], color='purple', alpha=0.3, label='Goal Check Area')

        self.ax1.legend(fontsize=10)

    def _setup_lidar_plot(self):
        """LiDAR 플롯 초기화"""
        self.ax2.set_title('LiDAR Obstacles (Polar View)', fontsize=14)
        self.ax2.set_xlabel('X (m)', fontsize=12)
        self.ax2.set_ylabel('Y (m)', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal')

        max_range = Constants.Visualization.LIDAR_MAX_RANGE
        self.ax2.set_xlim(-max_range, max_range)
        self.ax2.set_ylim(-max_range, max_range)

        self.lidar_points, = self.ax2.plot([], [], 'r.', markersize=3, label='Obstacles')
        self.robot_center, = self.ax2.plot([0], [0], 'bo', markersize=10, label='Robot')

        self.ax2.plot([], [], 'b-', linewidth=3, label='Target Heading')
        self.ax2.plot([], [], 'rD', markersize=6, label='LOS Target')

        self.ax2.legend(fontsize=10)

    def _setup_control_plot(self):
        """제어 출력값 플롯 초기화"""
        self.ax3.set_title('ONNX Model Output', fontsize=14)
        self.ax3.set_xlim(0, 1)
        self.ax3.set_ylim(-1.5, 1.5)
        self.ax3.set_xlabel('Linear Velocity', fontsize=12)
        self.ax3.set_ylabel('Angular Velocity', fontsize=12)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        self.ax3.axvline(x=0.5, color='black', linestyle='-', alpha=0.5)

        self.linear_bar = self.ax3.barh(0, 0, height=0.2, color='blue', alpha=0.7, label='Linear Vel')
        self.angular_bar = self.ax3.barh(0.5, 0, height=0.2, color='red', alpha=0.7, label='Angular Vel')

        self.linear_text = self.ax3.text(0.02, 0, '0.000', fontsize=10, va='center')
        self.angular_text = self.ax3.text(0.02, 0.5, '0.000', fontsize=10, va='center')
        self.mode_text = self.ax3.text(
            0.5, -1.2, 'Mode: UNKNOWN', fontsize=12, va='center', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
        )

        self.ax3.legend(fontsize=10)

    def clear_dynamic_elements(self):
        """동적 플롯 요소들 제거"""
        for element in self.dynamic_elements:
            try:
                element.remove()
            except:
                pass
        self.dynamic_elements.clear()

        # 축의 collections와 patches 제거
        for ax in [self.ax1, self.ax2]:
            for collection in ax.collections[:]:
                try:
                    collection.remove()
                except:
                    pass
            for patch in ax.patches[:]:
                try:
                    patch.remove()
                except:
                    pass

    def update_trajectory(
        self,
        positions: deque,
        heading: Optional[float] = None,
        target_heading: Optional[float] = None
    ):
        """궤적 업데이트 (NED: X=North, Y=East)"""
        if len(positions) < 2:
            return

        # positions = [[North, East], ...] from GPS (swapped)
        pos_array = np.array(positions)

        # NED plot: swap to X=North, Y=East
        self.trajectory_line.set_data(pos_array[:, 1], pos_array[:, 0])  # X=North, Y=East
        self.current_pos.set_data([pos_array[-1, 1]], [pos_array[-1, 0]])

        # 헤딩 화살표 (NED 좌표)
        if heading is not None:
            pos_ned = np.array([pos_array[-1, 1], pos_array[-1, 0]])  # [North, East]
            arrow_params = VizUtils.create_arrow_params(
                pos_ned, heading,
                Constants.Visualization.HEADING_ARROW_LENGTH,
                'red'
            )
            arrow = self.ax1.arrow(**arrow_params)
            self.dynamic_elements.append(arrow)

        # 목표 헤딩 화살표 (NED 좌표)
        if target_heading is not None:
            pos_ned = np.array([pos_array[-1, 1], pos_array[-1, 0]])  # [North, East]
            arrow_params = VizUtils.create_arrow_params(
                pos_ned, target_heading,
                Constants.Visualization.TARGET_HEADING_ARROW_LENGTH,
                'green'
            )
            arrow = self.ax1.arrow(**arrow_params)
            self.dynamic_elements.append(arrow)

    def update_lidar(
        self,
        lidar_x: np.ndarray,
        lidar_y: np.ndarray,
        target_heading: Optional[float] = None
    ):
        """LiDAR 업데이트"""
        self.lidar_points.set_data(lidar_x, lidar_y)

        # 목표 헤딩 화살표
        if target_heading is not None:
            heading_rad = np.radians(target_heading)
            length = Constants.Visualization.LIDAR_TARGET_HEADING_ARROW_LENGTH
            dx = -length * np.cos(heading_rad)
            dy = length * np.sin(heading_rad)

            arrow = self.ax2.arrow(
                0, 0, dx, dy,
                head_width=5.0, head_length=5.0,
                fc='blue', ec='blue', alpha=0.8, linewidth=3
            )
            self.dynamic_elements.append(arrow)

    def update_control_output(
        self,
        linear_vel: float,
        angular_vel: float,
        mode: str
    ):
        """제어 출력값 업데이트"""
        # 바 업데이트
        linear_width = max(0, min(1, (linear_vel + 1) / 2))
        angular_width = max(0, min(1, (angular_vel + 1) / 2))
        self.linear_bar[0].set_width(linear_width)
        self.angular_bar[0].set_width(angular_width)

        # 텍스트 업데이트
        self.linear_text.set_text(f'{linear_vel:.3f}')
        self.angular_text.set_text(f'{angular_vel:.3f}')

        # 색상 업데이트
        self.linear_bar[0].set_color(
            'blue' if linear_vel > 0 else 'red' if linear_vel < 0 else 'gray'
        )
        self.angular_bar[0].set_color(
            'green' if angular_vel > 0 else 'orange' if angular_vel < 0 else 'gray'
        )

        # 모드 업데이트
        mode_colors = {
            "DIRECT_CONTROL": "lightgreen", "ONNX_MODEL": "lightblue",
            "ONNX": "lightblue", "DIRECT": "lightgreen",
            "STOP": "lightcoral", "REACHED": "lightyellow"
        }
        color = mode_colors.get(mode, "lightgray")

        self.mode_text.set_text(f'Mode: {mode}')
        self.mode_text.set_bbox(
            dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
        )

    def update_waypoints(self, waypoints: List[List[float]], current: Optional[List[float]] = None):
        """웨이포인트 업데이트 (NED: X=North, Y=East)"""
        if len(waypoints) > 0:
            # waypoints = [[East, North], ...] from click events or storage
            wp_array = np.array(waypoints)
            # NED plot: swap to X=North, Y=East
            self.waypoint_markers.set_data(wp_array[:, 1], wp_array[:, 0])

        if current:
            # current = [East, North]
            marker = self.ax1.plot(
                [current[1]], [current[0]],  # X=North, Y=East
                'rs', markersize=12, markeredgecolor='black', markeredgewidth=2
            )[0]
            self.dynamic_elements.append(marker)

    def update_los_target(self, los_target: List[float], robot_pos: List[float]):
        """LOS target 시각화 (NED: X=North, Y=East)"""
        # 입력: los_target = [East, North], robot_pos = [East, North]
        # NED plot을 위해 swap: matplotlib X=North, Y=East

        # 궤적 플롯 (NED 좌표계)
        marker = self.ax1.scatter(
            [los_target[1]], [los_target[0]],  # X=North, Y=East
            c='red', marker='D', s=100, alpha=0.8, zorder=6
        )
        self.dynamic_elements.append(marker)

        line, = self.ax1.plot(
            [robot_pos[1], los_target[1]],  # X=North
            [robot_pos[0], los_target[0]],  # Y=East
            'r--', alpha=0.7, linewidth=2, zorder=5
        )
        self.dynamic_elements.append(line)

        # LiDAR 플롯 (relative)
        rel_north = los_target[1] - robot_pos[1]  # North 차이
        rel_east = los_target[0] - robot_pos[0]   # East 차이
        marker2 = self.ax2.scatter([rel_north], [rel_east], c='red', marker='D', s=80, alpha=0.8, zorder=6)
        line2, = self.ax2.plot([0, rel_north], [0, rel_east], 'r--', alpha=0.7, linewidth=2, zorder=5)
        self.dynamic_elements.extend([marker2, line2])

    def update_obstacle_check_area(self, area_points: List[List[float]], robot_pos: List[float]):
        """장애물 검사 영역 시각화 (NED: X=North, Y=East)"""
        if not area_points or len(area_points) < 2:
            return

        # 점들을 배열로 변환 [East, North]
        area_array = np.array(area_points)

        # NED plot을 위해 swap: matplotlib X=North, Y=East
        # 점들을 scatter로 표시
        scatter = self.ax1.scatter(
            area_array[:, 1], area_array[:, 0],  # X=North, Y=East
            c='orange', marker='.', s=10, alpha=0.4, zorder=3,
            label='Obstacle Check Area'
        )
        self.dynamic_elements.append(scatter)

    def update_goal_check_areas(self, goal_check_areas: List[Dict]):
        """Goal check 영역 시각화 (NED: X=North, Y=East)"""
        for area in goal_check_areas:
            if 'corners' in area and len(area['corners']) >= 4:
                corners = np.array(area['corners'])  # [[East1, North1], [East2, North2], ...]

                # NED plot을 위해 swap: matplotlib X=North, Y=East
                corners_ned = np.column_stack([corners[:, 0], corners[:, 1]])  # [[North1, East1], ...]

                # 폴리곤으로 표시
                from matplotlib.patches import Polygon
                poly = Polygon(
                    corners_ned, closed=True,
                    facecolor='purple', edgecolor='purple',
                    alpha=0.3, linewidth=2, zorder=4
                )
                self.ax1.add_patch(poly)
                self.dynamic_elements.append(poly)

    def draw(self):
        """화면 업데이트"""
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except:
            pass


class VizCallbackHandler:
    """시각화를 위한 ROS2 콜백 핸들러"""

    def __init__(self, sensor_manager: SensorDataManager, logger):
        """
        Args:
            sensor_manager: 센서 데이터 관리자
            logger: ROS2 logger
        """
        self.sensor_manager = sensor_manager
        self.logger = logger

        # 상태 변수들
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.current_mode = "UNKNOWN"
        self.current_control_mode = "UNKNOWN"
        self.current_los_target: Optional[List[float]] = None
        self.current_obstacle_check_area: List[List[float]] = []
        self.current_goal_check_areas: List[Dict] = []

    def process_control_output(self, msg) -> Tuple[float, float]:
        """제어 출력값 처리"""
        if len(msg.data) >= 2:
            self.linear_velocity = float(msg.data[0])
            self.angular_velocity = float(msg.data[1])
        return self.linear_velocity, self.angular_velocity

    def process_mode(self, msg) -> str:
        """모드 정보 처리"""
        self.current_mode = msg.data
        return self.current_mode

    def process_control_mode(self, msg) -> str:
        """제어 모드 정보 처리"""
        self.current_control_mode = msg.data
        return self.current_control_mode

    def process_los_target(self, msg):
        """LOS target 처리"""
        if len(msg.data) >= 2:
            self.current_los_target = [msg.data[0], msg.data[1]]
        else:
            self.current_los_target = None

    def process_obstacle_check_area(self, msg):
        """장애물 체크 영역 처리"""
        self.current_obstacle_check_area = [
            [msg.data[i], msg.data[i + 1]]
            for i in range(0, len(msg.data) - 1, 2)
        ]

    def process_goal_check(self, msg):
        """goal_check 영역 처리"""
        if len(msg.data) < 9 or msg.data[0] != 4.0:
            self.current_goal_check_areas = []
            return

        corners = [
            [msg.data[i+1], msg.data[i]]
            for i in range(1, len(msg.data) - 1, 2)
        ]

        if len(corners) >= 4:
            self.current_goal_check_areas = [{'type': msg.data[0], 'corners': corners}]
        else:
            self.current_goal_check_areas = []

    def get_display_mode(self) -> str:
        """표시할 모드 가져오기"""
        return (self.current_control_mode
                if self.current_control_mode != "UNKNOWN"
                else self.current_mode)

    def get_target_heading(self, current_heading: float) -> Optional[float]:
        """목표 헤딩 계산"""
        if abs(self.angular_velocity) > 0.01:
            angular_angle = self.angular_velocity * 60.0
            return current_heading + angular_angle
        return None
