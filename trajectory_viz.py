#!/usr/bin/env python3
"""
VRX 로봇 궤적 시각화
- Position Trajectory에 헤딩도 표시
- 간단하고 안정적인 버전
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray, String
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from utils import SensorDataManager
from utils.config import Constants

class TrajectoryVizNode(Node):
    """VRX 로봇 궤적 시각화 노드"""
    
    def __init__(self):
        super().__init__('trajectory_viz_node')

        self._setup_subscribers()
        self._initialize_variables()
        self.setup_matplotlib()

        self.waypoint_pub = self.create_publisher(Point, Constants.Topics.WAYPOINT, 10)
        self.timer = self.create_timer(Constants.Visualization.UPDATE_RATE, self.update_plot)

        self.get_logger().info('🗺️ VRX 로봇 궤적 시각화 시작!')
        self.get_logger().info('🖱️  궤적 플롯에서 클릭하여 웨이포인트를 설정하세요!')

    def _setup_subscribers(self):
        """ROS2 구독자 설정"""
        subscriptions = [
            (NavSatFix, Constants.Topics.GPS_FIX, self.gps_callback),
            (Imu, Constants.Topics.IMU_DATA, self.imu_callback),
            (LaserScan, Constants.Topics.LIDAR_SCAN, self.lidar_callback),
            (Float64MultiArray, Constants.Topics.CONTROL_OUTPUT, self.control_output_callback),
            (String, Constants.Topics.CURRENT_MODE, self.mode_callback),
            (Float64MultiArray, Constants.Topics.GOAL_CHECK_AREAS, self.goal_check_callback),
            (String, Constants.Topics.CONTROL_MODE, self.control_mode_callback),
            (Float64MultiArray, Constants.Topics.OBSTACLE_CHECK_AREA, self.obstacle_check_area_callback),
            (Float64MultiArray, Constants.Topics.LOS_TARGET, self.los_target_callback)
        ]

        for msg_type, topic, callback in subscriptions:
            self.create_subscription(msg_type, topic, callback, 10)

    def _initialize_variables(self):
        """변수 초기화"""
        self.sensor_manager = SensorDataManager()

        # 히스토리
        self.position_history = deque(maxlen=Constants.Visualization.POSITION_HISTORY_MAXLEN)
        self.heading_history = deque(maxlen=Constants.Visualization.HEADING_HISTORY_MAXLEN)

        # 축 범위 설정
        self.axis_initialized = False
        self.center_x = self.center_y = 0.0
        self.axis_margin = Constants.Visualization.AXIS_MARGIN
        self.axis_margin_y = Constants.Visualization.AXIS_MARGIN_Y
        self.axis_margin_x = Constants.Visualization.AXIS_MARGIN_X
        self.heading_offset = 0.0

        # 웨이포인트
        self.waypoints = []
        self.current_waypoint = None

        # 배 관련
        self.boat_width = Constants.Visualization.BOAT_WIDTH
        self.safety_margin = Constants.Visualization.SAFETY_MARGIN
        self.total_width = self.boat_width + self.safety_margin

        # 시각화 데이터
        self.path_width_points = []
        self.path_check_points = []
        self.current_path_area = None
        self.current_path_lines = []
        self.goal_check_areas = []
        self.goal_check_lines = []

        # 제어 관련
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.current_mode = "UNKNOWN"
        self.current_control_mode = "UNKNOWN"

        # 영역 데이터
        self.current_goal_check_areas = []
        self.current_obstacle_check_area = []
        self.current_los_target = None

    def _calculate_path_width_points(self, start_pos, end_pos):
        """배 폭만큼의 경로 점들 계산 - 네모 영역 전체"""
        if len(start_pos) < 2 or len(end_pos) < 2:
            return [], []

        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        if distance < 0.1:
            return [], []

        # 방향 벡터
        direction = np.array([(end_pos[0] - start_pos[0]) / distance,
                              (end_pos[1] - start_pos[1]) / distance])
        perp = np.array([-direction[1], direction[0]])  # 수직 벡터

        # 네 모서리 점들 계산
        half_width = self.total_width / 2.0
        corners = [
            start_pos + half_width * perp,
            end_pos + half_width * perp,
            end_pos - half_width * perp,
            start_pos - half_width * perp,
            start_pos + half_width * perp  # 닫기
        ]

        # 경로 체크 포인트들
        num_points = max(int(distance / 5.0), 5)
        check_points = [start_pos + t * (end_pos - start_pos)
                       for t in np.linspace(0, 1, num_points + 1)]

        return corners, check_points
    
    def setup_matplotlib(self):
        """matplotlib 설정"""
        # Figure 생성 (config에서 크기 가져오기)
        self.fig = plt.figure(figsize=Constants.Visualization.FIGURE_SIZE)
        
        # 서브플롯 레이아웃 설정
        gs = self.fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[4, 1])
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # 궤적 플롯
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # LiDAR 플롯
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # 트랙바 영역
        
        self.fig.suptitle('VRX Robot Trajectory & LiDAR Obstacles (UTM Coordinates)', fontsize=16, fontweight='bold')
        
        # 1. 궤적 플롯 설정 (왼쪽)
        self.ax1.set_title('Robot Position Trajectory & Heading', fontsize=14)
        self.ax1.set_xlabel('UTM X Position (m)', fontsize=12)
        self.ax1.set_ylabel('UTM Y Position (m)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        # 초기 축 범위는 나중에 GPS 데이터로 설정
        self.ax1.set_xlim(-100, 100)
        self.ax1.set_ylim(-100, 100)
        
        # 2. 원형좌표계 LiDAR 플롯 설정 (중앙)
        self.ax2.set_title('LiDAR Obstacles (Polar View)', fontsize=14)
        self.ax2.set_xlabel('X (m)', fontsize=12)
        self.ax2.set_ylabel('Y (m)', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(-50, 50)
        self.ax2.set_ylim(-50, 50)
        
        # 3. 트랙바 영역 설정 (오른쪽)
        self.ax3.set_title('ONNX Model Output', fontsize=14)
        self.ax3.set_xlim(0, 1)
        self.ax3.set_ylim(-1.5, 1.5)
        self.ax3.set_xlabel('Linear Velocity', fontsize=12)
        self.ax3.set_ylabel('Angular Velocity', fontsize=12)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        self.ax3.axvline(x=0.5, color='black', linestyle='-', alpha=0.5)
        
        # 초기 플롯 요소들
        # 궤적 플롯
        self.trajectory_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Trajectory')
        self.current_pos, = self.ax1.plot([], [], 'ro', markersize=12, label='Current Position')
        self.heading_arrow = None
        
        # 범례용 더미 플롯 (화살표용)
        self.ax1.plot([], [], 'r-', linewidth=3, label='IMU Heading')
        self.ax1.plot([], [], 'g-', linewidth=3, label='Target Heading')
        
        # goal_check 영역 범례용 더미 플롯
        self.ax1.plot([], [], 'orange', linewidth=2, alpha=0.6, label='Goal Check Left')
        self.ax1.plot([], [], 'red', linewidth=2, alpha=0.8, label='Goal Check Center')
        self.ax1.plot([], [], 'purple', linewidth=2, alpha=0.6, label='Goal Check Right')
        
        # ROS goal_check 영역 범례용 더미 플롯
        self.ax1.fill([], [], color='purple', alpha=0.3, label='Goal Check Area (ROS)')
        
        # LOS target 범례용 더미 플롯
        self.ax1.plot([], [], 'rD', markersize=8, label='LOS Target')
        self.ax1.plot([], [], 'r--', linewidth=2, label='LOS Line')
        
        # LiDAR 플롯
        self.lidar_points, = self.ax2.plot([], [], 'r.', markersize=3, label='Obstacles')
        self.robot_center, = self.ax2.plot([], [], 'bo', markersize=10, label='Robot')
        
        # LiDAR 창 범례용 더미 플롯 (목표 heading용)
        self.ax2.plot([], [], 'b-', linewidth=3, label='Target Heading')
        
        # LiDAR 창 LOS target 범례용 더미 플롯
        self.ax2.plot([], [], 'rD', markersize=6, label='LOS Target')
        self.ax2.plot([], [], 'r--', linewidth=2, label='LOS Line')
        
        # 제어 출력값 플롯 (트랙바 형태)
        self.linear_bar = self.ax3.barh(0, 0, height=0.2, color='blue', alpha=0.7, label='Linear Vel')
        self.angular_bar = self.ax3.barh(0.5, 0, height=0.2, color='red', alpha=0.7, label='Angular Vel')
        
        # 제어 출력값 텍스트 표시
        self.linear_text = self.ax3.text(0.02, 0, '0.000', fontsize=10, va='center', ha='left')
        self.angular_text = self.ax3.text(0.02, 0.5, '0.000', fontsize=10, va='center', ha='left')
        
        # v5 모드 표시 텍스트
        self.mode_text = self.ax3.text(0.5, -1.2, 'Mode: UNKNOWN', fontsize=12, va='center', ha='center', 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 범례
        self.ax1.legend(fontsize=10)
        self.ax2.legend(fontsize=10)
        self.ax3.legend(fontsize=10)
        
        # 마우스 클릭 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 웨이포인트 표시용 플롯 요소
        self.waypoint_markers, = self.ax1.plot([], [], 'bo', markersize=8, label='Waypoints')
        
        # 창 표시
        plt.ion()  # 인터랙티브 모드
        plt.show(block=False)
        plt.pause(0.1)
        
        self.get_logger().info('✅ matplotlib 창이 표시되었습니다!')
    
    def on_click(self, event):
        """마우스 클릭 이벤트 핸들러"""
        if event.inaxes == self.ax1 and event.button == 1:  # 좌클릭, 궤적 플롯에서만
            # 클릭한 좌표를 웨이포인트로 설정
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # 웨이포인트 추가
                self.waypoints.append([x, y])
                self.current_waypoint = [x, y]
                
                # ROS2 메시지로 발행
                waypoint_msg = Point()
                waypoint_msg.x = float(x)
                waypoint_msg.y = float(y)
                waypoint_msg.z = 0.0
                self.waypoint_pub.publish(waypoint_msg)
                
                self.get_logger().info(f'🎯 웨이포인트 설정: ({x:.1f}, {y:.1f})')
    
    def gps_callback(self, msg):
        """GPS 데이터 콜백"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is None:
            return

        utm_x, utm_y = gps_data['utm_x'], gps_data['utm_y']

        if not self.axis_initialized:
            self.center_x = self.center_y = 0.0
            self.axis_initialized = True
            self.ax1.set_xlim(self.center_x - self.axis_margin_x, self.center_x + self.axis_margin_x)
            self.ax1.set_ylim(self.center_y - self.axis_margin_y, self.center_y + self.axis_margin_y)
            self.get_logger().info(f'축 범위 설정: ±{self.axis_margin}m')

        self.position_history.append([utm_x, utm_y])
        self.get_logger().debug(f'GPS: X={utm_x:.2f}m, Y={utm_y:.2f}m')

    def imu_callback(self, msg):
        """IMU 데이터 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.heading_history.append(imu_data['yaw_degrees'])
        self.get_logger().debug(f'IMU Heading: {imu_data["yaw_degrees"]:.1f}°')

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백"""
        lidar_data = self.sensor_manager.process_lidar_data(msg)
        self.get_logger().debug(f'LiDAR: {lidar_data["valid_count"]}개 장애물')

    def control_output_callback(self, msg):
        """ONNX 모델 제어 출력값 콜백"""
        if len(msg.data) >= 2:
            self.linear_velocity = float(msg.data[0])
            self.angular_velocity = float(msg.data[1])
            self.get_logger().debug(f'제어: L={self.linear_velocity:.3f}, A={self.angular_velocity:.3f}')

    def mode_callback(self, msg):
        """v5 모드 정보 콜백"""
        self.current_mode = msg.data
        self.get_logger().debug(f'모드: {self.current_mode}')
    
    def goal_check_callback(self, msg):
        """goal_check 영역 정보 콜백"""
        if len(msg.data) < 9 or msg.data[0] != 4.0:
            self.current_goal_check_areas = []
            return

        corners = [[msg.data[i], msg.data[i + 1]]
                   for i in range(1, len(msg.data) - 1, 2)]

        if len(corners) >= 4:
            self.current_goal_check_areas = [{'type': msg.data[0], 'corners': corners}]
            self.get_logger().debug(f'goal_check: {len(corners)}개 모서리')
        else:
            self.current_goal_check_areas = []

    def control_mode_callback(self, msg):
        """제어 모드 정보 콜백"""
        self.current_control_mode = msg.data
        self.get_logger().debug(f'제어 모드: {self.current_control_mode}')

    def obstacle_check_area_callback(self, msg):
        """장애물 체크 영역 정보 콜백"""
        self.current_obstacle_check_area = [[msg.data[i], msg.data[i + 1]]
                                             for i in range(0, len(msg.data) - 1, 2)]
        self.get_logger().debug(f'장애물 영역: {len(self.current_obstacle_check_area)}개 점')

    def los_target_callback(self, msg):
        """LOS target 정보 콜백"""
        self.current_los_target = [msg.data[0], msg.data[1]] if len(msg.data) >= 2 else None
        if self.current_los_target:
            self.get_logger().debug(f'LOS target: ({self.current_los_target[0]:.1f}, {self.current_los_target[1]:.1f})')
    
    def update_plot(self):
        """플롯 업데이트"""
        try:
            # 매번 기존 플롯 요소들 완전 제거 (중첩 방지)
            self.clear_all_plots()
            
            # 궤적 업데이트
            self.update_trajectory_plot()
            
            # LiDAR 업데이트 (원형좌표계)
            self.update_lidar_plot()
            
            # 궤적 플롯에 LiDAR 데이터 추가
            self.update_trajectory_with_lidar()
            
            # 웨이포인트 표시 업데이트
            self.update_waypoints_plot()
            
            # 배 폭 경로 업데이트
            self.update_path_width_plot()
            
            # goal_check 경로 영역 시각화 업데이트 (ROS 메시지 기반)
            self.update_goal_check_area_from_ros()
            
            # 장애물 체크 영역 시각화 업데이트
            self.update_obstacle_check_area()
            
            # LOS target 시각화 업데이트
            self.update_los_target()
            
            # 제어 출력값 트랙바 업데이트
            self.update_control_output_plot()
            
            # 화면 업데이트
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f'플롯 업데이트 오류: {e}')
    
    def update_trajectory_plot(self):
        """궤적 플롯 업데이트"""
        if len(self.position_history) > 1:
            positions = np.array(self.position_history)
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            # 궤적 선 업데이트
            self.trajectory_line.set_data(x_coords, y_coords)
            
            # 현재 위치 업데이트
            if len(x_coords) > 0:
                self.current_pos.set_data([x_coords[-1]], [y_coords[-1]])
                
                # 헤딩 화살표 업데이트
                if len(self.heading_history) > 0:
                    current_heading = self.heading_history[-1]
                    arrow_length = Constants.Visualization.HEADING_ARROW_LENGTH  # config에서 가져오기
                    
                    # 헤딩 방향 계산 (UTM 좌표계 기준)
                    # UTM 좌표계: X=Easting(동서), Y=Northing(남북)
                    # IMU 헤딩: 0도=서쪽, 90도=북쪽, 180도=동쪽, 270도=남쪽
                    # 
                    # 헤딩을 UTM 좌표계에 맞게 변환:
                    # - 0도(서쪽) → X축 음의 방향
                    # - 90도(북쪽) → Y축 양의 방향  
                    # - 180도(동쪽) → X축 양의 방향
                    # - 270도(남쪽) → Y축 음의 방향
                    
                    # 헤딩 보정 적용
                    corrected_heading = current_heading + self.heading_offset
                    
                    # 헤딩을 라디안으로 변환하고 UTM 좌표계에 맞게 계산 (서쪽 기준)
                    heading_rad = np.radians(corrected_heading)
                    dx = arrow_length * np.cos(heading_rad)  # 동서 방향 (Easting) - 서쪽 기준
                    dy = arrow_length * np.sin(heading_rad)  # 남북 방향 (Northing) - 서쪽 기준
                    
                    # 새 화살표 추가 (기존 화살표는 clear_goal_check_plots에서 제거됨)
                    self.heading_arrow = self.ax1.arrow(
                        x_coords[-1], y_coords[-1], 
                        dx, dy, 
                        head_width=3.0, head_length=3.0, 
                        fc='red', ec='red', alpha=0.8, linewidth=3
                    )
                
                # 목표 heading 화살표 추가 (제어 출력값 기반)
                if hasattr(self, 'angular_velocity') and abs(self.angular_velocity) > 0.01:  # 임계값 추가
                    # Angular velocity를 각도로 변환 (-1~1 → -60도~60도)
                    angular_angle = self.angular_velocity * 60.0  # -1~1을 -60~60도로 매핑
                    # 현재 IMU heading에 angular_angle을 더함
                    target_heading = current_heading + angular_angle
                    target_heading_rad = np.radians(target_heading)

                    # 목표 heading 화살표 길이 (config에서 가져오기)
                    target_arrow_length = Constants.Visualization.TARGET_HEADING_ARROW_LENGTH
                    target_dx = target_arrow_length * np.cos(target_heading_rad)
                    target_dy = target_arrow_length * np.sin(target_heading_rad)
                    
                    # 새 목표 heading 화살표 추가 (초록색으로 헤딩과 구분)
                    # 기존 화살표는 clear_goal_check_plots에서 제거됨
                    self.target_heading_arrow = self.ax1.arrow(
                        x_coords[-1], y_coords[-1], 
                        target_dx, target_dy, 
                        head_width=4.0, head_length=4.0, 
                        fc='green', ec='green', alpha=0.8, linewidth=3
                    )
                
                # 축 범위는 고정 (첫 번째 GPS 데이터 기준)
                # 자동 조정 제거하여 축이 변하지 않도록 함
    
    def update_lidar_plot(self):
        """LiDAR 플롯 업데이트 (원형좌표계)"""
        # 센서 관리자에서 LiDAR 데이터 가져오기
        lidar_x, lidar_y = self.sensor_manager.get_lidar_cartesian()
        
        if len(lidar_x) > 0:
            # LiDAR 장애물 포인트 업데이트
            self.lidar_points.set_data(lidar_x, lidar_y)
            
            # 로봇 중심점 업데이트 (항상 원점)
            self.robot_center.set_data([0], [0])
            
            # 목표 heading 화살표 추가 (LiDAR 창용)
            if hasattr(self, 'angular_velocity') and abs(self.angular_velocity) > 0.01 and len(self.heading_history) > 0:
                # Angular velocity를 각도로 변환 (-1~1 → -60도~60도)
                angular_angle = self.angular_velocity * 60.0  # -1~1을 -60~60도로 매핑
                # 현재 IMU heading에 angular_angle을 더함
                current_heading = self.heading_history[-1]
                target_heading = current_heading + angular_angle
                target_heading_rad = np.radians(target_heading)

                # 목표 heading 화살표 길이 (config에서 가져오기)
                target_arrow_length = Constants.Visualization.LIDAR_TARGET_HEADING_ARROW_LENGTH
                target_dx = -target_arrow_length * np.cos(target_heading_rad)
                target_dy = target_arrow_length * np.sin(target_heading_rad)
                
                # 새 목표 heading 화살표 추가 (LiDAR 창용, 파란색으로 구분)
                # 기존 화살표는 clear_goal_check_plots에서 제거됨
                self.lidar_target_heading_arrow = self.ax2.arrow(
                    0, 0,  # 원점에서 시작
                    target_dx, target_dy, 
                    head_width=5.0, head_length=5.0, 
                    fc='blue', ec='blue', alpha=0.8, linewidth=3
                )
            
            # 축 범위 고정 (config에서 최대 범위 가져오기)
            max_range = Constants.Visualization.LIDAR_MAX_RANGE
            self.ax2.set_xlim(-max_range, max_range)
            self.ax2.set_ylim(-max_range, max_range)
    
    def update_trajectory_with_lidar(self):
        """궤적 플롯에 LiDAR 데이터 추가"""
        # 센서 관리자에서 LiDAR 데이터 가져오기
        lidar_x, lidar_y = self.sensor_manager.get_lidar_cartesian()
        
        if len(lidar_x) > 0 and len(self.position_history) > 0:
            # 현재 로봇 위치 가져오기
            current_pos = self.position_history[-1]
            robot_x, robot_y = current_pos[0], current_pos[1]
            
            # LiDAR 데이터를 로봇 중심 좌표계에서 전역 UTM 좌표계로 변환
            # LiDAR 좌표계: X=전방, Y=좌측 (로봇 기준)
            # UTM 좌표계: X=Easting, Y=Northing (지구 기준)
            
            # 현재 헤딩 가져오기
            if len(self.heading_history) > 0:
                current_heading = self.heading_history[-1] + self.heading_offset
                heading_rad = np.radians(current_heading)
                
                # 헤딩에 따른 회전 변환
                cos_h = np.cos(heading_rad)
                sin_h = np.sin(heading_rad)
                
                # LiDAR 좌표를 UTM 좌표로 변환 (90도 회전 적용)
                # LiDAR: X=전방, Y=좌측 → UTM: X=Easting, Y=Northing
                # 90도 회전: X → Y, Y → -X
                rotated_lidar_x = lidar_y  # Y축을 X축으로
                rotated_lidar_y = -lidar_x  # X축을 -Y축으로
                
                # 헤딩에 따른 회전 변환 적용
                utm_lidar_x = robot_x + (rotated_lidar_x * cos_h - rotated_lidar_y * sin_h)
                utm_lidar_y = robot_y + (rotated_lidar_x * sin_h + rotated_lidar_y * cos_h)
                
                # LiDAR 포인트를 궤적 플롯에 표시
                if not hasattr(self, 'lidar_trajectory_points'):
                    self.lidar_trajectory_points, = self.ax1.plot([], [], 'r.', markersize=2, alpha=0.6, label='LiDAR Obstacles')
                    self.ax1.legend()  # 범례 업데이트
                
                self.lidar_trajectory_points.set_data(utm_lidar_x, utm_lidar_y)
    
    def update_waypoints_plot(self):
        """웨이포인트 플롯 업데이트"""
        if len(self.waypoints) > 0:
            waypoints_array = np.array(self.waypoints)
            self.waypoint_markers.set_data(waypoints_array[:, 0], waypoints_array[:, 1])
            
            # 현재 웨이포인트 강조 표시 (빨간색)
            if self.current_waypoint:
                # 현재 웨이포인트 마커 추가 (빨간색으로 강조)
                # 기존 마커는 clear_goal_check_plots에서 제거됨
                self.current_waypoint_marker, = self.ax1.plot(
                    [self.current_waypoint[0]], [self.current_waypoint[1]], 
                    'rs', markersize=12, markeredgecolor='black', markeredgewidth=2,
                    label='Current Waypoint'
                )

    def update_path_width_plot(self):
        """배 폭 경로 시각화 업데이트"""
        if len(self.position_history) > 0 and self.current_waypoint is not None:
            current_pos = np.array([self.position_history[-1][0], self.position_history[-1][1]])
            target_pos = np.array(self.current_waypoint)

            # 배 폭 경로 점들 계산
            path_width_points, path_check_points = self._calculate_path_width_points(current_pos, target_pos)
            
            if len(path_width_points) > 0:
                # 배 폭 경로 시각화 (네모 영역 채우기)
                width_x = [p[0] for p in path_width_points]
                width_y = [p[1] for p in path_width_points]
                
                # 네모 영역 채우기 (현재 영역으로 저장)
                self.current_path_area = self.ax1.fill(
                    width_x, width_y, color='lightblue', alpha=0.2, 
                    label='Boat Width Area', zorder=1
                )[0]
                
                # 네모 영역 테두리 (라인으로 저장)
                border_line = self.ax1.plot(
                    width_x, width_y, 'blue', alpha=0.5, linewidth=1, zorder=2
                )[0]
                self.current_path_lines.append(border_line)
                
                # 경로 체크 포인트들 (중앙선) (라인으로 저장)
                check_x = [p[0] for p in path_check_points]
                check_y = [p[1] for p in path_check_points]
                check_line = self.ax1.plot(
                    check_x, check_y, 'blue', marker='.', markersize=3, 
                    alpha=0.7, label='Path Check Points', zorder=3
                )[0]
                self.current_path_lines.append(check_line)
    
    def clear_all_plots(self):
        """모든 플롯 요소들 완전 제거 (중첩 방지)"""
        # 리스트로 관리되는 플롯 요소들 제거
        for obj_list in [self.goal_check_areas, self.goal_check_lines, self.current_path_lines]:
            for obj in obj_list:
                try:
                    obj.remove()
                except:
                    pass
            obj_list.clear()

        # ROS goal_check 영역들 제거
        for area_obj in self.current_goal_check_areas:
            if 'plot_objects' in area_obj:
                for plot_obj in area_obj['plot_objects']:
                    try:
                        plot_obj.remove()
                    except:
                        pass
        self.current_goal_check_areas = []

        # 배 폭 경로 영역 제거
        if self.current_path_area is not None:
            try:
                self.current_path_area.remove()
            except:
                pass
            self.current_path_area = None

        # 동적 플롯 요소들 제거 (화살표, 마커 등)
        dynamic_attrs = [
            'lidar_target_heading_arrow', 'target_heading_arrow', 'heading_arrow',
            'lidar_trajectory_points', 'current_waypoint_marker',
            'obstacle_check_area_line', 'obstacle_check_area_points',
            'lidar_obstacle_check_area_line', 'lidar_obstacle_check_area_points',
            'los_target_marker', 'los_target_line',
            'lidar_los_target_marker', 'lidar_los_target_line'
        ]

        for attr in dynamic_attrs:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).remove()
                    delattr(self, attr)
                except:
                    pass

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
    
    
    def update_goal_check_area_from_ros(self):
        """ROS 메시지로 받은 goal_check 영역 시각화 업데이트"""
        if len(self.current_goal_check_areas) > 0:
            for area_obj in self.current_goal_check_areas:
                if area_obj['type'] == 4.0 and 'corners' in area_obj:  # 직사각형 영역
                    corners = area_obj['corners']
                    
                    # 직사각형 그리기
                    if len(corners) >= 4:
                        # 모서리 점들을 순서대로 정렬 (시계방향)
                        corner_x = [corner[0] for corner in corners]
                        corner_y = [corner[1] for corner in corners]
                        
                        # 직사각형 영역 채우기 (보라색으로 표시)
                        fill_area = self.ax1.fill(
                            corner_x, corner_y, 
                            color='purple', alpha=0.3, 
                            label='Goal Check Area', zorder=2
                        )[0]
                        
                        # 직사각형 테두리 (보라색 선)
                        border_line = self.ax1.plot(
                            corner_x + [corner_x[0]], corner_y + [corner_y[0]],  # 닫힌 다각형
                            color='purple', alpha=0.8, linewidth=2, zorder=3
                        )[0]
                        
                        # 플롯 객체들을 저장하여 나중에 제거할 수 있도록 함
                        area_obj['plot_objects'] = [fill_area, border_line]
                        
                        self.get_logger().debug(f'goal_check 직사각형 영역 표시: {len(corners)}개 모서리')
    
    def update_obstacle_check_area(self):
        """장애물 체크 영역 시각화 업데이트"""
        if len(self.current_obstacle_check_area) > 0:
            # 체크 영역 점들을 연결하여 선으로 표시
            check_y = [point[0] for point in self.current_obstacle_check_area]
            check_x = [point[1] for point in self.current_obstacle_check_area]
            
            # 체크 영역 선 그리기 (주황색으로 표시)
            self.obstacle_check_area_line, = self.ax1.plot(
                check_x, check_y, 
                color='orange', alpha=0.7, linewidth=2, 
                label='Obstacle Check Area', zorder=4
            )
            
            # 체크 영역 점들 표시 (작은 원으로)
            self.obstacle_check_area_points = self.ax1.scatter(
                check_x, check_y, 
                color='orange', s=20, alpha=0.8, 
                zorder=5
            )
            
            # LiDAR 창 (ax2)에도 직사각형 영역 표시
            if len(self.position_history) > 0:
                current_pos = self.position_history[-1]
                
                # 전역 좌표를 로봇 중심 좌표로 변환
                relative_check_x = []
                relative_check_y = []
                
                for point in self.current_obstacle_check_area:
                    # 전역 좌표에서 로봇 중심 좌표로 변환
                    rel_x = point[0] - current_pos[0]
                    rel_y = point[1] - current_pos[1]
                    relative_check_x.append(rel_x)
                    relative_check_y.append(rel_y)
                
                # LiDAR 창에 직사각형 영역 표시 (주황색)
                self.lidar_obstacle_check_area_line, = self.ax2.plot(
                    relative_check_x, relative_check_y, 
                    color='orange', alpha=0.7, linewidth=2, 
                    label='Obstacle Check Area', zorder=4
                )
                
                # LiDAR 창에 체크 영역 점들 표시
                self.lidar_obstacle_check_area_points = self.ax2.scatter(
                    relative_check_x, relative_check_y, 
                    color='orange', s=20, alpha=0.8, 
                    zorder=5
                )

    def update_los_target(self):
        """LOS target 시각화 업데이트"""
        if self.current_los_target is not None:
            # LOS target 마커 표시 (빨간색 다이아몬드로 구분)
            self.los_target_marker = self.ax1.scatter(
                [self.current_los_target[0]], [self.current_los_target[1]], 
                c='red', marker='D', s=100, alpha=0.8, 
                label='LOS Target', zorder=6
            )
            
            # LOS target에서 현재 위치로 선 그리기 (빨간색 점선)
            if len(self.position_history) > 0:
                current_pos = self.position_history[-1]
                self.los_target_line, = self.ax1.plot(
                    [current_pos[0], self.current_los_target[0]], 
                    [current_pos[1], self.current_los_target[1]], 
                    'r--', alpha=0.7, linewidth=2, 
                    label='LOS Line', zorder=5
                )
                
                # LiDAR 창에도 LOS target 표시 (로봇 중심 좌표로 변환)
                rel_los_x = self.current_los_target[0] - current_pos[0]
                rel_los_y = self.current_los_target[1] - current_pos[1]
                
                self.lidar_los_target_marker = self.ax2.scatter(
                    [rel_los_x], [rel_los_y], 
                    c='red', marker='D', s=80, alpha=0.8, 
                    label='LOS Target', zorder=6
                )
                
                # LiDAR 창에서 로봇 중심에서 LOS target으로 선 그리기
                self.lidar_los_target_line, = self.ax2.plot(
                    [0, rel_los_x], [0, rel_los_y], 
                    'r--', alpha=0.7, linewidth=2, 
                    label='LOS Line', zorder=5
                )

    def update_control_output_plot(self):
        """제어 출력값 트랙바 업데이트"""
        # 속도 트랙바 업데이트
        linear_width = max(0, min(1, (self.linear_velocity + 1) / 2))
        angular_width = max(0, min(1, (self.angular_velocity + 1) / 2))
        self.linear_bar[0].set_width(linear_width)
        self.angular_bar[0].set_width(angular_width)

        # 텍스트 업데이트
        self.linear_text.set_text(f'{self.linear_velocity:.3f}')
        self.angular_text.set_text(f'{self.angular_velocity:.3f}')

        # 모드 표시 및 색상 결정
        mode_colors = {
            "DIRECT_CONTROL": "lightgreen", "ONNX_MODEL": "lightblue",
            "ONNX": "lightblue", "ONNX_BOTH": "lightblue", "ONNX_FORWARD": "lightblue",
            "ONNX_PATH": "lightblue", "ONNX_CLOSE": "lightblue",
            "DIRECT": "lightgreen", "DIRECT_CLEAR": "lightgreen", "DIRECT_FORWARD": "lightgreen",
            "DIRECT_PATH": "lightgreen", "DIRECT_UNKNOWN": "lightgreen",
            "STOP": "lightcoral", "REACHED": "lightyellow"
        }

        display_mode = self.current_control_mode if self.current_control_mode != "UNKNOWN" else self.current_mode
        mode_color = mode_colors.get(display_mode, "lightgray")

        self.mode_text.set_text(f'Mode: {display_mode}')
        self.mode_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=mode_color, alpha=0.8))

        # 속도 바 색상 업데이트
        self.linear_bar[0].set_color('blue' if self.linear_velocity > 0 else 'red' if self.linear_velocity < 0 else 'gray')
        self.angular_bar[0].set_color('green' if self.angular_velocity > 0 else 'orange' if self.angular_velocity < 0 else 'gray')

    def destroy_node(self):
        """노드 종료 시 정리"""
        plt.close('all')
        super().destroy_node()

def main(args=None):
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