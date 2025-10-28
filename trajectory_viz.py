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
        
        # ROS2 서브스크라이버 (config에서 토픽명 가져오기)
        self.gps_sub = self.create_subscription(
            NavSatFix,
            Constants.Topics.GPS_FIX,
            self.gps_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            Constants.Topics.IMU_DATA,
            self.imu_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            Constants.Topics.LIDAR_SCAN,
            self.lidar_callback,
            10
        )

        # ONNX 모델 제어 출력값 서브스크라이버
        self.control_output_sub = self.create_subscription(
            Float64MultiArray,
            Constants.Topics.CONTROL_OUTPUT,
            self.control_output_callback,
            10
        )

        # v5 모드 정보 서브스크라이버
        self.mode_sub = self.create_subscription(
            String,
            Constants.Topics.CURRENT_MODE,
            self.mode_callback,
            10
        )

        # goal_check 영역 정보 서브스크라이버
        self.goal_check_sub = self.create_subscription(
            Float64MultiArray,
            Constants.Topics.GOAL_CHECK_AREAS,
            self.goal_check_callback,
            10
        )

        # 제어 모드 정보 서브스크라이버
        self.control_mode_sub = self.create_subscription(
            String,
            Constants.Topics.CONTROL_MODE,
            self.control_mode_callback,
            10
        )

        # 장애물 체크 영역 정보 서브스크라이버
        self.obstacle_check_area_sub = self.create_subscription(
            Float64MultiArray,
            Constants.Topics.OBSTACLE_CHECK_AREA,
            self.obstacle_check_area_callback,
            10
        )

        # LOS target 정보 서브스크라이버
        self.los_target_sub = self.create_subscription(
            Float64MultiArray,
            Constants.Topics.LOS_TARGET,
            self.los_target_callback,
            10
        )
        
        # 센서 데이터 관리자 초기화
        self.sensor_manager = SensorDataManager()

        # 히스토리 (config에서 길이 가져오기)
        self.position_history = deque(maxlen=Constants.Visualization.POSITION_HISTORY_MAXLEN)
        self.heading_history = deque(maxlen=Constants.Visualization.HEADING_HISTORY_MAXLEN)

        # 축 범위 고정을 위한 변수 (config에서 가져오기)
        self.axis_initialized = False
        self.center_x = 0.0
        self.center_y = 0.0
        self.axis_margin = Constants.Visualization.AXIS_MARGIN
        self.axis_margin_y = Constants.Visualization.AXIS_MARGIN_Y
        self.axis_margin_x = Constants.Visualization.AXIS_MARGIN_X

        # 헤딩 보정 (필요시 조정)
        self.heading_offset = 0.0  # 헤딩 오프셋 (도 단위)

        # matplotlib 설정
        self.setup_matplotlib()

        # 웨이포인트 퍼블리셔 (config에서 토픽명 가져오기)
        self.waypoint_pub = self.create_publisher(Point, Constants.Topics.WAYPOINT, 10)
        
        # 웨이포인트 관련 변수
        self.waypoints = []  # 클릭한 웨이포인트들 저장
        self.current_waypoint = None

        # 배 폭 및 안전 여유 (config에서 가져오기)
        self.boat_width = Constants.Visualization.BOAT_WIDTH
        self.safety_margin = Constants.Visualization.SAFETY_MARGIN
        self.total_width = self.boat_width + self.safety_margin  # 총 폭
        
        # 배 폭 경로 시각화용 변수
        self.path_width_points = []  # 배 폭 경로 점들
        self.path_check_points = []  # 경로 체크 포인트들
        
        # 현재 네모 영역 추적용 변수
        self.current_path_area = None  # 현재 네모 영역 (Polygon 객체)
        self.current_path_lines = []  # 현재 경로 라인들
        
        # goal_check 영역 시각화용 변수
        self.goal_check_areas = []  # goal_check에서 체크하는 영역들
        self.goal_check_lines = []  # goal_check 경계선들
        
        # ONNX 모델 제어 출력값 저장
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # v5 모드 표시용 변수
        self.current_mode = "UNKNOWN"  # 현재 모드 (ONNX/DIRECT/UNKNOWN)
        
        # goal_check 영역 정보 저장용 변수
        self.current_goal_check_areas = []  # 현재 goal_check 영역들
        
        # 제어 모드 관련 변수
        self.current_control_mode = "UNKNOWN"  # 현재 제어 모드 (DIRECT_CONTROL/ONNX_MODEL/UNKNOWN)
        
        # 장애물 체크 영역 관련 변수
        self.current_obstacle_check_area = []  # 현재 장애물 체크 영역 점들
        
        # LOS target 관련 변수
        self.current_los_target = None  # 현재 LOS target 위치

        # 타이머로 주기적 업데이트 (config에서 주기 가져오기)
        self.timer = self.create_timer(Constants.Visualization.UPDATE_RATE, self.update_plot)
        
        self.get_logger().info('🗺️ VRX 로봇 궤적 시각화 시작!')
        self.get_logger().info('🖱️  궤적 플롯에서 클릭하여 웨이포인트를 설정하세요!')

    def calculate_path_width_points(self, start_pos, end_pos):
        """배 폭만큼의 경로 점들 계산 - 네모 영역 전체"""
        if len(start_pos) < 2 or len(end_pos) < 2:
            return [], []
            
        # 시작점과 끝점 사이의 거리
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        if distance < 0.1:
            return [], []
            
        # 방향 벡터 계산
        direction_x = (end_pos[0] - start_pos[0]) / distance
        direction_y = (end_pos[1] - start_pos[1]) / distance
        
        # 수직 방향 벡터 (배 폭 방향)
        perp_x = -direction_y
        perp_y = direction_x
        
        # 네모 영역의 네 모서리 점들 계산
        # 시작점의 양쪽 모서리
        start_left_x = start_pos[0] + (self.total_width / 2.0) * perp_x
        start_left_y = start_pos[1] + (self.total_width / 2.0) * perp_y
        start_right_x = start_pos[0] - (self.total_width / 2.0) * perp_x
        start_right_y = start_pos[1] - (self.total_width / 2.0) * perp_y
        
        # 끝점의 양쪽 모서리
        end_left_x = end_pos[0] + (self.total_width / 2.0) * perp_x
        end_left_y = end_pos[1] + (self.total_width / 2.0) * perp_y
        end_right_x = end_pos[0] - (self.total_width / 2.0) * perp_x
        end_right_y = end_pos[1] - (self.total_width / 2.0) * perp_y
        
        # 네모 영역을 그리기 위한 점들 (시계방향)
        path_width_points = [
            [start_left_x, start_left_y],   # 시작점 왼쪽
            [end_left_x, end_left_y],       # 끝점 왼쪽
            [end_right_x, end_right_y],     # 끝점 오른쪽
            [start_right_x, start_right_y], # 시작점 오른쪽
            [start_left_x, start_left_y]    # 시작점 왼쪽 (닫기)
        ]
        
        # 경로 체크 포인트들 (중앙선)
        path_check_points = []
        num_check_points = max(int(distance / 5.0), 5)  # 5m 간격으로 체크
        for i in range(num_check_points + 1):
            t = i / num_check_points
            path_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            path_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            path_check_points.append([path_x, path_y])
        
        return path_width_points, path_check_points
    
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
        if gps_data is not None:
            # UTM 좌표로 변환된 위치 사용
            utm_x = gps_data['utm_x']
            utm_y = gps_data['utm_y']
            
            # 첫 번째 GPS 데이터로 축 범위 고정 설정 (이미 센서 전처리에서 기준점 설정됨)
            if not self.axis_initialized:
                self.center_x = 0.0  # 첫 번째 GPS 값이 (0,0)으로 설정됨
                self.center_y = 0.0
                self.axis_initialized = True
                
                # 축 범위 고정 설정 (기준점을 중심으로)
                self.ax1.set_xlim(self.center_x - self.axis_margin_x, self.center_x + self.axis_margin_x)
                self.ax1.set_ylim(self.center_y - self.axis_margin_y, self.center_y + self.axis_margin_y)
                
                self.get_logger().info(f'축 범위 고정 설정: 중심 (0,0) 기준, 범위 ±{self.axis_margin}m')
            
            # 위치 히스토리 업데이트
            self.position_history.append([utm_x, utm_y])
            
            self.get_logger().info(f'GPS 데이터 수신: UTM X={utm_x:.2f}m, UTM Y={utm_y:.2f}m')
    
    def imu_callback(self, msg):
        """IMU 데이터 콜백"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        
        # 헤딩 히스토리 업데이트
        self.heading_history.append(imu_data['yaw_degrees'])
        
        corrected_heading = imu_data["yaw_degrees"] + self.heading_offset
        self.get_logger().info(f'IMU 데이터 수신: 원본 Heading={imu_data["yaw_degrees"]:.1f}°, 보정된 Heading={corrected_heading:.1f}°')
    
    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백"""
        lidar_data = self.sensor_manager.process_lidar_data(msg)
        
        self.get_logger().info(f'LiDAR 데이터 수신: {lidar_data["valid_count"]}개 장애물 (원본: {lidar_data["raw_count"]}개)')
    
    def control_output_callback(self, msg):
        """ONNX 모델 제어 출력값 콜백"""
        if len(msg.data) >= 2:
            self.linear_velocity = float(msg.data[0])
            self.angular_velocity = float(msg.data[1])
            self.get_logger().info(f'제어 출력값 수신: Linear={self.linear_velocity:.3f}, Angular={self.angular_velocity:.3f}')
    
    def mode_callback(self, msg):
        """v5 모드 정보 콜백"""
        self.current_mode = msg.data
        self.get_logger().info(f'현재 모드: {self.current_mode}')
    
    def goal_check_callback(self, msg):
        """goal_check 영역 정보 콜백 (main_onnx_v5.py에서 받음)"""
        if len(msg.data) > 0:
            # 메시지 데이터 파싱
            # 형식: [type, x1, y1, x2, y2, x3, y3, x4, y4] (직사각형의 경우)
            area_type = msg.data[0]
            
            if area_type == 4.0 and len(msg.data) >= 9:  # 직사각형 영역
                # 4개 모서리 점들 추출
                corners = []
                for i in range(1, len(msg.data), 2):
                    if i + 1 < len(msg.data):
                        corners.append([msg.data[i], msg.data[i + 1]])
                
                if len(corners) >= 4:
                    self.current_goal_check_areas = [{
                        'type': area_type,
                        'corners': corners
                    }]
                    self.get_logger().debug(f'goal_check 영역 수신: {len(corners)}개 모서리')
            else:
                # 다른 타입의 영역들은 현재 처리하지 않음
                self.current_goal_check_areas = []
        else:
            self.current_goal_check_areas = []
    
    def control_mode_callback(self, msg):
        """제어 모드 정보 콜백 (main_onnx_v5_final.py에서 받음)"""
        self.current_control_mode = msg.data
        self.get_logger().info(f'현재 제어 모드: {self.current_control_mode}')
    
    def obstacle_check_area_callback(self, msg):
        """장애물 체크 영역 정보 콜백 (main_onnx_v5_final.py에서 받음)"""
        if len(msg.data) > 0:
            # 메시지 데이터를 점들로 변환 (x1, y1, x2, y2, ... 형태)
            self.current_obstacle_check_area = []
            for i in range(0, len(msg.data), 2):
                if i + 1 < len(msg.data):
                    self.current_obstacle_check_area.append([msg.data[i], msg.data[i + 1]])
            self.get_logger().debug(f'장애물 체크 영역 수신: {len(self.current_obstacle_check_area)}개 점')
        else:
            self.current_obstacle_check_area = []
    
    def los_target_callback(self, msg):
        """LOS target 정보 콜백 (main_onnx_v5_final.py에서 받음)"""
        if len(msg.data) >= 2:
            self.current_los_target = [msg.data[0], msg.data[1]]
            self.get_logger().debug(f'LOS target 수신: ({self.current_los_target[0]:.1f}, {self.current_los_target[1]:.1f})')
        else:
            self.current_los_target = None
    
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

    def clear_path_width_plots(self):
        """배 폭 관련 모든 플롯 제거 (legacy)"""
        # clear_all_plots에서 처리됨
        pass

    def update_path_width_plot(self):
        """배 폭 경로 시각화 업데이트"""
        if len(self.position_history) > 0 and self.current_waypoint is not None:
            # 현재 위치와 현재 웨이포인트 사이의 배 폭 경로 계산
            current_pos = [self.position_history[-1][0], self.position_history[-1][1]]
            target_pos = self.current_waypoint
            
            # 배 폭 경로 점들 계산
            path_width_points, path_check_points = self.calculate_path_width_points(current_pos, target_pos)
            
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
        # goal_check 영역들 제거 (legacy)
        for area in self.goal_check_areas:
            try:
                area.remove()
            except:
                pass
        self.goal_check_areas.clear()
        
        # goal_check 라인들 제거 (legacy)
        for line in self.goal_check_lines:
            try:
                line.remove()
            except:
                pass
        self.goal_check_lines.clear()
        
        # ROS로 받은 goal_check 영역들 제거
        if hasattr(self, 'current_goal_check_areas'):
            for area_obj in self.current_goal_check_areas:
                if 'plot_objects' in area_obj:
                    for plot_obj in area_obj['plot_objects']:
                        try:
                            plot_obj.remove()
                        except:
                            pass
            # current_goal_check_areas 초기화
            self.current_goal_check_areas = []
        
        # 배 폭 경로 영역 제거
        if self.current_path_area is not None:
            try:
                self.current_path_area.remove()
                self.current_path_area = None
            except:
                pass
        
        # 배 폭 경로 라인들 제거
        for line in self.current_path_lines:
            try:
                line.remove()
            except:
                pass
        self.current_path_lines.clear()
        
        # 모든 화살표 제거
        arrow_attributes = [
            'lidar_target_heading_arrow',
            'target_heading_arrow', 
            'heading_arrow'
        ]
        
        for attr in arrow_attributes:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).remove()
                    delattr(self, attr)
                except:
                    pass
        
        # 모든 마커/포인트 제거
        marker_attributes = [
            'lidar_trajectory_points',
            'current_waypoint_marker',
            'obstacle_check_area_line',
            'obstacle_check_area_points',
            'lidar_obstacle_check_area_line',
            'lidar_obstacle_check_area_points',
            'los_target_marker',
            'los_target_line',
            'lidar_los_target_marker',
            'lidar_los_target_line'
        ]
        
        for attr in marker_attributes:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).remove()
                    delattr(self, attr)
                except:
                    pass
        
        # 모든 축의 추가된 아티스트들 제거
        for ax in [self.ax1, self.ax2]:
            # 축에 추가된 모든 라인, 패치, 텍스트 제거
            for artist in ax.get_children():
                if hasattr(artist, '_goal_check_marker') or hasattr(artist, '_path_width_marker'):
                    try:
                        artist.remove()
                    except:
                        pass
            
            # 축의 collections 제거 (화살표, 패치 등)
            for collection in ax.collections[:]:
                try:
                    collection.remove()
                except:
                    pass
            
            # 축의 patches 제거
            for patch in ax.patches[:]:
                try:
                    patch.remove()
                except:
                    pass

    def clear_goal_check_plots(self):
        """goal_check 관련 모든 플롯 제거 (legacy)"""
        # clear_all_plots에서 처리됨
        pass
    
    def clear_goal_check_rectangles(self):
        """goal_check 사각형들만 전용으로 제거"""
        # ROS로 받은 goal_check 영역들 제거
        if hasattr(self, 'current_goal_check_areas'):
            for area_obj in self.current_goal_check_areas:
                if 'plot_objects' in area_obj:
                    for plot_obj in area_obj['plot_objects']:
                        try:
                            plot_obj.remove()
                        except:
                            pass
            # current_goal_check_areas 초기화
            self.current_goal_check_areas = []
        
        # ax1에서 모든 patches와 collections 제거 (사각형 관련)
        for patch in self.ax1.patches[:]:
            try:
                patch.remove()
            except:
                pass
        
        for collection in self.ax1.collections[:]:
            try:
                collection.remove()
            except:
                pass
    
    def calculate_goal_check_areas(self, current_pos, target_pos, goal_distance, goal_psi, boat_width=0.1):
        """
        goal_check에서 체크하는 영역들을 계산
        main_onnx_v5.py의 goal_check 함수와 동일한 로직
        """
        l = goal_distance
        theta = int(np.degrees(np.arctan2(boat_width/2, l))) + np.pi/2
        
        areas = []
        
        # 좌측 경계 영역들
        for i in range(0, 90 - int(theta)):
            angle = self.normalize_angle(int(goal_psi) + i)
            r = boat_width / (2 * np.cos(np.radians(i)) + 1)
            
            # 각도에 따른 영역 계산
            area_info = {
                'center': current_pos,
                'angle': angle,
                'radius': r,
                'type': 'left_boundary'
            }
            areas.append(area_info)
        
        # 전방 중앙선 영역들
        for i in range(-int(theta), int(theta) + 1):
            angle = self.normalize_angle(int(goal_psi) + i)
            
            area_info = {
                'center': current_pos,
                'angle': angle,
                'radius': l,
                'type': 'center_line'
            }
            areas.append(area_info)
        
        # 우측 경계 영역들
        for i in range(0, 90 - int(theta)):
            angle = self.normalize_angle(int(goal_psi) + 180 - i)
            r = boat_width / (2 * np.cos(np.radians(i)) + 1)
            
            area_info = {
                'center': current_pos,
                'angle': angle,
                'radius': r,
                'type': 'right_boundary'
            }
            areas.append(area_info)
        
        return areas
    
    def normalize_angle(self, angle):
        """각도를 0-359도 범위로 정규화"""
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
    
    def update_goal_check_area_from_ros(self):
        """ROS 메시지로 받은 goal_check 영역 시각화 업데이트"""
        # 먼저 이전 goal_check 영역들을 완전히 제거
        self.clear_goal_check_rectangles()
        
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
            check_x = [point[0] for point in self.current_obstacle_check_area]
            check_y = [point[1] for point in self.current_obstacle_check_area]
            
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

    def update_goal_check_area(self):
        """goal_check 영역 시각화 업데이트 (legacy)"""
        if len(self.position_history) > 0 and self.current_waypoint is not None:
            # 현재 위치와 목표 위치
            current_pos = [self.position_history[-1][0], self.position_history[-1][1]]
            target_pos = self.current_waypoint
            
            # 거리와 방향 계산
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            goal_distance = np.sqrt(dx**2 + dy**2)
            goal_psi = np.degrees(np.arctan2(dx, dy))
            goal_psi = self.normalize_angle(int(goal_psi))
            
            # goal_check 영역들 계산
            goal_check_areas = self.calculate_goal_check_areas(
                current_pos, target_pos, goal_distance, goal_psi
            )
            
            # 영역들을 시각화
            for i, area in enumerate(goal_check_areas):
                center = area['center']
                angle = area['angle']
                radius = area['radius']
                area_type = area['type']
                
                # 각도에 따른 방향 벡터 계산
                angle_rad = np.radians(angle)
                end_x = center[0] + radius * np.cos(angle_rad)
                end_y = center[1] + radius * np.sin(angle_rad)
                
                # 영역 타입에 따른 색상 설정
                if area_type == 'left_boundary':
                    color = 'orange'
                    alpha = 0.6
                elif area_type == 'center_line':
                    color = 'red'
                    alpha = 0.8
                elif area_type == 'right_boundary':
                    color = 'purple'
                    alpha = 0.6
                else:
                    color = 'gray'
                    alpha = 0.5
                
                # 선 그리기
                line = self.ax1.plot(
                    [center[0], end_x], [center[1], end_y],
                    color=color, alpha=alpha, linewidth=2, zorder=4
                )[0]
                self.goal_check_lines.append(line)
                
                # 각도와 거리 정보 표시 (일부만)
                if i % 10 == 0:  # 10개마다 하나씩만 표시
                    text_x = (center[0] + end_x) / 2
                    text_y = (center[1] + end_y) / 2
                    self.ax1.text(
                        text_x, text_y, f'{int(radius)}m',
                        fontsize=8, color=color, alpha=0.8,
                        ha='center', va='center'
                    )
    
    def update_control_output_plot(self):
        """제어 출력값 트랙바 업데이트"""
        # Linear velocity 트랙바 업데이트 (0~1 범위)
        linear_width = max(0, min(1, (self.linear_velocity + 1) / 2))  # -1~1을 0~1로 정규화
        self.linear_bar[0].set_width(linear_width)
        
        # Angular velocity 트랙바 업데이트 (-1~1 범위)
        angular_width = max(0, min(1, (self.angular_velocity + 1) / 2))  # -1~1을 0~1로 정규화
        self.angular_bar[0].set_width(angular_width)
        
        # 텍스트 업데이트
        self.linear_text.set_text(f'{self.linear_velocity:.3f}')
        self.angular_text.set_text(f'{self.angular_velocity:.3f}')
        
        # 제어 모드 표시 업데이트 (main_onnx_v5_final.py에서 받은 정보 우선)
        mode_color = "lightgray"
        display_mode = "UNKNOWN"
        
        if self.current_control_mode != "UNKNOWN":
            # main_onnx_v5_final.py에서 받은 제어 모드 정보 사용
            display_mode = self.current_control_mode
            if self.current_control_mode == "DIRECT_CONTROL":
                mode_color = "lightgreen"
            elif self.current_control_mode == "ONNX_MODEL":
                mode_color = "lightblue"
        else:
            # 기존 v5 모드 정보 사용 (fallback)
            display_mode = self.current_mode
            if self.current_mode in ["ONNX", "ONNX_BOTH", "ONNX_FORWARD", "ONNX_PATH", "ONNX_CLOSE"]:
                mode_color = "lightblue"
            elif self.current_mode in ["DIRECT", "DIRECT_CLEAR", "DIRECT_FORWARD", "DIRECT_PATH", "DIRECT_UNKNOWN"]:
                mode_color = "lightgreen"
            elif self.current_mode == "STOP":
                mode_color = "lightcoral"
            elif self.current_mode == "REACHED":
                mode_color = "lightyellow"
            elif self.current_mode == "UNKNOWN":
                mode_color = "lightgray"
        
        self.mode_text.set_text(f'Mode: {display_mode}')
        self.mode_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=mode_color, alpha=0.8))
        
        # 색상 변경 (값에 따라)
        if self.linear_velocity > 0:
            self.linear_bar[0].set_color('blue')
        elif self.linear_velocity < 0:
            self.linear_bar[0].set_color('red')
        else:
            self.linear_bar[0].set_color('gray')
            
        if self.angular_velocity > 0:
            self.angular_bar[0].set_color('green')
        elif self.angular_velocity < 0:
            self.angular_bar[0].set_color('orange')
        else:
            self.angular_bar[0].set_color('gray')
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        # goal_check 플롯들 정리
        self.clear_goal_check_plots()
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