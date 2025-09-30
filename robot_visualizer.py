#!/usr/bin/env python3
"""
VRX 로봇 실시간 시각화
- LiDAR, GPS, IMU 데이터로 로봇을 matplotlib으로 실시간 시각화
- 안정적인 GUI 표시
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from collections import deque
import threading

class RobotVisualizerNode(Node):
    """VRX 로봇 실시간 시각화 노드"""
    
    def __init__(self):
        super().__init__('robot_visualizer_node')
        
        # ROS2 서브스크라이버
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
            self.lidar_callback,
            10
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/wamv/sensors/gps/gps/fix',
            self.gps_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/wamv/sensors/imu/imu/data',
            self.imu_callback,
            10
        )
        
        # 데이터 저장
        self.lidar_data = None
        self.gps_data = None
        self.imu_data = None
        
        # 로봇 상태
        self.robot_position = [0.0, 0.0]
        self.robot_heading = 0.0
        
        # 히스토리
        self.position_history = deque(maxlen=100)
        self.heading_history = deque(maxlen=100)
        
        # matplotlib 설정
        self.setup_matplotlib()
        
        # 웨이포인트 퍼블리셔 (클릭한 점을 v3로 전송)
        self.waypoint_pub = self.create_publisher(Point, '/vrx/waypoint', 10)
        
        # 웨이포인트 관련 변수
        self.waypoints = []  # 클릭한 웨이포인트들 저장
        self.current_waypoint = None
        
        # 타이머로 주기적 업데이트
        self.timer = self.create_timer(0.1, self.update_plots)  # 10Hz 업데이트
        
        self.get_logger().info('🤖 VRX 로봇 실시간 시각화 시작!')
        self.get_logger().info('🖱️  궤적 플롯에서 클릭하여 웨이포인트를 설정하세요!')
    
    def setup_matplotlib(self):
        """matplotlib 설정"""
        # Figure와 subplot 생성
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('VRX Robot Real-time Visualization', fontsize=16, fontweight='bold')
        
        # 1. 로봇 위치 및 LiDAR (상단 왼쪽)
        self.ax1.set_title('Robot Position & LiDAR Data', fontsize=12)
        self.ax1.set_xlabel('X Position (m)')
        self.ax1.set_ylabel('Y Position (m)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        
        # 2. GPS 위치 (상단 오른쪽)
        self.ax2.set_title('GPS Position', fontsize=12)
        self.ax2.set_xlabel('Longitude')
        self.ax2.set_ylabel('Latitude')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal')
        
        # 3. IMU 헤딩 (하단 왼쪽)
        self.ax3.set_title('IMU Heading (Yaw)', fontsize=12)
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Heading (degrees)')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(-180, 180)
        
        # 4. 위치 궤적 (하단 오른쪽)
        self.ax4.set_title('Position Trajectory', fontsize=12)
        self.ax4.set_xlabel('X (m)')
        self.ax4.set_ylabel('Y (m)')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.set_aspect('equal')
        
        # 초기 플롯 요소들
        self.robot_marker, = self.ax1.plot([], [], 'ro', markersize=15, label='Robot')
        self.robot_arrow = None
        self.lidar_points, = self.ax1.plot([], [], 'b.', markersize=2, label='LiDAR')
        self.trajectory_line, = self.ax4.plot([], [], 'b-', linewidth=2, label='Trajectory')
        self.current_pos, = self.ax4.plot([], [], 'ro', markersize=10, label='Current Position')
        self.gps_marker, = self.ax2.plot([], [], 'go', markersize=10, label='GPS Position')
        self.heading_line, = self.ax3.plot([], [], 'g-', linewidth=2, label='Heading')
        
        # 범례
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
        self.ax4.legend()
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 마우스 클릭 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 웨이포인트 표시용 플롯 요소
        self.waypoint_markers, = self.ax4.plot([], [], 'ro', markersize=8, label='Waypoints')
        
        # 창 표시
        plt.ion()  # 인터랙티브 모드
        plt.show(block=False)
        plt.pause(0.1)
        
        self.get_logger().info('✅ matplotlib 창이 표시되었습니다!')
    
    def on_click(self, event):
        """마우스 클릭 이벤트 핸들러"""
        if event.inaxes == self.ax4 and event.button == 1:  # 좌클릭, 궤적 플롯에서만
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
    
    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백"""
        ranges = np.array(msg.ranges)
        # 유효한 범위만 필터링
        valid_mask = np.isfinite(ranges) & (ranges > 0) & (ranges < 50)
        valid_ranges = ranges[valid_mask]
        valid_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))[valid_mask]
        
        self.lidar_data = {
            'ranges': valid_ranges,
            'angles': valid_angles,
            'timestamp': time.time()
        }
    
    def gps_callback(self, msg):
        """GPS 데이터 콜백"""
        if msg.latitude != 0.0 and msg.longitude != 0.0:
            # 간단한 변환 (실제로는 UTM 변환 필요)
            x = (msg.longitude - 151.2) * 111000
            y = (msg.latitude - -33.8) * 111000
            
            self.gps_data = {
                'latitude': msg.latitude,
                'longitude': msg.longitude,
                'x': x,
                'y': y,
                'timestamp': time.time()
            }
            
            # 로봇 위치 업데이트
            self.robot_position = [x, y]
            self.position_history.append([x, y])
    
    def imu_callback(self, msg):
        """IMU 데이터 콜백"""
        orientation = msg.orientation
        yaw = self.quaternion_to_yaw(orientation)
        
        self.imu_data = {
            'orientation': orientation,
            'yaw': yaw,
            'yaw_degrees': np.degrees(yaw),
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration,
            'timestamp': time.time()
        }
        
        # 로봇 헤딩 업데이트
        self.robot_heading = np.degrees(yaw)
        self.heading_history.append(self.robot_heading)
    
    def quaternion_to_yaw(self, orientation):
        """쿼터니언을 Yaw 각도로 변환"""
        w = orientation.w
        x = orientation.x
        y = orientation.y
        z = orientation.z
        
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw
    
    def update_plots(self):
        """플롯 업데이트 (타이머용)"""
        try:
            # 1. 로봇 위치 및 LiDAR 업데이트
            self.update_robot_plot()
            
            # 2. GPS 위치 업데이트
            self.update_gps_plot()
            
            # 3. IMU 헤딩 업데이트
            self.update_heading_plot()
            
            # 4. 위치 궤적 업데이트
            self.update_trajectory_plot()
            
            # 5. 웨이포인트 표시 업데이트
            self.update_waypoints_plot()
            
            # 화면 업데이트
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f'플롯 업데이트 오류: {e}')
    
    def update_robot_plot(self):
        """로봇 위치 및 LiDAR 플롯 업데이트"""
        if self.robot_position:
            # 로봇 위치 업데이트
            self.robot_marker.set_data([self.robot_position[0]], [self.robot_position[1]])
            
            # 로봇 방향 화살표 업데이트
            arrow_length = 5.0
            dx = arrow_length * np.sin(np.radians(self.robot_heading))
            dy = arrow_length * np.cos(np.radians(self.robot_heading))
            
            # 기존 화살표 제거
            if hasattr(self, 'robot_arrow') and self.robot_arrow:
                self.robot_arrow.remove()
            
            # 새 화살표 추가
            self.robot_arrow = self.ax1.arrow(
                self.robot_position[0], self.robot_position[1], 
                dx, dy, 
                head_width=1.0, head_length=1.0, 
                fc='red', ec='red', alpha=0.8, linewidth=3
            )
            
            # LiDAR 데이터 업데이트
            if self.lidar_data is not None:
                ranges = self.lidar_data['ranges']
                angles = self.lidar_data['angles']
                
                # LiDAR 데이터를 로봇 기준 좌표로 변환
                lidar_x = self.robot_position[0] + ranges * np.sin(angles + np.radians(self.robot_heading))
                lidar_y = self.robot_position[1] + ranges * np.cos(angles + np.radians(self.robot_heading))
                
                self.lidar_points.set_data(lidar_x, lidar_y)
                
                # 축 범위 자동 조정
                if len(lidar_x) > 0:
                    x_margin = 20
                    y_margin = 20
                    self.ax1.set_xlim(self.robot_position[0] - x_margin, self.robot_position[0] + x_margin)
                    self.ax1.set_ylim(self.robot_position[1] - y_margin, self.robot_position[1] + y_margin)
    
    def update_gps_plot(self):
        """GPS 플롯 업데이트"""
        if self.gps_data is not None:
            lat = self.gps_data['latitude']
            lon = self.gps_data['longitude']
            
            self.gps_marker.set_data([lon], [lat])
            
            # 축 범위 자동 조정
            if len(self.position_history) > 1:
                positions = np.array(self.position_history)
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]
                
                if len(x_coords) > 0:
                    x_margin = (np.max(x_coords) - np.min(x_coords)) * 0.1
                    y_margin = (np.max(y_coords) - np.min(y_coords)) * 0.1
                    
                    self.ax2.set_xlim(np.min(x_coords) - x_margin, np.max(x_coords) + x_margin)
                    self.ax2.set_ylim(np.min(y_coords) - y_margin, np.max(y_coords) + y_margin)
    
    def update_heading_plot(self):
        """헤딩 플롯 업데이트"""
        if len(self.heading_history) > 0:
            time_axis = np.arange(len(self.heading_history))
            self.heading_line.set_data(time_axis, list(self.heading_history))
            
            # X축 범위 조정 (최근 100개 포인트만 표시)
            if len(self.heading_history) > 100:
                self.ax3.set_xlim(len(self.heading_history) - 100, len(self.heading_history))
            else:
                self.ax3.set_xlim(0, max(100, len(self.heading_history)))
    
    def update_trajectory_plot(self):
        """궤적 플롯 업데이트"""
        if len(self.position_history) > 1:
            positions = np.array(self.position_history)
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            self.trajectory_line.set_data(x_coords, y_coords)
            
            if len(x_coords) > 0:
                self.current_pos.set_data([x_coords[-1]], [y_coords[-1]])
                
                # 축 범위 자동 조정
                x_margin = (np.max(x_coords) - np.min(x_coords)) * 0.1
                y_margin = (np.max(y_coords) - np.min(y_coords)) * 0.1
                
                if x_margin > 0 and y_margin > 0:
                    self.ax4.set_xlim(np.min(x_coords) - x_margin, np.max(x_coords) + x_margin)
                    self.ax4.set_ylim(np.min(y_coords) - y_margin, np.max(y_coords) + y_margin)
    
    def update_waypoints_plot(self):
        """웨이포인트 플롯 업데이트"""
        if len(self.waypoints) > 0:
            waypoints_array = np.array(self.waypoints)
            self.waypoint_markers.set_data(waypoints_array[:, 0], waypoints_array[:, 1])
            
            # 현재 웨이포인트 강조 표시
            if self.current_waypoint:
                # 기존 현재 웨이포인트 마커 제거
                if hasattr(self, 'current_waypoint_marker'):
                    self.current_waypoint_marker.remove()
                
                # 현재 웨이포인트 마커 추가
                self.current_waypoint_marker, = self.ax4.plot(
                    [self.current_waypoint[0]], [self.current_waypoint[1]], 
                    'rs', markersize=12, markeredgecolor='black', markeredgewidth=2,
                    label='Current Waypoint'
                )
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        plt.close('all')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RobotVisualizerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
####sssszx