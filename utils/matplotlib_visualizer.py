"""
Matplotlib 기반 시각화 모듈
- 배의 위치와 헤딩 정보 시각화
- 주변 장애물의 2D 정보 표시
- 직교좌표계와 원형좌표계 동시 표시
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import threading
import time
from collections import deque

class MatplotlibVisualizer(Node):
    """Matplotlib 기반 시각화 노드"""
    
    def __init__(self):
        super().__init__('matplotlib_visualizer')
        
        # ROS2 서브스크라이버
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/wamv/pose',
            self.pose_callback,
            10
        )
        
        self.tracking_sub = self.create_subscription(
            Float32MultiArray,
            '/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions',
            self.tracking_callback,
            10
        )
        
        # 데이터 저장
        self.boat_pose = None
        self.boat_heading = 0.0
        self.boat_position = [0.0, 0.0]
        self.obstacles = []
        self.target_heading = 0.0
        
        # 히스토리 저장 (궤적 표시용)
        self.position_history = deque(maxlen=100)
        self.heading_history = deque(maxlen=100)
        
        # Matplotlib 설정
        self.setup_matplotlib()
        
        # 애니메이션 시작
        self.start_animation()
        
        self.get_logger().info('📊 Matplotlib 시각화 노드 시작!')
    
    def setup_matplotlib(self):
        """Matplotlib 설정"""
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # Figure와 subplot 생성
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.suptitle('VRX Robot Navigation Visualization', fontsize=16, fontweight='bold')
        
        # 왼쪽 subplot: 직교좌표계 (배 위치 및 장애물)
        self.ax1.set_title('Boat Position & Obstacles (Cartesian)', fontsize=12)
        self.ax1.set_xlabel('X Position (m)')
        self.ax1.set_ylabel('Y Position (m)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        
        # 오른쪽 subplot: 원형좌표계 (헤딩 및 장애물 방향)
        self.ax2.set_title('Heading & Obstacles (Polar)', fontsize=12)
        self.ax2.set_xlim(-1.2, 1.2)
        self.ax2.set_ylim(-1.2, 1.2)
        self.ax2.set_aspect('equal')
        self.ax2.grid(True, alpha=0.3)
        
        # 원형좌표계 그리드 설정
        self.setup_polar_grid()
        
        # 초기 플롯 요소들
        self.boat_marker, = self.ax1.plot([], [], 'bo', markersize=10, label='Boat')
        self.boat_arrow = self.ax1.arrow(0, 0, 0, 0, head_width=0.5, head_length=0.3, fc='blue', ec='blue')
        self.trajectory_line, = self.ax1.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Trajectory')
        self.obstacle_markers = []
        
        # 원형좌표계 요소들
        self.heading_arrow = self.ax2.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=3)
        self.target_arrow = self.ax2.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=3)
        self.obstacle_arrows = []
        
        # 범례 설정
        self.ax1.legend(loc='upper right')
        self.ax2.legend(['Current Heading', 'Target Heading', 'Obstacles'], loc='upper right')
        
        # 플롯 초기화
        self.ax1.set_xlim(-50, 50)
        self.ax1.set_ylim(-50, 50)
    
    def setup_polar_grid(self):
        """원형좌표계 그리드 설정"""
        # 원형 그리드 그리기
        for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.3, linestyle='--')
            self.ax2.add_patch(circle)
        
        # 방향 표시 (북, 동, 남, 서)
        directions = ['N', 'E', 'S', 'W']
        angles = [90, 0, -90, 180]  # 도 단위
        
        for direction, angle in zip(directions, angles):
            x = 1.1 * np.cos(np.radians(angle))
            y = 1.1 * np.sin(np.radians(angle))
            self.ax2.text(x, y, direction, ha='center', va='center', fontsize=12, fontweight='bold')
    
    def pose_callback(self, msg):
        """배의 위치 및 헤딩 정보 콜백"""
        # 위치 정보
        self.boat_position = [msg.pose.position.x, msg.pose.position.y]
        
        # 헤딩 정보 (쿼터니언에서 오일러 각으로 변환)
        orientation = msg.pose.orientation
        self.boat_heading = self.quaternion_to_yaw(orientation)
        
        # 히스토리 저장
        self.position_history.append(self.boat_position.copy())
        self.heading_history.append(self.boat_heading)
        
        self.get_logger().debug(f'배 위치: ({self.boat_position[0]:.2f}, {self.boat_position[1]:.2f}), 헤딩: {np.degrees(self.boat_heading):.1f}°')
    
    def tracking_callback(self, msg):
        """장애물 추적 정보 콜백"""
        if len(msg.data) >= 7:
            # 데이터 파싱: [red_x, red_y, red_depth, green_x, green_y, green_depth, timestamp]
            red_x = msg.data[0]
            red_y = msg.data[1]
            red_depth = msg.data[2]
            green_x = msg.data[3]
            green_y = msg.data[4]
            green_depth = msg.data[5]
            
            # 장애물 정보 업데이트
            self.obstacles = []
            
            if red_x > 0 and red_y > 0:
                # 이미지 좌표를 실제 좌표로 변환 (간단한 변환)
                real_x, real_y = self.image_to_real_coords(red_x, red_y, red_depth)
                self.obstacles.append({
                    'type': 'red_buoy',
                    'position': [real_x, real_y],
                    'depth': red_depth,
                    'color': 'red'
                })
            
            if green_x > 0 and green_y > 0:
                real_x, real_y = self.image_to_real_coords(green_x, green_y, green_depth)
                self.obstacles.append({
                    'type': 'green_buoy',
                    'position': [real_x, real_y],
                    'depth': green_depth,
                    'color': 'green'
                })
    
    def image_to_real_coords(self, img_x, img_y, depth):
        """이미지 좌표를 실제 좌표로 변환"""
        # 간단한 변환 (실제로는 카메라 캘리브레이션 필요)
        # 이미지 중앙을 기준으로 변환
        img_center_x = 640  # 이미지 중앙
        img_center_y = 360
        
        # 픽셀 오프셋 계산
        offset_x = img_x - img_center_x
        offset_y = img_center_y - img_y  # Y축 뒤집기
        
        # 실제 좌표로 변환 (깊이 기반)
        scale_factor = depth * 0.001  # 스케일 팩터 조정
        real_x = self.boat_position[0] + offset_x * scale_factor
        real_y = self.boat_position[1] + offset_y * scale_factor
        
        return real_x, real_y
    
    def quaternion_to_yaw(self, orientation):
        """쿼터니언을 Yaw 각도로 변환"""
        # ZYX 오일러 각도 변환
        w = orientation.w
        x = orientation.x
        y = orientation.y
        z = orientation.z
        
        # Yaw 계산
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw
    
    def update_plot(self, frame):
        """플롯 업데이트 (애니메이션용)"""
        # 직교좌표계 업데이트
        self.update_cartesian_plot()
        
        # 원형좌표계 업데이트
        self.update_polar_plot()
        
        return [self.boat_marker, self.trajectory_line] + self.obstacle_markers + [self.heading_arrow, self.target_arrow] + self.obstacle_arrows
    
    def update_cartesian_plot(self):
        """직교좌표계 플롯 업데이트"""
        if self.boat_position:
            # 배 위치 업데이트
            self.boat_marker.set_data([self.boat_position[0]], [self.boat_position[1]])
            
            # 배 방향 화살표 업데이트
            arrow_length = 2.0
            dx = arrow_length * np.cos(self.boat_heading)
            dy = arrow_length * np.sin(self.boat_heading)
            
            # 기존 화살표 제거
            if hasattr(self, 'boat_arrow'):
                self.boat_arrow.remove()
            
            # 새 화살표 추가
            self.boat_arrow = self.ax1.arrow(
                self.boat_position[0], self.boat_position[1], 
                dx, dy, 
                head_width=0.5, head_length=0.3, 
                fc='blue', ec='blue', alpha=0.8
            )
            
            # 궤적 업데이트
            if len(self.position_history) > 1:
                positions = list(self.position_history)
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                self.trajectory_line.set_data(x_coords, y_coords)
            
            # 장애물 마커 업데이트
            # 기존 장애물 마커 제거
            for marker in self.obstacle_markers:
                marker.remove()
            self.obstacle_markers.clear()
            
            # 새 장애물 마커 추가
            for obstacle in self.obstacles:
                pos = obstacle['position']
                color = obstacle['color']
                marker = self.ax1.scatter(pos[0], pos[1], c=color, s=100, alpha=0.7, 
                                        marker='o', edgecolors='black', linewidth=1)
                self.obstacle_markers.append(marker)
                
                # 장애물 정보 텍스트
                self.ax1.text(pos[0], pos[1] + 1, f"{obstacle['type']}\n{obstacle['depth']:.2f}m", 
                            ha='center', va='bottom', fontsize=8, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def update_polar_plot(self):
        """원형좌표계 플롯 업데이트"""
        # 현재 헤딩 화살표 업데이트
        if hasattr(self, 'heading_arrow'):
            self.heading_arrow.remove()
        
        heading_x = 0.8 * np.cos(self.boat_heading)
        heading_y = 0.8 * np.sin(self.boat_heading)
        
        self.heading_arrow = self.ax2.arrow(
            0, 0, heading_x, heading_y,
            head_width=0.1, head_length=0.1,
            fc='red', ec='red', linewidth=3, alpha=0.8
        )
        
        # 목표 헤딩 화살표 (예시: 북쪽)
        target_heading = 0.0  # 북쪽
        if hasattr(self, 'target_arrow'):
            self.target_arrow.remove()
        
        target_x = 0.6 * np.cos(target_heading)
        target_y = 0.6 * np.sin(target_heading)
        
        self.target_arrow = self.ax2.arrow(
            0, 0, target_x, target_y,
            head_width=0.1, head_length=0.1,
            fc='green', ec='green', linewidth=3, alpha=0.8
        )
        
        # 장애물 방향 화살표 업데이트
        # 기존 장애물 화살표 제거
        for arrow in self.obstacle_arrows:
            arrow.remove()
        self.obstacle_arrows.clear()
        
        # 새 장애물 화살표 추가
        for obstacle in self.obstacles:
            # 장애물의 상대적 방향 계산
            boat_pos = np.array(self.boat_position)
            obstacle_pos = np.array(obstacle['position'])
            relative_pos = obstacle_pos - boat_pos
            
            # 거리와 각도 계산
            distance = np.linalg.norm(relative_pos)
            if distance > 0:
                angle = np.arctan2(relative_pos[1], relative_pos[0])
                
                # 화살표 길이 (거리에 반비례, 최대 0.5)
                arrow_length = min(0.5, 2.0 / distance)
                
                arrow_x = arrow_length * np.cos(angle)
                arrow_y = arrow_length * np.sin(angle)
                
                color = obstacle['color']
                arrow = self.ax2.arrow(
                    0, 0, arrow_x, arrow_y,
                    head_width=0.05, head_length=0.05,
                    fc=color, ec=color, linewidth=2, alpha=0.7
                )
                self.obstacle_arrows.append(arrow)
    
    def start_animation(self):
        """애니메이션 시작"""
        # matplotlib 백엔드를 설정 (GUI 환경에서 안전하게 실행)
        import matplotlib
        matplotlib.use('TkAgg')  # Tkinter 백엔드 사용
        
        # 애니메이션 설정
        self.animation = FuncAnimation(
            self.fig, self.update_plot, 
            interval=100,  # 100ms 간격
            blit=False,    # blit 비활성화 (화살표 때문에)
            cache_frame_data=False
        )
        
        # GUI 이벤트 루프를 별도 스레드에서 실행
        def run_matplotlib():
            try:
                plt.tight_layout()
                plt.show(block=False)  # non-blocking 모드로 실행
            except Exception as e:
                self.get_logger().error(f"Matplotlib GUI 오류: {e}")
        
        matplotlib_thread = threading.Thread(target=run_matplotlib, daemon=True)
        matplotlib_thread.start()
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        plt.close('all')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MatplotlibVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
