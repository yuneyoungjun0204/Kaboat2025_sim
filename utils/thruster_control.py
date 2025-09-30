"""
스러스터 제어 모듈
- ROS2를 통한 스러스터 명령 전송
- 로봇 제어 명령 퍼블리시
"""

import rclpy
from std_msgs.msg import Float64, String

class ThrusterController:
    """스러스터 제어기"""
    
    def __init__(self, node):
        self.node = node
        
        # ROS2 퍼블리셔
        self.left_thrust_pub = self.node.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.node.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.node.create_publisher(String, '/detection/status', 10)
        self.target_x_pub = self.node.create_publisher(Float64, '/approach/target_x', 10)
        
        self.node.get_logger().info('🚀 스러스터 컨트롤러 초기화 완료')
    
    def publish_thrust_commands(self, left_thrust, right_thrust):
        """스러스터 명령 퍼블리시"""
        left_msg = Float64()
        left_msg.data = float(left_thrust)
        self.left_thrust_pub.publish(left_msg)
        
        right_msg = Float64()
        right_msg.data = float(right_thrust)
        self.right_thrust_pub.publish(right_msg)
    
    def publish_target_x(self, target_x):
        """목표 X값 퍼블리시"""
        target_msg = Float64()
        target_msg.data = float(target_x)
        self.target_x_pub.publish(target_msg)
    
    def publish_status(self, status_text):
        """상태 정보 퍼블리시"""
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
    
    def stop_robot(self):
        """로봇 정지"""
        self.publish_thrust_commands(0.0, 0.0)
        self.publish_status("ROBOT STOPPED")
    
    def emergency_stop(self):
        """비상 정지"""
        self.publish_thrust_commands(0.0, 0.0)
        self.publish_status("EMERGENCY STOP")
        self.node.get_logger().warn('🚨 비상 정지!')
