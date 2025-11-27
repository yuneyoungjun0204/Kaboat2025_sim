import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
from std_msgs.msg import Bool, Float32, Float64MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from threading import Lock

class MissionMsgPublisherforPx4(Node):
    """
    A ROS 2 node that sends Offboard control commands to PX4.
    It subscribes to obstacle avoidance and auto-berthing states,
    and publishes appropriate TrajectorySetpoint and OffboardControlMode messages.
    """
    
    # --- Constants ---
    # 50Hz (Now managed as a parameter)

    # PX4 uORB Topic Names
    TOPIC_OFFBOARD_CONTROL_MODE = '/fmu/in/offboard_control_mode'
    TOPIC_TRAJECTORY_SETPOINT = '/fmu/in/trajectory_setpoint'

    # Subscriber Topic Names
    TOPIC_WP_CMD = '/obstacle_detected'
    TOPIC_PC_CMD = '/avoid_yaw'
    TOPIC_CONTROL_FLAG = '/berthing_det_flag'

    def __init__(self):
        """Initializes the node, setting up publishers, subscribers, timer, and lock."""
        super().__init__('mission_msg_pub_px4')

        # --- Parameter Declaration ---
        self.declare_parameter('timer_period', 0.02)  # [sec] Timer period (Default: 0.02s = 50Hz)
        
        # Get parameter value
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value

        # Reliable QoS profile for control messages
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Lock for data protection in a multi-threaded environment
        self.data_lock = Lock()

        # Data initialization
        with self.data_lock:
            self.ctrl_flag = False
            self.u_d = 0.0
            self.yaw_d = 0.0
            self.x_e_d = 0.0
            self.y_e_d = 0.0

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, self.TOPIC_OFFBOARD_CONTROL_MODE, px4_qos)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, self.TOPIC_TRAJECTORY_SETPOINT, px4_qos)

        # # Create subscribers
        # self.create_subscription(Bool, self.TOPIC_CONTROL_FLAG, self.control_flag_callback, 1)
        # self.create_subscription(Float64MultiArray, self.TOPIC_WP_CMD, self.wp_cmd_callback, 1)
        # self.create_subscription(Float64MultiArray, self.TOPIC_PC_CMD, self.pc_cmd_callback, 1)
        
        # Timer for periodic message publishing (uses parameter value)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Mission message publisher for PX4 has been initialized with {1.0/timer_period:.1f} Hz rate.")

    # # --- Subscriber Callbacks ---
    # def control_flag_callback(self, msg):
    #     with self.data_lock:
    #         self.ctrl_flag = msg.data

    # def wp_cmd_callback(self, msg):
    #     with self.data_lock:
    #         self.u_d = msg.data[0]
    #         self.yaw_d = msg.data[1]

    # def pc_cmd_callback(self, msg):
    #     with self.data_lock:
    #         self.x_e_d = msg.data[0]
    #         self.y_e_d = msg.data[1]


    def publish_offboard_control_mode(self, timestamp: int):
        """Publishes the OffboardControlMode message."""
        msg = OffboardControlMode()
        # with self.data_lock:
        #     msg.position = self.ctrl_flag
        msg.position = False
        msg.attitude = False
        msg.velocity = False
        msg.acceleration = False
        msg.body_rate = False
        msg.timestamp = timestamp 
        self.offboard_control_mode_publisher.publish(msg)
        self.get_logger().info(
            f"Publishing Offboard Mode: Position={msg.position}, Attitude={msg.attitude}", 
            throttle_duration_sec=1
        ) 

    def publish_trajectory_setpoint(self, timestamp: int):
        """Publishes the target position and Yaw setpoint."""
        msg = TrajectorySetpoint()
        
        msg.position = [0.0, 0.0, 0.0]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.yaw = 90.0
        msg.timestamp = timestamp 
        self.trajectory_setpoint_publisher.publish(msg)
        
        log_message = (
            f"Publishing setpoint:"
        )
        self.get_logger().info(log_message, throttle_duration_sec=1) 

    def timer_callback(self):
        """Periodically called by the timer to publish control messages."""
        current_time_us = int(self.get_clock().now().nanoseconds / 1000)
        self.publish_offboard_control_mode(current_time_us)
        self.publish_trajectory_setpoint(current_time_us)

def main(args=None):
    rclpy.init(args=args)
    node = MissionMsgPublisherforPx4()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        node.get_logger().error(f"Executor failed: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()