#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import threading
from math import sin, cos, atan2, pi
import numpy as np
import open3d as o3d
from pyproj import Transformer, Proj

from scipy.ndimage import median_filter # [VO_LITE] Import spatial filter

from std_msgs.msg import Float32, Bool, Header, ColorRGBA
from px4_msgs.msg import VehicleGlobalPosition, PositionSetpointTriplet, VehicleLocalPosition, ModeFlag
from sensor_msgs.msg import PointCloud2, LaserScan 
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point 

class USVGuidanceAndAvoidanceNode(Node):
    def __init__(self):
        super().__init__('usv_guidance_and_avoidance_node')
        
        # === QoS Profiles ===
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        livox_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)

        # === LLA to NED Converter (Optimized) ===
        self.proj_lla = Proj(proj='latlong', ellps='WGS84')
        self.proj_ecef = Proj(proj='geocent', ellps='WGS84')
        self.transformer_to_ecef = Transformer.from_proj(self.proj_lla, self.proj_ecef)
        self.ref_x, self.ref_y, self.ref_z = self.transformer_to_ecef.transform(self.lon_origin, self.lat_origin, self.alt_origin)
        
        lat_rad, lon_rad = np.radians(self.lat_origin), np.radians(self.lon_origin)
        self.R_ecef_to_ned = np.array([
            [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad),  np.cos(lat_rad)],
            [-np.sin(lon_rad),                   np.cos(lon_rad),                    0],
            [-np.cos(lat_rad) * np.cos(lon_rad), -np.cos(lat_rad) * np.sin(lon_rad), -np.sin(lat_rad)]
        ])

        # === Class Variables (State & Parameters) ===
        self.current_pose_lla = None
        self.current_pose_ned = None
        self.current_yaw = None
        self.current_velocity_xy = (0.0, 0.0) # (vx, vy) in NED
        self.goal = [0.0, 0.0]
        self.wp_ned = []
        self.last_waypoint = [0, 0]
        self.final_yaw_cmd = 0.0
        self.smoothed_yaw_cmd = 0.0
        self.last_frame_id = 'map'
        self.angle_min = -pi
        self.angle_increment = np.radians(1)
        self.num_bins = int(2 * pi / self.angle_increment)
        self.bin_centers = self.angle_min + np.arange(self.num_bins) * self.angle_increment
        self.mode_f = True
        self.is_obstacle_detected = False
        self.min_distance = float('inf')
        self.vfh_ranges = np.full(self.num_bins, self.lidar_processing_range) 
        self.last_lidar_stamp = None
        self.last_lidar_header = None
        
     
        # === Subscribers & Publishers ===
        self.create_subscription(VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.odom_callback, px4_qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, px4_qos)
        self.create_subscription(PositionSetpointTriplet, '/fmu/out/position_setpoint_triplet', self.wp_callback, px4_qos)
        self.create_subscription(ModeFlag, '/fmu/out/mode_flag', self.mode_callback, px4_qos)
        self.create_subscription(PointCloud2, '/livox/lidar', self.lidar_callback, livox_qos)

        
        # [EXECUTION_FLAG]
        # Create timer or log based on the flag value
        if self.use_timer_execution:
            if self.control_loop_rate <= 0:
                self.get_logger().warn("control_loop_rate_hz must be positive. Defaulting to 20.0 Hz.")
                self.control_loop_rate = 20.0
            
            timer_period_sec = 1.0 / self.control_loop_rate
            self.control_timer = self.create_timer(timer_period_sec, self.control_loop_callback)
            self.get_logger().info(f"Execution mode: TIMER-BASED (running at {self.control_loop_rate} Hz)")
        else:
            self.control_timer = None # Explicitly state no timer
            self.get_logger().info("Execution mode: DATA-DRIVEN (bag file mode)")
        
        self.get_logger().info("USV Guidance and Avoidance Node (Multi-Threaded, [FINAL_INTELLIGENT, VO_LITE]) started successfully.")


    # =================================================================
    #                       Helper Functions
    # =================================================================
    def lla_to_ned(self, lat, lon, alt):
        x, y, z = self.transformer_to_ecef.transform(lon, lat, alt)
        return self.R_ecef_to_ned @ np.array([x - self.ref_x, y - self.ref_y, z - self.ref_z])

    def _cost_to_color(self, cost, min_cost, max_cost):
        norm_cost = max(0.0, min(1.0, (cost - min_cost) / (max_cost - min_cost + 1e-6)))
        r = norm_cost
        g = 1.0 - norm_cost
        b = 0.0
        return (r, g, b)

    def pi2pi(self, rad): return (rad + pi) % (2 * pi) - pi
    
    def numpy_xyz_to_cloud(self, xyz, frame_id):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=frame_id)
        return point_cloud2.create_cloud_xyz32(header, xyz[:, :3].tolist())
        
    def yaw_to_quat_z(self, yaw): return (0.0, 0.0, np.sin(yaw * 0.5), np.cos(yaw * 0.5))

    # =================================================================
    #                  Thread-Safe Callback Functions
    # =================================================================
    
    def odom_callback(self, msg):
        north, east, down = self.lla_to_ned(msg.lat, msg.lon, msg.alt)
        with self.state_lock:
            self.current_pose_ned = (north, east, down)

    def vehicle_local_position_callback(self, msg):
        with self.state_lock:
            self.current_yaw = msg.heading
            self.current_velocity_xy = (msg.vx, msg.vy)

    def wp_callback(self, msg):
        with self.state_lock:
            current_pos_ned_safe = self.current_pose_ned
            alt_approx = current_pos_ned_safe[2] if current_pos_ned_safe else self.alt_origin
            
            if np.isnan(msg.previous.lat) or msg.previous.lat == 0.0 or not current_pos_ned_safe:
                n0, e0, d0 = current_pos_ned_safe or (0.0, 0.0, 0.0)
            else:
                n0, e0, d0 = self.lla_to_ned(msg.previous.lat, msg.previous.lon, alt_approx)
            
            if np.isnan(msg.current.lat) or msg.current.lat == 0.0 or not current_pos_ned_safe:
                n1, e1, d1 = current_pos_ned_safe or (0.0, 0.0, 0.0)
            else:
                n1, e1, d1 = self.lla_to_ned(msg.current.lat, msg.current.lon, alt_approx)
            
            self.wp_ned = [n0, e0, d0, n1, e1, d1]
            self.last_waypoint = [n0, e0]
            self.goal = [n1, e1]
    
    def mode_callback(self, msg):
        if msg.modethree + msg.modefour + msg.modefive > 0.5:
            m_flags = False 
        else:
            m_flags = True 
        with self.vfh_lock:
            self.mode_f = m_flags

    def _reset_detection_unsafe(self):
        # (Assumes the calling function already acquired vfh_lock)
        # [VO_LITE] vfh_ranges initialization logic removed -> managed directly in lidar_callback
        self.is_obstacle_detected = False
        self.min_distance = float('inf')

    def lidar_callback(self, msg):
        with self.vfh_lock:
            mode = self.mode_f
        
        if not mode:
            self.publish_visuals([], [], msg.header.frame_id) 
            self.obstacle_detected_publisher.publish(Bool(data=False))
            return

        try:
            points_generator = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array([[p[0], p[1], p[2]] for p in points_generator], dtype=np.float32)

        except Exception as e:
            self.get_logger().error(f"Failed to read points from PointCloud2 message: {e}")
            return

        mask_inside_footprint = (points[:, 0] > self.fp_x_min) & (points[:, 0] < self.fp_x_max) & \
                                (points[:, 1] > self.fp_y_min) & (points[:, 1] < self.fp_y_max)
        points = points[~mask_inside_footprint]

        if points.shape[0] == 0:
            current_stamp_ns = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
            with self.vfh_lock: 
                # [VO_LITE] Process empty scans for v_app calculation
                ranges = np.full(self.num_bins, self.lidar_processing_range)
                dt_ns = current_stamp_ns - self.prev_lidar_stamp_ns
                if dt_ns > 10_000_000: # 10ms
                    dt_sec = dt_ns / 1_000_000_000.0
                    delta_ranges = ranges - self.prev_vfh_ranges
                    v_app = delta_ranges / dt_sec
                    v_app_filtered = median_filter(v_app, size=3, mode='wrap')
                    self.apparent_velocities = 0.7 * self.apparent_velocities + 0.3 * v_app_filtered
                
                self._reset_detection_unsafe()
                self.vfh_ranges = ranges # [VO_LITE] Save empty scan
                self.last_lidar_header = msg.header
                self.last_lidar_stamp = current_stamp_ns
                self.prev_vfh_ranges = np.copy(ranges)
                self.prev_lidar_stamp_ns = current_stamp_ns

            self.publish_visuals([], [], msg.header.frame_id)
            
            if not self.use_timer_execution:
                self.control_loop_callback()
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if self.voxel_size > 0.0:
            pcd = pcd.voxel_down_sample(self.voxel_size)
        pts = np.asarray(pcd.points)
        
        mask = (pts[:, 2] > self.ground_z_thresh) & \
               (pts[:, 2] < self.ceiling_z_thresh) & \
               (np.hypot(pts[:, 0], pts[:, 1]) < self.lidar_processing_range)
        
        obstacle_points = pts[mask]
        
        ranges = np.full(self.num_bins, self.lidar_processing_range) 
        


   
# =================================================================
#                         Main Function
# =================================================================
def main(args=None):
    rclpy.init(args=args)
    
    from rclpy.parameter import Parameter
    node = USVGuidanceAndAvoidanceNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT) received. Shutting down...')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()