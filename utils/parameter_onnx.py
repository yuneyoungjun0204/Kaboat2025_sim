#!/usr/bin/env python3
"""
ONNX Controller Sensor Parameter Manager
Handles all sensor data subscriptions and processing for VRX ONNX Controller
"""

import numpy as np
import time
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from geometry_msgs.msg import Point
from utils import SensorDataManager


class ONNXSensorManager:
    """Manages all sensor data for ONNX controller"""

    # Constants
    MAX_LIDAR_DISTANCE = 100.0
    LIDAR_SIZE = 201
    LIDAR_ANGLE_RANGE = 100

    def __init__(self, node):
        """
        Initialize sensor manager
        Args:
            node: ROS2 node instance to create subscriptions
        """
        self.node = node
        self.sensor_manager = SensorDataManager()

        # Sensor data
        self.lidar_distances = np.full(self.LIDAR_SIZE, self.MAX_LIDAR_DISTANCE, dtype=np.float32)
        self.agent_heading = 0.0
        self.angular_velocity_y = 0.0
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.previous_angular_velocity = np.zeros(3)
        self.last_angular_velocity_update_time = 0.0

        # Waypoint management
        self.waypoints = []
        self.current_target_index = 0
        self.waypoint_reached = False

        # Scaling factors
        self.angular_velocity_y_scale = 1
        self.lidar_scale_factor = 1.0

        # Control callback (set by controller)
        self.control_callback = None

        # Setup subscriptions
        self._setup_subscriptions()

    def _setup_subscriptions(self):
        """Setup all ROS2 subscriptions"""
        self.node.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
                                     self._lidar_callback, 10)
        self.node.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix',
                                     self._gps_callback, 10)
        self.node.create_subscription(Imu, '/wamv/sensors/imu/imu/data',
                                     self._imu_callback, 10)
        self.node.create_subscription(Point, '/vrx/waypoint',
                                     self._waypoint_callback, 10)

    def set_control_callback(self, callback):
        """Set callback to be called when LiDAR data is updated"""
        self.control_callback = callback

    def _angle_to_index(self, angle_deg):
        """Convert angle in degrees to LiDAR array index"""
        angle_deg = ((angle_deg + 180) % 360) - 180
        if -self.LIDAR_ANGLE_RANGE <= angle_deg <= self.LIDAR_ANGLE_RANGE:
            return np.clip(int(angle_deg + self.LIDAR_ANGLE_RANGE), 0, self.LIDAR_SIZE - 1)
        return None

    def get_lidar_distance_at_angle(self, angle_deg):
        """Get LiDAR distance at specific angle"""
        idx = self._angle_to_index(angle_deg)
        return self.lidar_distances[idx] if idx is not None else self.MAX_LIDAR_DISTANCE

    @property
    def current_waypoint(self):
        """Get current target waypoint"""
        if self.current_target_index < len(self.waypoints):
            return np.array(self.waypoints[self.current_target_index], dtype=np.float32)
        return None

    def get_waypoint_positions(self):
        """Get current, previous, and next waypoint positions"""
        zeros = np.zeros(2, dtype=np.float32)
        if not self.waypoints:
            return zeros, zeros, zeros

        current = self.current_waypoint if self.current_waypoint is not None else zeros
        previous = (np.array(self.waypoints[self.current_target_index - 1], dtype=np.float32)
                   if self.current_target_index > 0 else zeros)
        next_wp = (np.array(self.waypoints[self.current_target_index + 1], dtype=np.float32)
                  if self.current_target_index + 1 < len(self.waypoints) else current.copy())

        return current, previous, next_wp

    # ============ Sensor Callbacks ============

    def _waypoint_callback(self, msg):
        """Waypoint callback"""
        self.waypoints.append([msg.y, msg.x])
        self.current_target_index = len(self.waypoints) - 1
        self.waypoint_reached = False

    def _gps_callback(self, msg):
        """GPS callback"""
        gps_data = self.sensor_manager.process_gps_data(msg)
        if gps_data is not None:
            self.agent_position = np.array([gps_data['utm_y'], gps_data['utm_x']], dtype=np.float32)

    def _imu_callback(self, msg):
        """IMU callback"""
        imu_data = self.sensor_manager.process_imu_data(msg)
        self.agent_heading = imu_data['yaw_degrees'] % 360.0

        current_angular_velocity = np.array([msg.angular_velocity.x,
                                             msg.angular_velocity.y,
                                             msg.angular_velocity.z])
        self.previous_angular_velocity = current_angular_velocity
        self.last_angular_velocity_update_time = time.time()
        self.angular_velocity_y = np.clip(current_angular_velocity[2] *
                                          self.angular_velocity_y_scale, -180, 180)

    def _lidar_callback(self, msg):
        """LiDAR callback and trigger control"""
        ranges = np.array(msg.ranges, dtype=np.float32)
        raw_ranges = np.full(self.LIDAR_SIZE, self.MAX_LIDAR_DISTANCE, dtype=np.float32)

        for i, distance in enumerate(ranges):
            angle_deg = np.degrees(msg.angle_min + i * msg.angle_increment)
            idx = self._angle_to_index(angle_deg)
            if idx is not None:
                if np.isinf(distance) or np.isnan(distance) or distance >= self.MAX_LIDAR_DISTANCE:
                    raw_ranges[idx] = self.MAX_LIDAR_DISTANCE
                else:
                    raw_ranges[idx] = distance / self.lidar_scale_factor

        self.lidar_distances = raw_ranges

        # Trigger control callback if set
        if self.control_callback:
            self.control_callback()
