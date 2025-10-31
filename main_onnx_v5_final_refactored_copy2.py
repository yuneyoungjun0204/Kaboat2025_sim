#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from std_msgs.msg import Float64, Float64MultiArray, String
from utils.parameter_onnx import ONNXSensorManager
from utils.avoid_control import AvoidanceController


class VRXONNXControllerV5Refactored(Node):
    # Constants
    WAYPOINT_THRESHOLD = 15.0
    THRUST_SCALE = 2000
    ONNX_INPUT_SIZE = 426
    TIMER_PERIOD = 0.01  # 100Hz

    # Publisher configurations: (topic, msg_type)
    PUBLISHERS = {
        'left_thrust': ('/wamv/thrusters/left/thrust', Float64),
        'right_thrust': ('/wamv/thrusters/right/thrust', Float64),
        'model_input': ('/vrx/model_input', Float64MultiArray),
        'lidar': ('/vrx/lidar_data', Float64MultiArray),
        'heading': ('/vrx/agent_heading', Float64),
        'angular_vel': ('/vrx/angular_velocity', Float64),
        'position': ('/vrx/agent_position', Float64MultiArray),
        'current_waypoint': ('/vrx/current_waypoint', Float64MultiArray),
        'previous_waypoint': ('/vrx/previous_waypoint', Float64MultiArray),
        'next_waypoint': ('/vrx/next_waypoint', Float64MultiArray),
        'previous_moment': ('/vrx/previous_moment', Float64),
        'previous_force': ('/vrx/previous_force', Float64),
        'control_output': ('/vrx/control_output', Float64MultiArray),
        'control_mode': ('/vrx/control_mode', String),
        'obstacle_check_area': ('/vrx/obstacle_check_area', Float64MultiArray),
        'los_target': ('/vrx/los_target', Float64MultiArray),
    }

    def __init__(self):
        super().__init__('vrx_onnx_controller_v5_refactored')

        # ONNX model
        model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray.onnx'
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Sensor manager
        self.sensors = ONNXSensorManager(self)
        self.sensors.set_control_callback(self.control_vrx)

        # Publishers
        self.pubs = {name: self.create_publisher(msg_type, topic, 10)
                     for name, (topic, msg_type) in self.PUBLISHERS.items()}

        # Avoidance controller
        self.avoidance_controller = AvoidanceController(
            boat_width=1.32, boat_height=50.0, max_lidar_distance=100.0,
            los_delta=10.0, los_lookahead_min=30.0, los_lookahead_max=80.0, filter_alpha=0.5
        )

        # Control state
        self.left_thrust = self.right_thrust = 0.0
        self.use_direct_control = False
        self.previous_moment_input = self.previous_force_input = 0.0
        self.v_scale, self.w_scale = 1.0, -1.0

        # Timer for thruster commands
        self.create_timer(self.TIMER_PERIOD, self._publish_thrusts)

    def _pub(self, name, data, is_string=False):
        """Unified publish method"""
        if is_string:
            msg = String()
            msg.data = data
        elif isinstance(data, (int, float)):
            msg = Float64()
            msg.data = float(data)
        else:
            msg = Float64MultiArray()
            msg.data = np.asarray(data, dtype=float).flatten().tolist()
        self.pubs[name].publish(msg)

    @staticmethod
    def _sanitize(value, default=0.0):
        """Replace inf/nan with default"""
        return default if np.isinf(value) or np.isnan(value) else value

    def control_vrx(self):
        """Main control logic"""
        target = self.sensors.current_waypoint
        if target is None:
            self.left_thrust = self.right_thrust = 0.0
            return

        # Check waypoint reached
        distance = np.linalg.norm(self.sensors.agent_position - target)
        if distance < self.WAYPOINT_THRESHOLD:
            if not self.sensors.waypoint_reached:
                self.sensors.waypoint_reached = True
                self.sensors.current_target_index += 1
                if self.sensors.current_target_index < len(self.sensors.waypoints):
                    self.sensors.waypoint_reached = False
            self.left_thrust = self.right_thrust = 0.0
            return

        # Get LOS target and check obstacles
        los_target = self.avoidance_controller.get_los_target(
            self.sensors.agent_position, self.sensors.waypoints, self.sensors.current_target_index
        )

        self.use_direct_control, linear_vel, angular_vel, check_area = \
            self.avoidance_controller.check_obstacles_and_get_control(
                self.sensors.agent_position, los_target, self.sensors.agent_heading,
                self.sensors.lidar_distances, self.sensors.get_lidar_distance_at_angle,
                self.get_onnx_control
            )

        # Apply filters and update state
        linear_vel, angular_vel = self.avoidance_controller.apply_filters(linear_vel, angular_vel)
        self.previous_moment_input, self.previous_force_input = angular_vel, linear_vel

        # Calculate and filter thrusts
        raw_left, raw_right = self._calc_thrusts(linear_vel, angular_vel)
        self.left_thrust, self.right_thrust = self.avoidance_controller.apply_thrust_filters(
            raw_left, raw_right
        )

        # Publish visualization
        self._pub_viz(check_area, los_target, linear_vel, angular_vel)

    def get_onnx_control(self):
        """ONNX model inference"""
        current_wp, previous_wp, next_wp = self.sensors.get_waypoint_positions()

        # Build observation array
        obs = np.zeros(self.ONNX_INPUT_SIZE // 2, dtype=np.float32)
        obs[:201] = self.sensors.lidar_distances
        obs[201] = self._sanitize(self.sensors.agent_heading)
        obs[202] = self._sanitize(self.sensors.angular_velocity_y)

        # Pack positions and waypoints
        idx = 203
        for arr in [self.sensors.agent_position, current_wp, previous_wp, next_wp]:
            obs[idx:idx+2] = [self._sanitize(arr[0]), self._sanitize(arr[1])]
            idx += 2

        obs[-2:] = [self.previous_moment_input, self.previous_force_input]

        # Stack and infer
        stacked_input = np.tile(obs, 2).reshape(1, self.ONNX_INPUT_SIZE)
        self._pub_model_inputs(stacked_input, current_wp, previous_wp, next_wp)

        outputs = self.session.run(None, {self.input_name: stacked_input})

        if len(outputs) > 4 and outputs[4] is not None:
            linear_vel = np.clip(outputs[4][0][1] * self.v_scale, 0.02, 1.0)
            angular_vel = np.clip(outputs[4][0][0] * self.w_scale, -1.0, 1.0)
            # Ensure differential drive constraints
            angular_vel = np.clip(angular_vel,
                                 max(-1.0, -linear_vel - 1.0, linear_vel - 1.0),
                                 min(1.0, 1.0 - linear_vel, linear_vel + 1.0))
            return linear_vel, angular_vel
        return 0.0, 0.0

    def _calc_thrusts(self, linear_vel, angular_vel):
        """Calculate thruster commands"""
        fwd = linear_vel * self.THRUST_SCALE
        turn = angular_vel * self.THRUST_SCALE
        return (np.clip(fwd + turn, -self.THRUST_SCALE, self.THRUST_SCALE),
                np.clip(fwd - turn, -self.THRUST_SCALE, self.THRUST_SCALE))

    def _pub_viz(self, check_area, los_target, linear_vel, angular_vel):
        """Publish visualization data"""
        self._pub('obstacle_check_area', check_area)
        self._pub('los_target', [self.sensors.agent_position[1] + los_target[1],
                                 self.sensors.agent_position[0] + los_target[0]])
        self._pub('control_output', [linear_vel, angular_vel])
        self._pub('control_mode', "DIRECT_CONTROL" if self.use_direct_control else "ONNX_MODEL",
                 is_string=True)

    def _pub_model_inputs(self, stacked_input, current_wp, previous_wp, next_wp):
        """Publish model inputs"""
        self._pub('model_input', stacked_input)
        self._pub('lidar', self.sensors.lidar_distances)
        self._pub('heading', self.sensors.agent_heading)
        self._pub('angular_vel', -self.sensors.angular_velocity_y)
        self._pub('position', self.sensors.agent_position)
        self._pub('current_waypoint', current_wp)
        self._pub('previous_waypoint', previous_wp)
        self._pub('next_waypoint', next_wp)
        self._pub('previous_moment', self.previous_moment_input)
        self._pub('previous_force', self.previous_force_input)

    def _publish_thrusts(self):
        """Timer callback - publish thruster commands"""
        self._pub('left_thrust', self.left_thrust)
        self._pub('right_thrust', self.right_thrust)

    def destroy_node(self):
        """Stop thrusters on shutdown"""
        self._pub('left_thrust', 0.0)
        self._pub('right_thrust', 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(VRXONNXControllerV5Refactored())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
