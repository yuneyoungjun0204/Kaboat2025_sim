#!/usr/bin/env python3
"""
VRX 시스템 설정 상수
- 모든 시스템 파라미터를 중앙 관리
- ROS 토픽명, PID 게인, 웨이포인트, 미션 파라미터 등
"""


class Constants:
    """시스템 전역 상수"""

    # ============================================================================
    # 타이머 주기
    # ============================================================================
    MAIN_LOOP_HZ = 100  # 20Hz
    MAIN_LOOP_PERIOD = 1.0 / MAIN_LOOP_HZ

    # ============================================================================
    # LiDAR 설정
    # ============================================================================
    LIDAR_ARRAY_SIZE = 201
    MAX_LIDAR_DISTANCE = 100.0
    LIDAR_ANGLE_RANGE = (-100, 100)  # degrees

    # ============================================================================
    # 센서 데이터
    # ============================================================================
    ANGULAR_VELOCITY_LIMIT = (-180, 180)

    # ============================================================================
    # 웨이포인트 설정
    # ============================================================================
    DEFAULT_WAYPOINT_RADIUS = 50.0

    # 미리 정의된 웨이포인트 (x, y, mission_type, radius, params)
    # mission_type: 'PASS_BETWEEN_BUOYS', 'CIRCLE_BUOY', 'WAYPOINT_FOLLOW', 'OBSTACLE_AVOID'
    PREDEFINED_WAYPOINTS = [
        (150, 0, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        (140, 42, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {'rotation_direction': 2, 'circle_radius': 15.0}),
        (0, 165, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {}),
        (0, 0, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {})
    ]

    # ============================================================================
    # ONNX 모델 설정
    # ============================================================================
    ONNX_MODEL_PATH = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray-9558758.onnx'
    # ONNX_MODEL_PATH = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray-23999963.onnx'
    ONNX_MODEL_PATH = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU//gpu/Ray.onnx'
    ONNX_INPUT_SIZE = 426
    ONNX_V_SCALE = 1.0
    ONNX_W_SCALE = -1.0
    ONNX_LINEAR_VELOCITY_RANGE = (0.12, 1.0)
    ONNX_ANGULAR_VELOCITY_RANGE = (-1.0, 1.0)

    # ============================================================================
    # 장애물 회피 설정
    # ============================================================================
    BOAT_WIDTH = 2.0
    BOAT_HEIGHT = 50.0
    LOS_DELTA = 10.0
    LOS_LOOKAHEAD_MIN = 30.0
    LOS_LOOKAHEAD_MAX = 80.0
    FILTER_ALPHA = 0.5

    # ============================================================================
    # 미션 설정
    # ============================================================================
    DEFAULT_THRUST_SCALE = 2000.0  # 1000 → 2000 (장애물 회피 성능 향상)
    MAX_COAST_FRAMES = 10
    TRACKER_FPS = 20.0

    # ============================================================================
    # 미션별 PID 게인 설정
    # ============================================================================
    # PassBetweenBuoys 미션
    PASS_BETWEEN_STEERING_GAIN = 0.003
    PASS_BETWEEN_FORWARD_SPEED = 0.5
    PASS_BETWEEN_MAX_STEERING = 0.3
    PASS_BETWEEN_MAX_DEPTH_DIFF = 5.0  # 부표 간 최대 깊이 차이 (미터)

    # CircleBuoy 미션 PID 게인
    CIRCLE_PID_KP = 0.8
    CIRCLE_PID_KI = 0.001
    CIRCLE_PID_KD = 0.4

    # CircleBuoy 미션 속도 파라미터
    CIRCLE_BASE_SPEED = 150.0
    CIRCLE_MIN_SPEED = 50.0
    CIRCLE_MAX_TURN_THRUST = 150.0

    # CircleBuoy 미션 target_x 결정식 파라미터
    CIRCLE_TX_BASE_X = 1240.0
    CIRCLE_TX_SLOPE = 700.0
    CIRCLE_TX_MIN_X = 800.0
    CIRCLE_TX_MAX_X = 1200.0

    # WaypointFollow 미션
    WAYPOINT_STEERING_GAIN = 0.01
    WAYPOINT_FORWARD_SPEED = 0.5
    WAYPOINT_MAX_STEERING = 0.5

    # ============================================================================
    # ROS2 토픽명 설정
    # ============================================================================
    class Topics:
        """ROS2 토픽명 관리"""

        # 센서 입력 토픽
        CAMERA_IMAGE = '/wamv/sensors/cameras/front_left_camera_sensor/image_raw'
        LIDAR_SCAN = '/wamv/sensors/lidars/lidar_wamv_sensor/scan'
        GPS_FIX = '/wamv/sensors/gps/gps/fix'
        IMU_DATA = '/wamv/sensors/imu/imu/data'

        # 웨이포인트 토픽
        WAYPOINT = '/vrx/waypoint'

        # 제어 출력 토픽
        LEFT_THRUST = '/wamv/thrusters/left/thrust'
        RIGHT_THRUST = '/wamv/thrusters/right/thrust'

        # 시스템 상태 토픽
        MISSION_STATUS = '/vrx/mission_status'
        DETECTIONS = '/vrx/detections'
        VISUALIZATION = '/vrx/visualization'
        CONTROL_OUTPUT = '/vrx/control_output'
        CONTROL_MODE = '/vrx/control_mode'
        OBSTACLE_CHECK_AREA = '/vrx/obstacle_check_area'
        LOS_TARGET = '/vrx/los_target'

        # trajectory_viz용 추가 토픽
        CURRENT_MODE = '/vrx/current_mode'
        GOAL_CHECK_AREAS = '/vrx/goal_check_areas'

    # ============================================================================
    # 시각화 설정 (trajectory_viz.py용)
    # ============================================================================
    class Visualization:
        """시각화 파라미터"""

        # 히스토리 길이
        POSITION_HISTORY_MAXLEN = 2000
        HEADING_HISTORY_MAXLEN = 2000

        # 축 범위 설정
        AXIS_MARGIN = 200.0
        AXIS_MARGIN_X = 60.0
        AXIS_MARGIN_Y = 180.0

        # 배 크기 및 안전 여유
        BOAT_WIDTH = 5.0
        SAFETY_MARGIN = 2.0

        # 헤딩 화살표 길이
        HEADING_ARROW_LENGTH = 10.0
        TARGET_HEADING_ARROW_LENGTH = 25.0
        LIDAR_TARGET_HEADING_ARROW_LENGTH = 30.0

        # LiDAR 최대 범위
        LIDAR_MAX_RANGE = 50.0

        # Figure 크기
        FIGURE_SIZE = (18, 10)

        # 업데이트 주기
        UPDATE_RATE = 0.1  # 10Hz

    # ============================================================================
    # 강제 미션 모드 설정 (트랙바 값)
    # ============================================================================
    class ForceMissionMode:
        """트랙바로 설정하는 강제 미션 모드"""
        NORMAL = 0  # 일반 모드 (웨이포인트 기반)
        OBSTACLE_AVOID = 1  # 강제 장애물 회피 모드
        PASS_BETWEEN_BUOYS = 2  # 강제 부표 사이 지나기 미션
        CIRCLE_BUOY = 3  # 강제 부표 한바퀴 돌기 미션
