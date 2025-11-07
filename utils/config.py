#!/usr/bin/env python3
"""
VRX 시스템 설정 상수
- 모든 시스템 파라미터를 중앙 관리
- ROS 토픽명, PID 게인, 웨이포인트, 미션 파라미터, 파일 경로 등
"""

import os
from pathlib import Path


class Constants:
    """시스템 전역 상수"""

    # ============================================================================
    # 프로젝트 경로
    # ============================================================================
    class Paths:
        """파일 및 디렉토리 경로 관리"""

        # 프로젝트 루트 디렉토리 (config.py 위치 기준)
        PROJECT_ROOT = Path(__file__).parent.parent.absolute()

        # NanoOWL 경로
        NANOOWL_DIR = Path('/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/nanoowl')

        # 모델 디렉토리
        MODELS_DIR = PROJECT_ROOT / 'models' / 'correct_IMU' / 'gpu'
        # MODELS_DIR = PROJECT_ROOT / 'models'

        # ONNX 모델 파일
        # ONNX_MODEL = MODELS_DIR / 'Once_observation.onnx'
        ONNX_MODEL = MODELS_DIR / 'Ray.onnx'
        ONNX_MODEL_FALLBACK_1 = MODELS_DIR / 'Ray-9558758.onnx'
        ONNX_MODEL_FALLBACK_2 = MODELS_DIR / 'Ray-23999963.onnx'
        

        @classmethod
        def get_onnx_model_path(cls) -> str:
            """
            사용 가능한 ONNX 모델 경로 반환

            Returns:
                str: 존재하는 첫 번째 ONNX 모델의 절대 경로

            Raises:
                FileNotFoundError: 사용 가능한 모델이 없을 때
            """
            for model_path in [cls.ONNX_MODEL, cls.ONNX_MODEL_FALLBACK_1, cls.ONNX_MODEL_FALLBACK_2]:
                if model_path.exists():
                    return str(model_path)
            raise FileNotFoundError(f"ONNX 모델을 찾을 수 없습니다: {cls.MODELS_DIR}")

    # ============================================================================
    # ROS2 통신 설정
    # ============================================================================
    class QueueSizes:
        """ROS2 퍼블리셔/서브스크라이버 큐 크기"""
        DEFAULT = 10
        SENSOR = 10
        CONTROL = 10
        STATUS = 10

    # ============================================================================
    # 타이머 주기
    # ============================================================================
    MAIN_LOOP_HZ = 100  # 100Hz
    MAIN_LOOP_PERIOD = 1.0 / MAIN_LOOP_HZ

    # ============================================================================
    # LiDAR 설정
    # ============================================================================
    LIDAR_ARRAY_SIZE = 201
    MAX_LIDAR_DISTANCE = 100.0
    LIDAR_ANGLE_RANGE = (-100, 100)  # degrees
    LIDAR_SCALE_FACTOR = 1.2  # LiDAR 거리값 스케일 조정 (1.0 = 변환 없음)

    # ============================================================================
    # 센서 데이터
    # ============================================================================
    ANGULAR_VELOCITY_LIMIT = (-180, 180)

    # ============================================================================
    # 웨이포인트 설정
    # ============================================================================
    DEFAULT_WAYPOINT_RADIUS = 20.0

    # 미리 정의된 웨이포인트 (x, y, mission_type, radius, params)
    # mission_type: 'PASS_BETWEEN_BUOYS', 'CIRCLE_BUOY', 'WAYPOINT_FOLLOW', 'OBSTACLE_AVOID'
    PREDEFINED_WAYPOINTS = [
        (150, 9, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {}),
        (160, 0, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        (140, 42, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {'rotation_direction': 2, 'circle_radius': 15.0}),
        (80, 42, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        (0, 0, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {})
    ]

    # ============================================================================
    # ONNX 모델 설정
    # ============================================================================
    # ONNX_MODEL_PATH는 이제 Paths.get_onnx_model_path()를 사용하세요
    @property
    def ONNX_MODEL_PATH(self) -> str:
        """하위 호환성을 위한 프로퍼티 (deprecated)"""
        return self.Paths.get_onnx_model_path()

    # Observation 크기 및 Temporal Stacking 설정
    # ───────────────────────────────────────────────────────────────────────────
    # OBSERVATION_SIZE: 단일 타임스텝의 observation 크기
    #   - LiDAR distances: 201개 (-100° ~ +100°)
    #   - Agent heading: 1개 (-180° ~ 180°)
    #   - Angular velocity Y: 1개 (deg/s)
    #   - Agent position: 2개 [North, East]
    #   - Current waypoint: 2개 [North, East]
    #   - Previous waypoint: 2개 [North, East]
    #   - Next waypoint: 2개 [North, East]
    #   - Previous moment input: 1개 (이전 angular velocity 명령)
    #   - Previous force input: 1개 (이전 linear velocity 명령)
    #   Total: 201 + 1 + 1 + 2 + 2 + 2 + 2 + 1 + 1 = 213
    OBSERVATION_SIZE = 213

    # STACK_COUNT: 모델에 입력될 temporal stacking 횟수
    #   - 1: 현재 프레임만 사용 (input size = 213)
    #   - 2: 이전 + 현재 프레임 사용 (input size = 426)
    #   - 3: 2 프레임 전 + 이전 + 현재 사용 (input size = 639)
    #   새로운 강화학습 모델을 사용할 때 이 값만 변경하면 됩니다.
    STACK_COUNT = 2

    # ONNX_INPUT_SIZE: 자동 계산됨 (OBSERVATION_SIZE * STACK_COUNT)
    ONNX_INPUT_SIZE = OBSERVATION_SIZE * STACK_COUNT  # 기본값: 213 * 2 = 426

    ONNX_V_SCALE = 1.0
    ONNX_W_SCALE = -1.0
    ONNX_LINEAR_VELOCITY_RANGE = (0.2, 1.0)
    ONNX_ANGULAR_VELOCITY_RANGE = (-1.0, 1.0)

    # ============================================================================
    # 장애물 회피 설정
    # ============================================================================
    BOAT_WIDTH = 3.5
    BOAT_HEIGHT = 50.0
    LOS_DELTA = 10.0
    LOS_LOOKAHEAD_MIN = 30.0
    LOS_LOOKAHEAD_MAX = 80.0
    FILTER_ALPHA = 0.5

    # ============================================================================
    # 미션 설정
    # ============================================================================
    DEFAULT_THRUST_SCALE = 3500.0  # 1000 → 2000 (장애물 회피 성능 향상)
    MAX_COAST_FRAMES = 10
    TRACKER_FPS = 20.0

    # ============================================================================
    # 미션별 PID 게인 설정
    # ============================================================================
    # PassBetweenBuoys 미션 파라미터
    PASS_BETWEEN_PID_KP = 0.0025
    PASS_BETWEEN_PID_KI = 0.0001
    PASS_BETWEEN_PID_KD = 0.001
    PASS_BETWEEN_STEERING_GAIN = 0.0005  # 비례 제어 게인
    PASS_BETWEEN_FORWARD_SPEED = 0.5  # 전진 속도
    PASS_BETWEEN_MAX_STEERING = 0.3  # 최대 조향 값
    PASS_BETWEEN_FALLBACK_SPEED = 0.2  # 부표 미탐지 시 속도
    PASS_BETWEEN_MAX_DEPTH_DIFF = 5.0  # 부표 간 최대 깊이 차이 (미터)

    # CircleBuoy 미션 PID 게인
    CIRCLE_PID_KP = 0.8
    CIRCLE_PID_KI = 0.001
    CIRCLE_PID_KD = 0.4

    # CircleBuoy 미션 속도 파라미터
    CIRCLE_BASE_SPEED = 150.0
    CIRCLE_MIN_SPEED = 50.0
    CIRCLE_MAX_TURN_THRUST = 150.0

    # CircleBuoy 미션 이미지 크기
    CIRCLE_IMAGE_WIDTH = 1280
    CIRCLE_IMAGE_HEIGHT = 720

    # CircleBuoy 미션 target_x 결정식 파라미터 (시계방향)
    CIRCLE_TX_BASE_X = 1240.0
    CIRCLE_TX_SLOPE = 700.0
    CIRCLE_TX_MIN_X = 800.0
    CIRCLE_TX_MAX_X = 1200.0

    # CircleBuoy 미션 반시계방향 파라미터
    CIRCLE_CCW_SLOPE = 200.0  # Counter-clockwise slope
    CIRCLE_CCW_MIN_X = 140.0
    CIRCLE_CCW_MAX_X = 640.0

    # CircleBuoy 완료 기준
    CIRCLE_COMPLETION_ROTATION = 350.0  # 350도 회전 시 완료
    CIRCLE_COMPLETION_SPEED = 0.3  # 완료 후 전진 속도

    # WaypointFollow 미션 파라미터
    WAYPOINT_STEERING_GAIN = 0.01
    WAYPOINT_FORWARD_SPEED = 0.5
    WAYPOINT_MAX_STEERING = 0.5
    WAYPOINT_MIN_DISTANCE = 1.0  # 목표 거리 최소값 (미터)

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
