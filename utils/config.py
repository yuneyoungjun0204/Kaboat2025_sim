#!/usr/bin/env python3
"""
VRX 시스템 설정 상수
"""


class Constants:
    """시스템 전역 상수"""
    # 타이머 주기
    MAIN_LOOP_HZ = 20  # 20Hz
    MAIN_LOOP_PERIOD = 1.0 / MAIN_LOOP_HZ

    # LiDAR 설정
    LIDAR_ARRAY_SIZE = 201
    MAX_LIDAR_DISTANCE = 100.0
    LIDAR_ANGLE_RANGE = (-100, 100)  # degrees

    # 센서 데이터
    ANGULAR_VELOCITY_LIMIT = (-180, 180)

    # 웨이포인트
    DEFAULT_WAYPOINT_RADIUS = 20.0

    # ONNX 모델
    ONNX_MODEL_PATH = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray-23999963.onnx'
    ONNX_MODEL_PATH = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray-9558758.onnx'
    ONNX_INPUT_SIZE = 426
    ONNX_V_SCALE = 1.0
    ONNX_W_SCALE = -1.0
    ONNX_LINEAR_VELOCITY_RANGE = (0.12, 1.0)
    ONNX_ANGULAR_VELOCITY_RANGE = (-1.0, 1.0)

    # 장애물 회피 설정
    BOAT_WIDTH = 5.0
    BOAT_HEIGHT = 50.0
    LOS_DELTA = 10.0
    LOS_LOOKAHEAD_MIN = 30.0
    LOS_LOOKAHEAD_MAX = 80.0
    FILTER_ALPHA = 0.5

    # 미션 설정
    DEFAULT_THRUST_SCALE = 1000.0
    MAX_COAST_FRAMES = 10
    TRACKER_FPS = 20.0
