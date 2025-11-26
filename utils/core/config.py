#!/usr/bin/env python3
"""
VRX 시스템 설정 상수
- 모든 시스템 파라미터를 중앙 관리
- ROS 토픽명, PID 게인, 웨이포인트, 미션 파라미터, 파일 경로 등

Version: 3.0 (Refactored 2025-01-18)
- 환경 전환 시스템 추가 (시뮬레이터/실제 PX4)
- 모든 하드코딩 파라미터 중앙화
- Quick Start 가이드 추가

================================================================================
🚀 QUICK START GUIDE (빠른 시작 가이드)
================================================================================

1. 환경 전환 (시뮬레이터 ↔ 실제 환경):
   - 아래 CURRENT_ENVIRONMENT 변수만 수정하세요
   - Environment.SIMULATOR: Gazebo 시뮬레이터
   - Environment.REAL_PX4: 실제 Pixhawk 하드웨어

2. 웨이포인트 설정:
   - PREDEFINED_WAYPOINTS 리스트 수정
   - WAYPOINT_MODE: 0=로컬좌표(미터), 1=GPS좌표(위경도)

3. 미션별 PID 튜닝:
   - 각 미션 섹션에서 PID_KP, PID_KI, PID_KD 수정
   - 예: CIRCLE_PID_KP, DOCK_SWAY_GAIN 등

4. 센서 설정:
   - LiDAR: LIDAR_ARRAY_SIZE, MAX_LIDAR_DISTANCE
   - 카메라: Topics.CAMERA_IMAGE

================================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any


# ============================================================================
# 🔧 환경 설정 (ENVIRONMENT CONFIGURATION)
# ============================================================================
class Environment:
    """실행 환경 정의"""
    SIMULATOR = "simulator"    # Gazebo 시뮬레이터
    REAL_PX4 = "real_px4"      # 실제 Pixhawk 하드웨어


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  🔧 여기만 수정하세요! 시뮬레이터 ↔ 실제 환경 전환                      │
# └─────────────────────────────────────────────────────────────────────────┘
CURRENT_ENVIRONMENT = Environment.REAL_PX4


class Constants:
    """시스템 전역 상수"""

    # ============================================================================
    # 프로젝트 경로
    # ============================================================================
    class Paths:
        """파일 및 디렉토리 경로 관리"""

        # 프로젝트 루트 디렉토리 (config.py 위치 기준: utils/core/config.py -> utils -> 프로젝트 루트)
        PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

        # NanoOWL 경로
        NANOOWL_DIR = Path('/home/yuneyoungjun/vrx_ws/src/vrx/vrx_env/nanoowl')

        # 모델 디렉토리
        MODELS_DIR = PROJECT_ROOT / 'models' / 'correct_IMU' / 'gpu'
        # MODELS_DIR = PROJECT_ROOT / 'models' / '207'
        # MODELS_DIR = PROJECT_ROOT / 'models'

        # ONNX 모델 파일
        # ONNX_MODEL = MODELS_DIR / 'Once_observation.onnx'
        ONNX_MODEL = MODELS_DIR / 'Ray.onnx'
        # ONNX_MODEL = MODELS_DIR / 'Ray-9558758.onnx'
        # ONNX_MODEL_FALLBACK_2 = MODELS_DIR / 'Ray-24499969.onnx'
        # ONNX_MODEL = MODELS_DIR / 'Ray-24499969.onnx'
        

        @classmethod
        def get_onnx_model_path(cls) -> str:
            """
            사용 가능한 ONNX 모델 경로 반환

            Returns:
                str: 존재하는 첫 번째 ONNX 모델의 절대 경로

            Raises:
                FileNotFoundError: 사용 가능한 모델이 없을 때
            """
            for model_path in [cls.ONNX_MODEL]:
                if model_path.exists():
                    return str(model_path)
            raise FileNotFoundError(f"ONNX 모델을 찾을 수 없습니다: {cls.MODELS_DIR}")

    # ============================================================================
    # PX4 통합 설정
    # ============================================================================
    class PX4:
        """PX4/Pixhawk 연동 파라미터"""

        # PX4 모드 활성화 (CURRENT_ENVIRONMENT에 따라 자동 설정)
        # REAL_PX4 환경이면 자동으로 True로 설정됨
        ENABLED = (CURRENT_ENVIRONMENT == Environment.REAL_PX4)

        # 제어 파라미터
        MAX_VELOCITY = 2.0          # 최대 전진 속도 (m/s)
        MIN_VELOCITY = 0.2
        MAX_YAW_RATE = 1.0          # 최대 yaw rate (rad/s)
        CONTROL_RATE_HZ = 50.0      # 제어 주기 (Hz)

        # 좌표계 원점 (LLA to NED 변환용)
        # LAT_ORIGIN = -33.72259952421798   # Sydney Regatta Centre
        # LON_ORIGIN = 150.67390369752246
        ALT_ORIGIN = 0.0

        # 속도 스케일링
        VELOCITY_SCALE = 1.0        # desired_speed → m/s 변환 계수
        YAW_RATE_SCALE = 1.0        # desired_moment → rad/s 변환 계수

        # Offboard 제어 설정
        OFFBOARD_SETPOINT_COUNT = 10  # Offboard 모드 전환 전 setpoint 개수

        # 위치 제어 전환 임계값
        POSITION_CONTROL_DISTANCE = 5.0  # 이 거리 이하에서 위치 제어 (m)

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
    LIDAR_SCALE_FACTOR = 5.0  # LiDAR 거리값 스케일 조정 (1.0 = 변환 없음)
    LIDAR_OBSTACLE_COUNT_THRESHOLD = 2  # ONNX 모드 전환을 위한 최소 장애물 감지 개수

    # LiDAR 필터링 설정
    LIDAR_FILTER_ENABLED = True  # 필터링 활성화 여부
    LIDAR_MIN_VALID_DISTANCE = 0.1  # 최소 유효 거리 (미터)
    LIDAR_MAX_VALID_DISTANCE = 100.0  # 최대 유효 거리 (미터)
    LIDAR_MEDIAN_FILTER_WINDOW = 1  # Median 필터 윈도우 크기 (홀수, 3-7 권장)
    LIDAR_TEMPORAL_FILTER_ALPHA = 1.0  # 시간적 필터 계수 (0-1, 낮을수록 부드러움)
    LIDAR_FILTER_DEBUG_LOG_INTERVAL = 100  # 필터링 통계 로그 출력 주기 (프레임 수, 0=비활성화)

    # ============================================================================
    # 센서 데이터
    # ============================================================================
    ANGULAR_VELOCITY_LIMIT = (-180, 180)

    # ============================================================================
    # 웨이포인트 설정
    # ============================================================================
    DEFAULT_WAYPOINT_RADIUS = 5.0

    # 웨이포인트 좌표계 모드
    # 0: 로컬 좌표계 (UTM 상대 좌표, 미터 단위)
    # 1: GPS 좌표계 (위도/경도)
    WAYPOINT_MODE = 1

    # GPS 기준점 (MODE=1일 때 사용)
    # - MODE=0: 로컬 좌표만 사용하므로 0,0으로 고정 (모든 상대 좌표 기준점)
    # - MODE=1: 실제 GPS 기준점 사용 (필요 시 아래 값을 수정)
    _GPS_REFERENCE_LAT_DEFAULT = 36.39601179  # Sydney Regatta Centre 기준
    _GPS_REFERENCE_LON_DEFAULT = 127.40155743

    if WAYPOINT_MODE == 0:
        GPS_REFERENCE_LAT = 0.0
        GPS_REFERENCE_LON = 0.0
    else:
        GPS_REFERENCE_LAT = _GPS_REFERENCE_LAT_DEFAULT
        GPS_REFERENCE_LON = _GPS_REFERENCE_LON_DEFAULT

    # 미리 정의된 웨이포인트
    # MODE=0: (x, y, mission_type, radius, params) - x,y는 미터 단위
    # MODE=1: (lat, lon, mission_type, radius, params) - lat,lon은 위도/경도
    # mission_type: 'PASS_BETWEEN_BUOYS', 'CIRCLE_BUOY', 'WAYPOINT_FOLLOW', 'OBSTACLE_AVOID', 'DOCK_MODE', 'ROTATION'
    #
    # 💡 PASS_BETWEEN_BUOYS 미션 동작:
    #    - 부표 2개 탐지: 중점으로 제어
    #    - 부표 미탐지: LOS guidance로 웨이포인트 추종
    #      * 이전 기준점 = 미션 시작 위치 (자동 설정, 첫 WP에서도 동일)
    #      * 목표 = 현재 웨이포인트
    PREDEFINED_WAYPOINTS = [
        # (100, 10, 'DOCK_MODE', DEFAULT_WAYPOINT_RADIUS, {
        #     'dock_index': 1,
        #     'dock_points': [
        #         [[5.0, 0.0], [20.0, 0.0]],  # 도킹 스테이션 1: 도킹 포인트, 보조 포인트
        #         [[5.0, 10.0], [40.0, 10.0]],  # 도킹 스테이션 2: 도킹 포인트, 보조 포인트
        #         [[5.0, 20.0], [60.0, 20.0]]   # 도킹 스테이션 3: 도킹 포인트, 보조 포인트
        #     ]
        # }),
        # MODE=0 (로컬 좌표) 예시:
        # (30, 10, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {'los_delta': 10.0}),
        # (50, 40, 'CIRCLE_BUOY', DEFAULT_WAYPOINT_RADIUS,{}),
        # # DOCK_MODE 예시: dock_index(1-3), dock_points는 상대 좌표로 3쌍 (도킹 포인트, 도킹 보조 포인트)
        # # dock_points 형식: [[dock1_point, dock1_aux], [dock2_point, dock2_aux], [dock3_point, dock3_aux]]
        # # 각 포인트는 [Easting, Northing] 형식의 상대 좌표 (미터)
        # (160, 0, 'CIRCLE_BUOY', DEFAULT_WAYPOINT_RADIUS, {'rotation_direction': 1, 'length': -8.0, 'radius_reach': 5.0, 'max_duration': 80.0}),
        # (15, -15, 'CIRCLE_BUOY', DEFAULT_WAYPOINT_RADIUS, {'length': 14.0, 'angle': 45.0, 'turn_flag': 1, 'radius': 2.0, 'los_delta': 10.0}),
        # (15, -15, 'STOP', DEFAULT_WAYPOINT_RADIUS, {'stop_duration': 15.0}),  # 3초 정지
        # (30, 0, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {'los_delta': 10.0}),
        # (30, 10, 'ROTATION', DEFAULT_WAYPOINT_RADIUS, {'desired_angle': 70.0}),
        # (50, 10, 'DOCK_MODE', DEFAULT_WAYPOINT_RADIUS, {
        #     'dock_index': 2,
        #     'dock_points': [
        #         [[5.0, 0.0], [10.0, 0.0]],  # 도킹 스테이션 1: 도킹 포인트, 보조 포인트
        #         [[5.0, 10.0], [10.0, 10.0]],  # 도킹 스테이션 2: 도킹 포인트, 보조 포인트
        #         [[5.0, 20.0], [10.0, 20.0]]   # 도킹 스테이션 3: 도킹 포인트, 보조 포인트
        #     ],
        #     'los_delta': 15.0  # DOCK_MODE LOS delta
        # }),
        # (55, 55, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        # (50, 0, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        # (0, 0, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {}),

# 36.39603706°, lon=127.40173229
        # STOP 미션 예시: (x, y, 'STOP', DEFAULT_WAYPOINT_RADIUS, {'stop_duration': 5.0})
        # (100, 50, 'STOP', DEFAULT_WAYPOINT_RADIUS, {'stop_duration': 3.0}),  # 3초 정지

        # MODE=1 (GPS 좌표) 예시: (WAYPOINT_MODE를 1로 변경 후 사용)
        # (-33.8575, 151.2160, 'DOCK_MODE', DEFAULT_WAYPOINT_RADIUS,  {'target_shape': 'red_square', 'los_delta': 15.0}),
        # (36.39603892, 127.40173437, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {'los_delta': 3.0}),
        # (36.3960382, 127.40173437, 'STOP', DEFAULT_WAYPOINT_RADIUS, {'stop_duration': 5.0}),
        # (36.3960382, 127.40173437, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {
        #     'previous_waypoint_lat': 36.39603892,
        #     'previous_waypoint_lon': 127.40173437
        # }),
        
        (36.3960382, 127.40173437, 'DOCK_MODE', DEFAULT_WAYPOINT_RADIUS, {
            'dock_control_mode': 'POSITION_CONTROL',  # 'LOS' 또는 'POSITION_CONTROL'
            'dock_index': 1,  # 도킹 스테이션 번호 (1-6)
            'dock_points': [  # 도킹 포인트 리스트 [[[dock_point], [aux_point]], ...]
                # 위경도 형식: [[lat, lon], [lat, lon]] 또는 상대 좌표: [[Easting, Northing], [Easting, Northing]]
                # 위경도는 -90~90 (위도), -180~180 (경도) 범위로 자동 감지
                [[36.3960382, 127.40173437], [36.3960382, 127.40173437]],  # 스테이션 1: [도킹 포인트(위경도), 보조 포인트(위경도)]
                [[36.3960382, 127.40173437], [36.3960382, 127.40173437]],  # 스테이션 2
                [[36.3960382, 127.40173437], [36.3960382, 127.40173437]],  # 스테이션 3
            ],
            'dock_approach_time': 30.0,  # 접근 단계 시간 (초)
            'dock_reverse_time': 5.0,  # 후진 단계 시간 (초)
            'dock_approach_speed': 0.5,  # 접근 속도
            'dock_reverse_speed': 0.3,  # 후진 속도
            'dock_reach_radius': 3.0,  # 도킹 포인트 도달 반경 (미터)
            # Position Control 모드용 파라미터 (control_mode='POSITION_CONTROL'일 때 사용)
            'x_error': 0.0,  # body-frame x 오차 (미터, 전방)
            'y_error': 0.0,  # body-frame y 오차 (미터, 좌측)
            'desired_psi': 10.0,  # 목표 헤딩 (도, 0=North)
            'los_delta': 15.0  # LOS 가이던스 delta (LOS 모드용)
        }),
        # (100, 10, 'ROTATION', DEFAULT_WAYPOINT_RADIUS, {'desired_angle': 70.0}),
        # (36.3960382, 127.40173437, 'CIRCLE_BUOY', DEFAULT_WAYPOINT_RADIUS, {'length': 14.0, 'angle': 45.0, 'turn_flag': 1, 'radius': 3.0, 'los_delta': 10.0}),
        # (-33.8580, 151.2165, 'DOCK_MODE', DEFAULT_WAYPOINT_RADIUS, {'target_shape': 'red_square', 'los_delta': 15.0}),
        (36.3960382, 127.40173437, 'STOP', DEFAULT_WAYPOINT_RADIUS, {'stop_duration': 5.0}),
        


        #####square 33
        # (36.39601179, 127.40155743, 'STOP', DEFAULT_WAYPOINT_RADIUS, {'stop_duration': 15.0}),  # 3초 정지
        # (36.39604027, 127.40173397, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {'los_delta': 10.0}),
        # (36.39604063, 127.40173261, 'CIRCLE_BUOY', DEFAULT_WAYPOINT_RADIUS, {'desired_angle': 70.0}),
        # (36.39615307, 127.40175246, 'DOCK_MODE', DEFAULT_WAYPOINT_RADIUS, {
        #     'dock_index': 2,
        #     'dock_points': [
        #         [[5.0, 0.0], [10.0, 0.0]],  # 도킹 스테이션 1: 도킹 포인트, 보조 포인트
        #         [[5.0, 10.0], [10.0, 10.0]],  # 도킹 스테이션 2: 도킹 포인트, 보조 포인트
        #         [[5.0, 20.0], [10.0, 20.0]]   # 도킹 스테이션 3: 도킹 포인트, 보조 포인트
        #     ],
        #     'los_delta': 15.0  # DOCK_MODE LOS delta
        # }),
        # (55, 55, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        # (50, 0, 'OBSTACLE_AVOID', DEFAULT_WAYPOINT_RADIUS, {}),
        # (0, 0, 'PASS_BETWEEN_BUOYS', DEFAULT_WAYPOINT_RADIUS, {}),
    ]

    # ============================================================================
    # ONNX 모델 설정
    # ============================================================================
    # ONNX 버전 선택: 1 = 기존 v1 (213 obs), 2 = v2 Unity ML-Agent 스타일 (207 obs)
    ONNX_VERSION = 1  # 1 또는 2

    # ONNX_MODEL_PATH는 이제 Paths.get_onnx_model_path()를 사용하세요
    @property
    def ONNX_MODEL_PATH(self) -> str:
        """하위 호환성을 위한 프로퍼티 (deprecated)"""
        return self.Paths.get_onnx_model_path()

    # Observation 크기 및 Temporal Stacking 설정 (v1용)
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
    ONNX_W_SCALE = 1.0
    ONNX_LINEAR_VELOCITY_RANGE = (0.2, 1.0)
    ONNX_ANGULAR_VELOCITY_RANGE = (-1.0, 1.0)

    # ============================================================================
    # 장애물 회피 설정
    # ============================================================================
    BOAT_WIDTH = 8.0
    BOAT_HEIGHT = 25.0
    LOS_DELTA = 35.0
    LOS_LOOKAHEAD_MIN = 20.0
    LOS_LOOKAHEAD_MAX = 40.0
    FILTER_ALPHA = 0.4

    # ============================================================================
    # 미션 설정
    # ============================================================================
    DEFAULT_THRUST_SCALE = 1500.0  # 1000 → 2000 (장애물 회피 성능 향상)
    MAX_COAST_FRAMES = 10
    TRACKER_FPS = 20.0

    # ============================================================================
    # 미션별 PID 게인 설정
    # ============================================================================
    # PassBetweenBuoys 미션 파라미터
    PASS_BETWEEN_PID_KP = 0.0024
    PASS_BETWEEN_PID_KI = 0.0001
    PASS_BETWEEN_PID_KD = 0.001
    PASS_BETWEEN_STEERING_GAIN = 0.0001  # 비례 제어 게인
    PASS_BETWEEN_FORWARD_SPEED = 0.4  # 전진 속도
    PASS_BETWEEN_MAX_STEERING = 0.3  # 최대 조향 값
    PASS_BETWEEN_FALLBACK_SPEED = 0.2  # 부표 미탐지 시 속도
    PASS_BETWEEN_MAX_DEPTH_DIFF = 5.0  # 부표 간 최대 깊이 차이 (미터)

    # CircleBuoy 미션 PID 게인
    CIRCLE_PID_KP = 0.8
    CIRCLE_PID_KI = 0.001
    CIRCLE_PID_KD = 0.4

    # CircleBuoy 미션 속도 파라미터
    CIRCLE_BASE_SPEED = 450.0
    CIRCLE_MIN_SPEED = 50.0
    CIRCLE_MAX_TURN_THRUST = 250.0
    CIRCLE_DEFAULT_ROTATION_DIR = 1
    locked_left_cmd = 0.0
    locked_right_cmd = 0.0

    # CircleBuoy 미션 이미지 크기
    CIRCLE_IMAGE_WIDTH = 1280
    CIRCLE_IMAGE_HEIGHT = 720

    # CircleBuoy 미션 target_x 결정식 파라미터 (시계방향)
    CIRCLE_TX_BASE_X = 1240.0
    CIRCLE_TX_SLOPE = 500.0
    CIRCLE_TX_MIN_X = 820.0
    CIRCLE_TX_MAX_X = 1000.0

    # CircleBuoy 미션 반시계방향 파라미터
    CIRCLE_CCW_SLOPE = 500.0  # Counter-clockwise slope
    CIRCLE_CCW_MIN_X = 310.0
    CIRCLE_CCW_MAX_X = 560.0

    # CircleBuoy 완료 기준
    CIRCLE_COMPLETION_ROTATION = 390.0  # 350도 회전 시 완료
    CIRCLE_COMPLETION_SPEED = 0.3  # 완료 후 전진 속도

    # CircleBuoy 명령 고정 기준
    CIRCLE_LOCK_DISTANCE_THRESHOLD = 0.65  # 부표와의 거리가 이 값 이하이면 명령 고정 (미터)

    # CircleBuoy SWAY 제어 (부드러운 원 그리기)
    CIRCLE_SWAY_STRENGTH = 0.3  # SWAY 힘 강도 (0-1)
    CIRCLE_SWAY_MAX_ANGLE = 30.0  # SWAY 최대 각도 (도)

    # CircleBuoy 1차 저주파 필터 (명령값 튀기 방지)
    CIRCLE_FILTER_ALPHA = 0.1 # 필터 계수 (0-1, 작을수록 부드러움)

    # CircleBuoy 속도 계산 파라미터
    CIRCLE_SPEED_ANGLE_THRESHOLD = 90.0  # 속도 계산 시 각도 임계값 (도)

    # WaypointFollow 미션 파라미터
    WAYPOINT_STEERING_GAIN = 0.003
    WAYPOINT_FORWARD_SPEED = 0.5
    WAYPOINT_MAX_STEERING = 0.5
    WAYPOINT_MIN_DISTANCE = 1.0  # 목표 거리 최소값 (미터)

    # # Dock_mode 미션 파라미터
    # DOCK_SWAY_GAIN = 0.0025  # Sway motion 비례 게인 (픽셀 오차 -> 추력)
    # DOCK_YAW_GAIN = 0.00008  # Yaw 회전 비례 게인
    # DOCK_MAX_SWAY_THRUST = 800.0  # 최대 횡방향 추력
    # DOCK_MAX_YAW_THRUST = 300.0  # 최대 회전 추력
    # DOCK_BASE_SURGE = 0.1  # 기본 전진 속도 (0-1)
    # DOCK_DEPTH_THRESHOLD = 0.5  # Depth 임계값 (가까움, 0-1 스케일)
    # DOCK_APPROACH_TIME = 1.0  # 직진 접근 시간 (초)
    # DOCK_REVERSE_TIME = 10.0  # 후진 시간 (초)
    # DOCK_APPROACH_SPEED = 0.2  # 최종 접근 속도
    # DOCK_REVERSE_SPEED = -0.3  # 후진 속도



    # Dock_mode 미션 파라미터
    DOCK_DEFAULT_THRUST_SCALE = 1000.0  # 도킹 미션 기본 thrust scale
    DOCK_SWAY_GAIN = 5.670915  # Sway motion 비례 게인 (픽셀 오차 -> 추력)
    DOCK_YAW_GAIN = 0.3548  # Yaw 회전 비례 게인
    DOCK_MAX_SWAY_THRUST = 500.0  # 최대 횡방향 추력
    DOCK_MAX_YAW_THRUST = 200.0  # 최대 회전 추력
    DOCK_BASE_SURGE = 0.1  # 기본 전진 속도 (0-1)
    DOCK_DEPTH_THRESHOLD = 0.65  # Depth 임계값 (가까움, 0-1 스케일)
    DOCK_APPROACH_TIME = 1.0  # 직진 접근 시간 (초)
    DOCK_REVERSE_TIME = 20.0  # 후진 시간 (초)
    DOCK_APPROACH_SPEED = 0.2  # 최종 접근 속도
    DOCK_REVERSE_SPEED = -0.6  # 후진 속도
    DOCK_CENTER_TOLERANCE = 600.0  # 이미지 중앙 허용 오차 (픽셀, ± 범위)
    DOCK_SWAY_STRENGTH = 0.5  # SWAY 제어 강도 (0-1)
    DOCK_SWAY_TO_YAW_THRESHOLD = 50.0  # SWAY에서 YAW로 전환하는 픽셀 오차 임계값
    DOCK_FALLBACK_SWAY_GAIN = 0.01  # 목표 미탐지 시 누적 각도 기반 SWAY 게인
    DOCK_FALLBACK_SURGE_VELOCITY = 0.05  # 목표 미탐지 시 전진 속도
    DOCK_ANGLE_FEEDBACK_GAIN = 0.02  # 누적 각도 피드백 게인 (업데이트: 0.005 → 0.02)
    DOCK_SURGE_ERROR_COEFFICIENT = 0.6  # 에러에 따른 surge 감소 계수

    # Dock SWAY 제어 세부 파라미터
    DOCK_ERROR_RATIO_MULTIPLIER = 2.0  # 오차 비율 계산 시 곱하는 값
    DOCK_SWAY_INITIAL_CLIP_VALUE = 1.0  # SWAY force 초기 클리핑 값
    DOCK_SWAY_FINAL_CLIP_VALUE = 0.8  # SWAY force 최종 클리핑 값 (각도 피드백 적용 후)
    DOCK_YAW_CLIP_VALUE = 1.0  # YAW moment 클리핑 값
    DOCK_BODY_FORCE_CLIP_VALUE = 1.0  # Body force 저장 시 클리핑 값

    # Dock SURGE 제어 파라미터
    DOCK_SURGE_REDUCTION_FACTOR = 0.6  # SWAY force에 따른 surge 속도 감소 계수
    DOCK_MIN_SURGE_RATIO = 0.1  # 최소 surge 속도 비율 (base_surge 대비)

    # Dock Thruster Allocation 파라미터
    DOCK_SWAY_MAX_ANGLE = 90.0  # SWAY 최대 각도 (도)
    DOCK_YAW_MAX_ANGLE_DIFF = 15.0  # YAW 각도 차이 (도)
    DOCK_SWAY_FORCE_THRESHOLD = 0.5  # SWAY force 임계값
    DOCK_SURGE_VELOCITY_THRESHOLD = 0.4  # Surge velocity 임계값
    DOCK_SWAY_ONLY_THRUST_COEFF = 0.5  # SWAY 전용 추력 계수
    DOCK_ANGLE_COMPENSATION_BASE = 1.0  # 각도 보상 기본값
    DOCK_ANGLE_COMPENSATION_COEFF = 0.3  # 각도 보상 계수
    DOCK_YAW_THRUST_DIFF_COEFF = 0.3  # YAW 추력 차이 계수
    DOCK_MAX_THRUST_LIMIT_COEFF = 1.5  # 최대 추력 제한 계수

    # Rotation 미션 파라미터
    ROTATION_DEFAULT_TARGET = 0.0  # 기본 목표 각도 (도)
    ROTATION_TOLERANCE = 3.0  # 목표 각도 허용 오차 (도)
    ROTATION_STABLE_FRAMES = 10  # 안정화 필요 프레임 수
    ROTATION_GAIN = 0.003  # 회전 비례 게인
    ROTATION_MAX_THRUST = 0.3  # 최대 회전 추력

    # Stop 미션 파라미터
    STOP_DEFAULT_DURATION = 5.0  # 기본 정지 시간 (초)








    # ============================================================================
    # ROS2 토픽명 설정
    # ============================================================================
    class Topics:
        """ROS2 토픽명 관리"""

        # 센서 입력 토픽
        CAMERA_IMAGE = 'image_raw'
        LIDAR_SCAN = '/scan'
        GPS_FIX = '/wamv/sensors/gps/gps/fix'
        IMU_DATA = '/wamv/sensors/imu/imu/data'

        # 웨이포인트 토픽
        WAYPOINT = '/vrx/waypoint'

        # 제어 출력 토픽
        LEFT_THRUST = '/wamv/thrusters/left/thrust'
        RIGHT_THRUST = '/wamv/thrusters/right/thrust'
        LEFT_POS = '/wamv/thrusters/left/pos'
        RIGHT_POS = '/wamv/thrusters/right/pos'

        # 통합 제어 명령 토픽 (배 독립적)
        DESIRED_SPEED = '/vrx/desired_speed'  # Surge velocity (-1~1)
        DESIRED_MOMENT = '/vrx/desired_moment'  # Yaw moment (-1~1)
        DESIRED_FORCE_Y = '/vrx/desired_force_y'  # Sway force (-1~1)

        # 시스템 상태 토픽
        MISSION_STATUS = '/vrx/mission_status'
        DETECTIONS = '/vrx/detections'
        DETECTION_DEPTHS = '/vrx/detection_depths'  # 탐지된 객체들의 depth 정보
        VISUALIZATION = '/vrx/visualization'
        CONTROL_OUTPUT = '/vrx/control_output'
        CONTROL_MODE = '/vrx/control_mode'
        OBSTACLE_CHECK_AREA = '/vrx/obstacle_check_area'
        LOS_TARGET = '/vrx/los_target'
        TARGET_DEPTH = '/vrx/target_depth'  # Dock_mode 목표 객체 깊이

        # trajectory_viz용 추가 토픽
        CURRENT_MODE = '/vrx/current_mode'
        GOAL_CHECK_AREAS = '/vrx/goal_check_areas'
        CIRCLE_BUOY_WAYPOINTS = '/vrx/circle_buoy_waypoints'  # CIRCLE_BUOY 미션 웨이포인트
        CIRCLE_BUOY_CURRENT_WAYPOINT = '/vrx/circle_buoy_current_waypoint'  # CIRCLE_BUOY 미션 현재 추종 웨이포인트
        ALL_WAYPOINTS = '/vrx/all_waypoints'  # 모든 웨이포인트 정보 (미션 타입 포함)
        GPS_REFERENCE = '/vrx/gps_reference'  # GPS 기준점 정보 [waypoint_mode, ref_lat, ref_lon]
        DOCK_POSITION_ERROR = '/vrx/dock_position_error'  # 도킹 미션 position control 오차 [x_error, y_error, desired_psi]

        # 장애물 회피용 토픽
        AVOID_YAW = '/avoid_yaw'  # ONNX 모드가 아닐 때 0, 1 번갈아 발행

        # ============================================================================
        # PX4 브릿지 토픽 (Pixhawk 연결용)
        # ============================================================================
        # 기존 시스템 → PX4 브릿지 노드 (중간 인터페이스)
        PX4_VELOCITY_YAW_CMD = '/px4_bridge/velocity_yaw_cmd'  # [u_d, yaw_d]
        PX4_POSITION_ERROR = '/px4_bridge/position_error'      # [x_e, y_e]
        PX4_CONTROL_FLAG = '/px4_bridge/control_flag'          # Bool

        # PX4 직접 토픽 (px4_msgs)
        PX4_OFFBOARD_CONTROL_MODE = '/fmu/in/offboard_control_mode'
        PX4_TRAJECTORY_SETPOINT = '/fmu/in/trajectory_setpoint'
        PX4_VEHICLE_GLOBAL_POSITION = '/fmu/out/vehicle_global_position'
        PX4_VEHICLE_LOCAL_POSITION = '/fmu/out/vehicle_local_position'
        PX4_POSITION_SETPOINT_TRIPLET = '/fmu/out/position_setpoint_triplet'
        PX4_VEHICLE_ODOMETRY = '/fmu/out/vehicle_odometry'  # 각속도, 위치, 속도
        PX4_VEHICLE_LOCAL_POSITION_SUB = '/fmu/out/vehicle_local_position'  # NED 위치, 속도, 가속도, heading

        # Livox LiDAR IMU 토픽 (각속도 데이터)
        LIVOX_IMU = '/livox/imu'  # Livox LiDAR 내장 IMU










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
        AXIS_MARGIN_X = 40.0
        AXIS_MARGIN_Y = 40.0

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
    # 시각화 파라미터 (VisualizationSystem)
    # ============================================================================
    class VisualizationParams:
        """시각화 시스템 파라미터"""
        # 탐지 임계값 (NanoOWL: 낮을수록 더 많은 탐지, 일반적으로 0.01~0.1 사용)
        DETECTION_THRESHOLD = 0.1  # 0.0004에서 0.01로 상향 조정 (탐지 개선)
        MIN_BOX_AREA = 2
        MAX_BOX_AREA = 800000
        MIN_DEPTH_THRESHOLD = 0.12  # 최소 깊이 (미터)
        MAX_DEPTH_THRESHOLD = 0.9   # 최대 깊이 (미터)

        # Depth 필터링 파라미터
        SPATIAL_SMOOTHING_ENABLED = True  # 공간적 depth smoothing 활성화
        SPATIAL_KERNEL_SIZE = 31  # Spatial smoothing 커널 크기 (홀수 권장)
        TEMPORAL_FILTER_ALPHA = 0.3  # EMA 시간적 필터 smoothing factor (0.2-0.4 권장)

        # IMM-PDAF 파라미터
        MAX_COAST_FRAMES = 4
        GATE_THRESHOLD = 9.0

        # 윈도우 설정
        WINDOW_NAME = 'VRX Mission Control'
        WINDOW_WIDTH = 640
        WINDOW_HEIGHT = 480

        # 색상 매핑 (BGR)
        COLOR_RED_CONE = (0, 0, 255)
        COLOR_GREEN_CONE = (0, 255, 0)
        COLOR_BLUE_BUOY = (255,0, 0)
        COLOR_ACCUMULATED_ANGLE = (0, 255, 255)  # 누적 각도 텍스트 색상








    # ============================================================================
    # 장애물 회피 제어 파라미터 (AvoidControl)
    # ============================================================================
    class AvoidControl:
        """장애물 회피 컨트롤러 파라미터 (avoid_control.py에서 사용)"""

        # LOS (Line of Sight) Guidance 파라미터 (기본값)
        LOS_DELTA = 10.0              # 수직 오프셋 (미터)
        LOS_LOOKAHEAD_MIN = 30.0      # 최소 look-ahead 거리 (미터)
        LOS_LOOKAHEAD_MAX = 80.0      # 최대 look-ahead 거리 (미터)
        LOS_LOOKAHEAD_FACTOR = 1.0    # look-ahead 거리 계산 계수
        
        # 미션별 LOS Delta 파라미터 (미션 파라미터에서 오버라이드 가능)
        DOCK_MODE_LOS_DELTA = 15.0    # DOCK_MODE 미션 LOS delta
        OBSTACLE_AVOID_LOS_DELTA = 10.0  # OBSTACLE_AVOID 미션 LOS delta
        CIRCLE_BUOY_LOS_DELTA = 10.0  # CIRCLE_BUOY 미션 LOS delta

        # ObstacleDetector 파라미터
        OBSTACLE_BOAT_WIDTH = 2.2     # 배 폭 (미터) - 장애물 검사용
        OBSTACLE_BOAT_HEIGHT = 50.0   # 배 높이/길이 (미터)
        OBSTACLE_COUNT_THRESHOLD = 8  # ONNX 모드 전환 최소 감지 개수
        OBSTACLE_COUNT_THRESHOLD_OFF = 8  # DIRECT 모드로 복귀하는 장애물 개수 (히스테리시스)

        # DirectController 속도 임계값 (거리별 선속도)
        DIRECT_SPEED_FAR_DISTANCE = 20.0   # 먼 거리 기준 (미터)
        DIRECT_SPEED_FAR = 1.0             # 먼 거리 속도
        DIRECT_SPEED_MID_DISTANCE = 10.0   # 중간 거리 기준 (미터)
        DIRECT_SPEED_MID = 0.6             # 중간 거리 속도
        DIRECT_SPEED_NEAR = 0.4            # 가까운 거리 속도

        # 긴급 정면 검사
        L_FRONT = 20.0                # 정면 긴급 장애물 검사 거리 (미터)

        # 1차 저주파 필터 (Low Pass Filter)
        LPF_ALPHA = 0.35              # 필터 계수 (0-1, 낮을수록 부드러움)

        # 각속도 클리핑
        ANGULAR_VELOCITY_CLIP = 0.7   # 최대 각속도 (-0.7 ~ 0.7)

    # ============================================================================
    # 스러스터 제어 파라미터 (ThrusterControl)
    # ============================================================================
    class ThrusterControl:
        """스러스터 및 루프 제어 파라미터"""

        # 스러스터 출력 범위
        THRUST_MIN = -2000            # 최소 추력
        THRUST_MAX = 2000             # 최대 추력

        # 제어 루프 최적화 (Jetson 최적화용)
        PARAM_UPDATE_INTERVAL = 1000  # 파라미터 업데이트 주기 (프레임)
        ROS_PUBLISH_SKIP_FRAMES = 5   # ROS 발행 스킵 프레임 수

        # 속도-추력 변환
        DIRECT_THRUST_SCALE = 2000    # main_avoid.py에서 사용

    # ============================================================================
    # 환경별 오버라이드 (Environment Overrides)
    # ============================================================================
    class EnvOverrides:
        """
        환경별 파라미터 오버라이드

        CURRENT_ENVIRONMENT에 따라 기본값을 덮어씁니다.
        여기에 없는 파라미터는 위의 기본값을 사용합니다.
        """

        # 실제 PX4 환경 오버라이드
        REAL_PX4 = {
            'AvoidControl': {
                'LOS_DELTA': 15.0,           # 실제 환경에서 더 작은 오프셋
                'LOS_LOOKAHEAD_MIN': 20.0,   # 실제 환경에서 더 짧은 look-ahead
                'LOS_LOOKAHEAD_MAX': 50.0,
                'LPF_ALPHA': 0.25,           # 더 부드러운 필터링
            },
            'ThrusterControl': {
                'ROS_PUBLISH_SKIP_FRAMES': 3,  # 더 빈번한 발행
            },
            'PX4': {
                'MAX_VELOCITY': 1.5,         # 실제 환경에서 더 낮은 최대 속도
            }
        }

        # 시뮬레이터 환경 (기본값 사용)
        SIMULATOR = {}








    # ============================================================================
    # 유틸리티 메서드
    # ============================================================================
    @classmethod
    def get_param(cls, category: str, param: str):
        """
        환경에 따른 파라미터 값 반환

        Args:
            category: 파라미터 카테고리 (예: 'AvoidControl', 'ThrusterControl')
            param: 파라미터 이름 (예: 'LOS_DELTA', 'THRUST_MAX')

        Returns:
            환경에 맞는 파라미터 값

        Example:
            >>> Constants.get_param('AvoidControl', 'LOS_DELTA')
            10.0  # 시뮬레이터 환경
            15.0  # 실제 PX4 환경
        """
        # 기본값 가져오기
        category_class = getattr(cls, category, None)
        if category_class is None:
            raise ValueError(f"Unknown category: {category}")

        base_value = getattr(category_class, param, None)
        if base_value is None:
            raise ValueError(f"Unknown param: {category}.{param}")

        # 환경별 오버라이드 확인
        if CURRENT_ENVIRONMENT == Environment.REAL_PX4:
            overrides = cls.EnvOverrides.REAL_PX4
            if category in overrides and param in overrides[category]:
                return overrides[category][param]

        return base_value

    @classmethod
    def get_current_environment(cls) -> str:
        """현재 환경 반환"""
        return CURRENT_ENVIRONMENT

    @classmethod
    def is_real_environment(cls) -> bool:
        """실제 환경인지 확인"""
        return CURRENT_ENVIRONMENT == Environment.REAL_PX4

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        설정 검증 및 진단 정보 반환

        Returns:
            Dict: 설정 상태 정보
                - 'valid': 설정이 유효한지
                - 'warnings': 경고 메시지 리스트
                - 'info': 정보 메시지 딕셔너리
        """
        warnings = []
        info = {}

        # ONNX 버전 확인
        if cls.ONNX_VERSION not in [1, 2]:
            warnings.append(f"잘못된 ONNX_VERSION: {cls.ONNX_VERSION} (1 또는 2만 가능)")

        info['onnx_version'] = cls.ONNX_VERSION
        info['onnx_input_size'] = cls.ONNX_INPUT_SIZE if cls.ONNX_VERSION == 1 else 207 * cls.STACK_COUNT
        info['thrust_scale'] = cls.DEFAULT_THRUST_SCALE
        info['lidar_size'] = cls.LIDAR_ARRAY_SIZE
        info['total_waypoints'] = len(cls.PREDEFINED_WAYPOINTS)

        # 경로 검증
        try:
            model_path = cls.Paths.get_onnx_model_path()
            info['onnx_model'] = str(model_path)
        except FileNotFoundError as e:
            warnings.append(f"ONNX 모델 파일 없음: {e}")
            info['onnx_model'] = None

        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'info': info
        }








    @classmethod
    def print_config_summary(cls) -> None:
        """설정 요약 출력 (디버깅용)"""
        print("=" * 70)
        print("VRX System Configuration Summary")
        print("=" * 70)

        # 환경 정보
        env_name = "Simulator (Gazebo)" if CURRENT_ENVIRONMENT == Environment.SIMULATOR else "Real PX4 (Pixhawk)"
        print(f"\n🌍 현재 환경: {env_name}")

        validation = cls.validate_config()

        print(f"\n📊 기본 정보:")
        for key, value in validation['info'].items():
            print(f"  - {key}: {value}")

        # 환경별 오버라이드 정보
        if CURRENT_ENVIRONMENT == Environment.REAL_PX4:
            print(f"\n🔧 환경별 오버라이드 적용됨:")
            for category, params in cls.EnvOverrides.REAL_PX4.items():
                for param, value in params.items():
                    print(f"  - {category}.{param}: {value}")

        if validation['warnings']:
            print(f"\n⚠️ 경고 ({len(validation['warnings'])}개):")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        else:
            print(f"\n✅ 설정 검증 통과")

        print("=" * 70)
