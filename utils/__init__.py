"""
VRX 로봇 제어 시스템 유틸리티 모듈
- 깊이 추정, 센서 데이터 전처리, 장애물 회피 제어
- 객체 탐지 시스템 (NanoOWL + MiDaS)
- 미션 전략 및 관리
- 시각화 시스템
- 웨이포인트 관리
- ROS2 통신 관리
"""

try:
    # 센서 및 제어 모듈
    from .depth_estimation import MiDaSHybridDepthEstimator
    from .sensor_preprocessing import GPSTransformer, LiDARProcessor, IMUProcessor, SensorDataManager
    from .avoid_control import (
        LOSGuidance, ObstacleDetector, DirectController,
        LowPassFilter, AvoidanceController
    )

    # 탐지 시스템
    from .detection_system import DetectionSystem, MissionType

    # 미션 전략
    from .mission_strategies import (
        BaseMissionStrategy,
        PassBetweenBuoysMission,
        CircleBuoyMission,
        WaypointFollowMission,
        ObstacleAvoidMission,
        MissionManager
    )

    # 시각화 시스템
    from .visualization_system import VisualizationSystem

    # 웨이포인트 관리
    from .waypoint_manager import WaypointManager

    # ROS2 통신
    from .ros_communication import ROSCommunicationManager

    # IMM-PDAF 트래커
    from .imm_pdaf_tracker import (
        IMMPDAFTracker, Track, create_tracker,
        MotionModel, NearlyConstantPosition, ConstantVelocity,
        ConstantAcceleration, SingerModel
    )

    __all__ = [
        # 센서 및 제어
        'MiDaSHybridDepthEstimator',
        'GPSTransformer',
        'LiDARProcessor',
        'IMUProcessor',
        'SensorDataManager',
        'LOSGuidance',
        'ObstacleDetector',
        'DirectController',
        'LowPassFilter',
        'AvoidanceController',

        # 탐지 시스템
        'DetectionSystem',
        'MissionType',

        # 미션 전략
        'BaseMissionStrategy',
        'PassBetweenBuoysMission',
        'CircleBuoyMission',
        'WaypointFollowMission',
        'ObstacleAvoidMission',
        'MissionManager',

        # 시각화
        'VisualizationSystem',

        # 웨이포인트
        'WaypointManager',

        # ROS2 통신
        'ROSCommunicationManager',

        # IMM-PDAF 트래커
        'IMMPDAFTracker',
        'Track',
        'create_tracker',
        'MotionModel',
        'NearlyConstantPosition',
        'ConstantVelocity',
        'ConstantAcceleration',
        'SingerModel'
    ]
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    __all__ = []
