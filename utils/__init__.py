"""
VRX 로봇 제어 시스템 유틸리티 모듈
- 깊이 추정, 색상 필터링, 객체 검출, 추적, 네비게이션 제어 기능
- 장애물 회피 제어 기능
- 미션 베이스 및 개별 미션 모듈
"""

try:
    from .depth_estimation import MiDaSHybridDepthEstimator
    from .color_filtering import ColorFilter
    from .object_detection import BlobDetector
    from .object_tracking import Track, MultiTargetTracker
    from .navigation_control import PIDController, NavigationController
    from .thruster_control import ThrusterController
    from .visualization import Visualizer
    from .trackbar_control import TrackbarController
    from .matplotlib_visualizer import MatplotlibVisualizer
    from .sensor_preprocessing import GPSTransformer, LiDARProcessor, IMUProcessor, SensorDataManager
    from .avoid_control import (
        LOSGuidance, ObstacleDetector, DirectController, 
        LowPassFilter, AvoidanceController
    )
    from .base_mission import BaseMission, MissionStatus
    from .mission_gate import GateMission
    from .mission_circle import CircleMission
    from .mission_avoid import AvoidMission
    from .black_buoy_detector import BlackBuoyDetector
    from .blob_detector_advanced import AdvancedBlobDetector

    __all__ = [
        'MiDaSHybridDepthEstimator',
        'ColorFilter',
        'BlobDetector',
        'Track',
        'MultiTargetTracker',
        'PIDController',
        'NavigationController',
        'ThrusterController',
        'Visualizer',
        'TrackbarController',
        'MatplotlibVisualizer',
        'GPSTransformer',
        'LiDARProcessor',
        'IMUProcessor',
        'SensorDataManager',
        'LOSGuidance',
        'ObstacleDetector',
        'DirectController',
        'LowPassFilter',
        'AvoidanceController',
        'BaseMission',
        'MissionStatus',
        'GateMission',
        'CircleMission',
        'AvoidMission',
        'BlackBuoyDetector',
        'AdvancedBlobDetector'
    ]
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    __all__ = []
