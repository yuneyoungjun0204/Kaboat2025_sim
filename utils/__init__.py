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
    # Core 모듈
    from .core.config import Constants
    from .core.helpers import normalize_heading, calculate_heading_error, find_buoy_with_fallback
    from .core.system_factory import VRXSystemFactory, QuickStart
    from .core.super_optimizer import SuperOptimizer, create_super_optimizer
    from .core.jetson_optimizer import setup_jetson

    # Sensors 모듈
    from .sensors.depth_estimation import MiDaSHybridDepthEstimator
    from .sensors.sensor_preprocessing import GPSTransformer, LiDARProcessor, IMUProcessor, SensorDataManager, normalize_angle_180
    from .sensors.sensor_callbacks import SensorCallbackHandler, LidarFilter
    from .sensors.depth_estimation_optimized import OptimizedDepthEstimator
    from .sensors.depth_estimation_ultra import UltraDepthEstimator, create_ultra_depth_estimator
    from .sensors.depth_filter import smooth_depth_spatially

    # Detection 모듈
    from .detection.detection_system import DetectionSystem, MissionType
    from .detection.detection_system_optimized import DetectionSystem as OptimizedDetectionSystem
    from .detection.imm_pdaf_tracker import (
        IMMPDAFTracker, Track, create_tracker,
        MotionModel, NearlyConstantPosition, ConstantVelocity,
        ConstantAcceleration, SingerModel
    )

    # Control 모듈
    from .control.avoid_control import (
        LOSGuidance, ObstacleDetector, DirectController,
        LowPassFilter, AvoidanceController
    )
    from .control.thruster_allocation import body_forces_to_thruster_commands
    from .control.onnx_controller import ONNXController

    # Mission 모듈
    from .mission.mission_strategies_new import (
        BaseMissionStrategy,
        PassBetweenBuoysMission,
        CircleBuoyMission,
        WaypointFollowMission,
        ObstacleAvoidMission,
        MissionManager
    )
    from .mission.mission_control import (
        MissionLoopExecutor,
        WaypointTransitionHandler,
        ObstacleAvoidExecutor
    )
    from .mission.mission_executor import MissionExecutor
    from .mission.waypoint_manager import WaypointManager, gps_to_local, local_to_gps
    from .mission.parameter_manager import ParameterManager

    # Visualization 모듈
    from .visualization.visualization_system import VisualizationSystem
    from .visualization.viz_components import PlotManager, VizCallbackHandler, VizUtils
    from .visualization.image_preprocessor import create_preprocessor

    # Communication 모듈
    from .communication.ros_communication import ROSCommunicationManager
    from .communication.px4_adapter import (
        PX4SensorAdapter,
        PX4CommandConverter,
        CoordinateConverter,
        PX4BridgeData,
        NEDPosition,
        VelocityYawCommand
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
        'SingerModel',

        # 새로운 유틸리티
        'Constants',
        'ParameterManager',
        'SensorCallbackHandler',
        'ONNXController',
        'MissionExecutor',

        # System Factory
        'VRXSystemFactory',
        'QuickStart',

        # Visualization Components
        'PlotManager',
        'VizCallbackHandler',
        'VizUtils',

        # Mission Control Components
        'MissionLoopExecutor',
        'WaypointTransitionHandler',
        'ObstacleAvoidExecutor',

        # PX4 Adapter
        'PX4SensorAdapter',
        'PX4CommandConverter',
        'CoordinateConverter',
        'PX4BridgeData',
        'NEDPosition',
        'VelocityYawCommand'
    ]
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    __all__ = []
