#!/usr/bin/env python3
"""
VRX 시스템 컴포넌트 Factory
- 컴포넌트 생성을 간소화하는 Factory 패턴
- 의존성 주입 및 초기화 자동화
"""

import torch
from typing import Dict, Any, Tuple, Optional
from rclpy.node import Node
from cv_bridge import CvBridge

from .config import Constants
from .depth_estimation import MiDaSHybridDepthEstimator
from .depth_estimation_optimized import OptimizedDepthEstimator
from .detection_system import DetectionSystem
from .sensor_preprocessing import SensorDataManager
from .sensor_callbacks import SensorCallbackHandler
from .avoid_control import AvoidanceController
from .mission_strategies_new import MissionManager
from .waypoint_manager import WaypointManager
from .mission_executor import MissionExecutor
from .visualization_system import VisualizationSystem
from .parameter_manager import ParameterManager
from .onnx_controller import ONNXController
from .imm_pdaf_tracker import create_tracker
from .ros_communication import ROSCommunicationManager
from .image_preprocessor import create_preprocessor


class VRXSystemFactory:
    """
    VRX 시스템 컴포넌트 Factory

    개발자 친화적인 인터페이스를 제공하여 시스템 초기화를 간소화합니다.

    Example:
        >>> factory = VRXSystemFactory(node, bridge, logger)
        >>> components = factory.create_all_components()
        >>> detection_system = components['detection_system']
    """

    def __init__(self, node: Node, bridge: CvBridge, logger):
        """
        Args:
            node: ROS2 노드 인스턴스
            bridge: CvBridge 인스턴스
            logger: ROS2 logger
        """
        self.node = node
        self.bridge = bridge
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_depth_estimator(self) -> OptimizedDepthEstimator:
        """
        최적화된 깊이 추정기 생성 (TensorRT + FP16)

        Returns:
            OptimizedDepthEstimator: 최적화된 깊이 추정기 인스턴스
        """
        self.logger.info("최적화된 깊이 추정기 초기화 중...")
        estimator = OptimizedDepthEstimator(
            model_type="DPT_Hybrid",
            input_size=256,  # 256x256 for speed
            use_tensorrt=False,  # TensorRT는 변환 필요 시 True
            device=self.device
        )
        self.logger.info("✓ 최적화된 깊이 추정기 초기화 완료 (FP16, 256x256)")
        return estimator

    def create_detection_system(
        self,
        depth_estimator: Optional[OptimizedDepthEstimator] = None,
        enable_preprocessing: bool = True
    ) -> DetectionSystem:
        """
        객체 탐지 시스템 생성

        Args:
            depth_estimator: 깊이 추정기 (None이면 자동 생성)
            enable_preprocessing: 이미지 전처리 활성화 (Depth만 4배 축소)

        Returns:
            DetectionSystem: 객체 탐지 시스템 인스턴스
        """
        self.logger.info("객체 탐지 시스템 초기화 중...")
        if depth_estimator is None:
            depth_estimator = self.create_depth_estimator()

        detection_system = DetectionSystem(
            depth_estimator,
            device=self.device,
            spatial_smoothing=Constants.VisualizationParams.SPATIAL_SMOOTHING_ENABLED,
            spatial_kernel_size=Constants.VisualizationParams.SPATIAL_KERNEL_SIZE
        )

        # 전처리기 설정 (Detection: 원본, Depth: 4배 축소)
        if enable_preprocessing:
            # Detection용: 원본 유지 (또는 약간 축소)
            image_preprocessor = None  # 원본 사용

            # Depth용: 4배 축소 (1280x720 → 320x180)
            depth_preprocessor = create_preprocessor('max_speed')  # 416x320

            detection_system.set_preprocessors(
                image_preprocessor=image_preprocessor,
                depth_preprocessor=depth_preprocessor
            )
            self.logger.info("✓ 전처리 활성화: Detection=원본(1280x720), Depth=저해상도(416x320, 4배 축소)")

        self.logger.info("✓ 객체 탐지 시스템 초기화 완료 "
                        f"(Spatial smoothing: {Constants.VisualizationParams.SPATIAL_SMOOTHING_ENABLED}, "
                        f"Kernel size: {Constants.VisualizationParams.SPATIAL_KERNEL_SIZE})")
        return detection_system

    def create_sensor_system(self) -> Tuple[SensorDataManager, SensorCallbackHandler]:
        """
        센서 시스템 생성

        Returns:
            Tuple[SensorDataManager, SensorCallbackHandler]:
                센서 데이터 관리자와 콜백 핸들러
        """
        self.logger.info("센서 시스템 초기화 중...")
        sensor_manager = SensorDataManager()
        sensor_handler = SensorCallbackHandler(
            self.bridge, sensor_manager, self.logger
        )
        self.logger.info("✓ 센서 시스템 초기화 완료")
        return sensor_manager, sensor_handler

    def create_avoidance_controller(self) -> AvoidanceController:
        """
        장애물 회피 컨트롤러 생성

        Returns:
            AvoidanceController: 장애물 회피 컨트롤러 인스턴스
        """
        self.logger.info("장애물 회피 컨트롤러 초기화 중...")
        controller = AvoidanceController(
            boat_width=Constants.BOAT_WIDTH,
            boat_height=Constants.BOAT_HEIGHT,
            max_lidar_distance=Constants.MAX_LIDAR_DISTANCE,
            los_delta=Constants.LOS_DELTA,
            los_lookahead_min=Constants.LOS_LOOKAHEAD_MIN,
            los_lookahead_max=Constants.LOS_LOOKAHEAD_MAX,
            filter_alpha=Constants.FILTER_ALPHA,
            obstacle_count_threshold=Constants.LIDAR_OBSTACLE_COUNT_THRESHOLD
        )
        self.logger.info(f"✓ 장애물 회피 컨트롤러 초기화 완료 (장애물 감지 임계값: {Constants.LIDAR_OBSTACLE_COUNT_THRESHOLD}개)")
        return controller

    def create_mission_system(
        self,
        avoidance_controller: AvoidanceController
    ) -> Tuple[MissionManager, WaypointManager, MissionExecutor]:
        """
        미션 시스템 생성 (관리자 + 실행자)

        Args:
            avoidance_controller: 장애물 회피 컨트롤러

        Returns:
            Tuple[MissionManager, WaypointManager, MissionExecutor]:
                미션 관리자, 웨이포인트 관리자, 미션 실행자
        """
        self.logger.info("미션 시스템 초기화 중...")

        # 미션 관리자
        mission_manager = MissionManager(
            thrust_scale=Constants.DEFAULT_THRUST_SCALE,
            avoidance_controller=avoidance_controller
        )

        # 웨이포인트 관리자
        waypoint_manager = WaypointManager()
        waypoint_manager.setup_predefined_waypoints()

        # 웨이포인트 모드 정보 로깅
        mode_str = waypoint_manager.get_waypoint_mode_str()
        total_waypoints = waypoint_manager.get_total_waypoints()
        self.logger.info(
            f"✓ 웨이포인트 시스템: {mode_str} 모드, "
            f"총 {total_waypoints}개 웨이포인트"
        )
        if waypoint_manager.waypoint_mode == 1:
            self.logger.info(
                f"  GPS 기준점: ({Constants.GPS_REFERENCE_LAT:.6f}, {Constants.GPS_REFERENCE_LON:.6f})"
            )

        # 미션 실행자
        mission_executor = MissionExecutor(mission_manager, waypoint_manager)

        self.logger.info("✓ 미션 시스템 초기화 완료")
        return mission_manager, waypoint_manager, mission_executor

    def create_visualization_system(self) -> Tuple[VisualizationSystem, ParameterManager]:
        """
        시각화 시스템 및 파라미터 관리자 생성

        Returns:
            Tuple[VisualizationSystem, ParameterManager]:
                시각화 시스템과 파라미터 관리자
        """
        self.logger.info("시각화 시스템 초기화 중...")
        visualization = VisualizationSystem()
        param_manager = ParameterManager(visualization)
        self.logger.info("✓ 시각화 시스템 초기화 완료")
        return visualization, param_manager

    def create_onnx_controller(self) -> ONNXController:
        """
        ONNX 컨트롤러 생성

        Returns:
            ONNXController: ONNX 컨트롤러 인스턴스
        """
        self.logger.info("ONNX 컨트롤러 초기화 중...")
        model_path = Constants.Paths.get_onnx_model_path()
        controller = ONNXController(model_path, self.logger)
        self.logger.info("✓ ONNX 컨트롤러 초기화 완료")
        return controller

    def create_tracker(self):
        """
        IMM-PDAF 트래커 생성

        Returns:
            IMMPDAFTracker: 트래커 인스턴스
        """
        self.logger.info("IMM-PDAF 트래커 초기화 중...")
        tracker = create_tracker(
            fps=Constants.TRACKER_FPS,
            max_coast_frames=Constants.MAX_COAST_FRAMES,
            depth_filter_alpha=Constants.VisualizationParams.TEMPORAL_FILTER_ALPHA
        )
        self.logger.info("✓ IMM-PDAF 트래커 초기화 완료 "
                        f"(Temporal filter alpha: {Constants.VisualizationParams.TEMPORAL_FILTER_ALPHA})")
        return tracker

    def create_ros_communication(self, callbacks: Dict[str, Any]) -> ROSCommunicationManager:
        """
        ROS2 통신 관리자 생성 및 설정

        Args:
            callbacks: 콜백 함수 딕셔너리
                {
                    'image': image_callback,
                    'lidar': lidar_callback,
                    'gps': gps_callback,
                    'imu': imu_callback,
                    'waypoint': waypoint_callback
                }

        Returns:
            ROSCommunicationManager: ROS2 통신 관리자
        """
        self.logger.info("ROS2 통신 설정 중...")
        ros_comm = ROSCommunicationManager(self.node)
        ros_comm.setup_subscribers(callbacks)
        ros_comm.setup_publishers()
        self.logger.info("✓ ROS2 통신 설정 완료")
        return ros_comm

    def create_all_components(
        self,
        setup_ros: bool = False,
        ros_callbacks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        모든 시스템 컴포넌트를 한 번에 생성

        Args:
            setup_ros: ROS2 통신 설정 여부
            ros_callbacks: ROS2 콜백 함수들 (setup_ros=True일 때 필수)

        Returns:
            Dict[str, Any]: 생성된 모든 컴포넌트
                {
                    'depth_estimator': MiDaSHybridDepthEstimator,
                    'detection_system': DetectionSystem,
                    'sensor_manager': SensorDataManager,
                    'sensor_handler': SensorCallbackHandler,
                    'avoidance_controller': AvoidanceController,
                    'mission_manager': MissionManager,
                    'waypoint_manager': WaypointManager,
                    'mission_executor': MissionExecutor,
                    'visualization': VisualizationSystem,
                    'param_manager': ParameterManager,
                    'onnx_controller': ONNXController,
                    'tracker': IMMPDAFTracker,
                    'ros_comm': ROSCommunicationManager (setup_ros=True일 때)
                }

        Example:
            >>> factory = VRXSystemFactory(node, bridge, logger)
            >>> components = factory.create_all_components()
            >>> # 개별 컴포넌트 접근
            >>> detection = components['detection_system']
            >>> mission_mgr = components['mission_manager']
        """
        self.logger.info("=" * 80)
        self.logger.info("VRX 시스템 컴포넌트 초기화 시작")
        self.logger.info("=" * 80)

        # 1. 깊이 추정기
        depth_estimator = self.create_depth_estimator()

        # 2. 객체 탐지 시스템
        detection_system = self.create_detection_system(depth_estimator)

        # 3. 센서 시스템
        sensor_manager, sensor_handler = self.create_sensor_system()

        # 4. 장애물 회피 컨트롤러
        avoidance_controller = self.create_avoidance_controller()

        # 5. 미션 시스템
        mission_manager, waypoint_manager, mission_executor = \
            self.create_mission_system(avoidance_controller)

        # 6. 시각화 및 파라미터
        visualization, param_manager = self.create_visualization_system()

        # 7. ONNX 컨트롤러
        onnx_controller = self.create_onnx_controller()

        # 8. 트래커
        tracker = self.create_tracker()

        # sensor_handler에 waypoint_manager 설정 (GPS 콜백에서 초기 위치 설정용)
        sensor_handler.set_waypoint_manager(waypoint_manager)

        components = {
            'depth_estimator': depth_estimator,
            'detection_system': detection_system,
            'sensor_manager': sensor_manager,
            'sensor_handler': sensor_handler,
            'avoidance_controller': avoidance_controller,
            'mission_manager': mission_manager,
            'waypoint_manager': waypoint_manager,
            'mission_executor': mission_executor,
            'visualization': visualization,
            'param_manager': param_manager,
            'onnx_controller': onnx_controller,
            'tracker': tracker,
        }

        # 9. ROS2 통신 (선택적)
        if setup_ros:
            if ros_callbacks is None:
                raise ValueError("setup_ros=True일 때 ros_callbacks는 필수입니다")
            ros_comm = self.create_ros_communication(ros_callbacks)
            components['ros_comm'] = ros_comm

        self.logger.info("=" * 80)
        self.logger.info("✅ VRX 시스템 컴포넌트 초기화 완료!")
        self.logger.info("=" * 80)

        return components

    def create_minimal_components(self) -> Dict[str, Any]:
        """
        장애물 회피에 필요한 최소 컴포넌트만 생성 (탐지/트래킹 제외)

        Returns:
            Dict[str, Any]: 생성된 컴포넌트
                {
                    'sensor_handler': SensorCallbackHandler,
                    'avoidance_controller': AvoidanceController,
                    'mission_manager': MissionManager,
                    'waypoint_manager': WaypointManager,
                    'mission_executor': MissionExecutor,
                    'onnx_controller': ONNXController,
                }

        Example:
            >>> factory = VRXSystemFactory(node, None, logger)
            >>> components = factory.create_minimal_components()
            >>> # 장애물 회피만 사용
        """
        self.logger.info("=" * 80)
        self.logger.info("VRX 장애물 회피 시스템 초기화 (최소 구성)")
        self.logger.info("=" * 80)

        # 1. 센서 시스템
        sensor_manager, sensor_handler = self.create_sensor_system()

        # 2. 장애물 회피 컨트롤러
        avoidance_controller = self.create_avoidance_controller()

        # 3. 미션 시스템
        mission_manager, waypoint_manager, mission_executor = \
            self.create_mission_system(avoidance_controller)

        # 4. ONNX 컨트롤러
        onnx_controller = self.create_onnx_controller()

        # sensor_handler에 waypoint_manager 설정 (GPS 콜백에서 초기 위치 설정용)
        sensor_handler.set_waypoint_manager(waypoint_manager)

        components = {
            'sensor_handler': sensor_handler,
            'avoidance_controller': avoidance_controller,
            'mission_manager': mission_manager,
            'waypoint_manager': waypoint_manager,
            'mission_executor': mission_executor,
            'onnx_controller': onnx_controller,
        }

        self.logger.info("=" * 80)
        self.logger.info("✅ 장애물 회피 시스템 초기화 완료!")
        self.logger.info("=" * 80)

        return components


class QuickStart:
    """
    빠른 시작을 위한 헬퍼 클래스

    Example:
        >>> from utils import QuickStart
        >>> # ROS2 노드에서
        >>> components = QuickStart.setup_vrx_system(self)
    """

    @staticmethod
    def setup_vrx_system(
        node: Node,
        bridge: Optional[CvBridge] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        VRX 시스템 빠른 설정

        Args:
            node: ROS2 노드
            bridge: CvBridge (None이면 자동 생성)
            device: 디바이스 ("cuda" 또는 "cpu", None이면 자동 감지)

        Returns:
            Dict[str, Any]: 생성된 모든 컴포넌트
        """
        if bridge is None:
            bridge = CvBridge()

        factory = VRXSystemFactory(node, bridge, node.get_logger())

        # 디바이스 설정
        if device is not None:
            factory.device = device

        return factory.create_all_components()
