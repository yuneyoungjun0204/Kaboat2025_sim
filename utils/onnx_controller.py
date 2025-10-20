#!/usr/bin/env python3
"""
ONNX 모델 제어 모듈
"""

import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Optional
from utils.config import Constants


class ONNXController:
    """ONNX 모델 기반 제어 클래스"""

    def __init__(self, model_path: str, logger):
        """
        Args:
            model_path: ONNX 모델 경로
            logger: ROS2 logger
        """
        self.logger = logger
        self.onnx_session = None
        self.onnx_input_name = None
        self.previous_moment_input = 0.0
        self.previous_force_input = 0.0

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """ONNX 모델 로딩"""
        self.logger.info("ONNX 모델 로딩 중...")
        try:
            self.onnx_session = ort.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.logger.info("✓ ONNX 모델 로딩 완료")
        except Exception as e:
            self.logger.error(f"ONNX 모델 로딩 실패: {e}")
            self.onnx_session = None

    def get_control(self, lidar_distances: np.ndarray, agent_heading: float,
                   angular_velocity_y: float, agent_position: np.ndarray,
                   current_waypoint: np.ndarray, previous_waypoint: np.ndarray,
                   next_waypoint: np.ndarray) -> Tuple[float, float]:
        """
        ONNX 모델 기반 제어 명령 생성

        Args:
            lidar_distances: LiDAR 거리 배열
            agent_heading: 에이전트 방향
            angular_velocity_y: 각속도
            agent_position: 에이전트 위치
            current_waypoint: 현재 웨이포인트
            previous_waypoint: 이전 웨이포인트
            next_waypoint: 다음 웨이포인트

        Returns:
            (linear_velocity, angular_velocity) 튜플
        """
        if self.onnx_session is None:
            return 0.0, 0.0

        try:
            observation_array = self._build_observation(
                lidar_distances, agent_heading, angular_velocity_y,
                agent_position, current_waypoint, previous_waypoint, next_waypoint
            )

            stacked_input = np.concatenate([observation_array, observation_array]).reshape(
                1, Constants.ONNX_INPUT_SIZE
            )

            outputs = self.onnx_session.run(None, {self.onnx_input_name: stacked_input})

            return self._parse_output(outputs)

        except Exception as e:
            self.logger.error(f"ONNX 추론 오류: {e}")
            return 0.0, 0.0

    def _build_observation(self, lidar_distances: np.ndarray, agent_heading: float,
                          angular_velocity_y: float, agent_position: np.ndarray,
                          current_waypoint: np.ndarray, previous_waypoint: np.ndarray,
                          next_waypoint: np.ndarray) -> np.ndarray:
        """ONNX 모델용 관측값 구성"""
        observation_values = list(lidar_distances) + [
            float(agent_heading),
            float(angular_velocity_y)
        ]

        for val in [agent_position, current_waypoint, previous_waypoint, next_waypoint]:
            for i in range(2):
                v = float(val[i])
                if np.isinf(v) or np.isnan(v):
                    v = 0.0
                observation_values.append(v)

        observation_values.extend([
            float(self.previous_moment_input),
            float(self.previous_force_input)
        ])

        return np.array(observation_values, dtype=np.float32)

    def _parse_output(self, outputs: List) -> Tuple[float, float]:
        """ONNX 모델 출력 파싱"""
        if len(outputs) > 2 and outputs[2] is not None:
            linear_velocity = np.clip(
                outputs[4][0][1] * Constants.ONNX_V_SCALE,
                Constants.ONNX_LINEAR_VELOCITY_RANGE[0],
                Constants.ONNX_LINEAR_VELOCITY_RANGE[1]
            )
            angular_velocity = np.clip(
                outputs[4][0][0] * Constants.ONNX_W_SCALE,
                Constants.ONNX_ANGULAR_VELOCITY_RANGE[0],
                Constants.ONNX_ANGULAR_VELOCITY_RANGE[1]
            )
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        return linear_velocity, angular_velocity
