#!/usr/bin/env python3
"""
ONNX 모델 제어 모듈
"""

import numpy as np
import onnxruntime as ort
from collections import deque
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

        # Observation history for temporal stacking (동적 stack count 지원)
        # maxlen=STACK_COUNT로 설정하여 자동으로 오래된 observation 제거
        self.observation_history = deque(maxlen=Constants.STACK_COUNT)

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """
        ONNX 모델 로딩

        Args:
            model_path: ONNX 모델 파일 경로

        Raises:
            FileNotFoundError: 모델 파일이 존재하지 않을 때
            RuntimeError: ONNX 런타임 초기화 실패 시
        """
        self.logger.info(f"ONNX 모델 로딩 중: {model_path}")
        try:
            from pathlib import Path
            if not Path(model_path).exists():
                raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {model_path}")

            self.onnx_session = ort.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name

            # 모델 입력 크기 검증
            expected_size = Constants.ONNX_INPUT_SIZE
            actual_size = self.onnx_session.get_inputs()[0].shape[1]
            if actual_size != expected_size:
                self.logger.warn(
                    f"⚠️ 모델 입력 크기 불일치: 예상={expected_size}, 실제={actual_size}"
                )

            self.logger.info("✓ ONNX 모델 로딩 완료")
        except FileNotFoundError as e:
            self.logger.error(f"❌ {e}")
            self.onnx_session = None
            raise
        except Exception as e:
            self.logger.error(f"❌ ONNX 모델 로딩 실패: {type(e).__name__}: {e}")
            self.onnx_session = None
            raise RuntimeError(f"ONNX 모델 로딩 실패: {e}") from e

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

            # Observation history에 현재 observation 추가
            self.observation_history.append(observation_array)

            # Cold start 처리: history가 STACK_COUNT보다 적으면 현재 observation으로 채우기
            # 예: STACK_COUNT=3이고 첫 프레임이면 [obs, obs, obs]로 채움
            while len(self.observation_history) < Constants.STACK_COUNT:
                self.observation_history.appendleft(observation_array)

            # Temporal stacking: [t-n+1, t-n+2, ..., t-1, t] 순서로 concatenate
            # 예: STACK_COUNT=2 → [t-1, t], STACK_COUNT=3 → [t-2, t-1, t]
            stacked_input = np.concatenate(list(self.observation_history)).reshape(1, Constants.ONNX_INPUT_SIZE)

            outputs = self.onnx_session.run(None, {self.onnx_input_name: stacked_input})

            return self._parse_output(outputs)

        except Exception as e:
            self.logger.error(f"ONNX 추론 오류: {e}")
            return 0.0, 0.0

    def _build_observation(self, lidar_distances: np.ndarray, agent_heading: float,
                          angular_velocity_y: float, agent_position: np.ndarray,
                          current_waypoint: np.ndarray, previous_waypoint: np.ndarray,
                          next_waypoint: np.ndarray) -> np.ndarray:
        """
        ONNX 모델용 단일 타임스텝 관측값 구성

        Args:
            lidar_distances: LiDAR 거리 배열 (201개, -100° ~ +100°)
            agent_heading: 에이전트 방향 (-180~180도, NED 좌표계)
            angular_velocity_y: Z축 각속도 (deg/s, + = CCW, - = CW)
            agent_position: 에이전트 위치 [North, East] (미터)
            current_waypoint: 현재 웨이포인트 [North, East]
            previous_waypoint: 이전 웨이포인트 [North, East]
            next_waypoint: 다음 웨이포인트 [North, East]

        Returns:
            관측값 배열 (크기: Constants.OBSERVATION_SIZE, 기본값 213)
            최종 모델 입력은 이 배열을 STACK_COUNT번 쌓아서 생성됨
        """
        observation_values = list(lidar_distances) + [
            float(agent_heading),         # -180~180도
            float(angular_velocity_y)     # deg/s
        ]

        for waypoint in [[agent_position[1],agent_position[0]], [current_waypoint[1],current_waypoint[0]], [previous_waypoint[1],previous_waypoint[0]], [next_waypoint[1],next_waypoint[0]]]:
            observation_values.extend([0.0 if np.isinf(v) or np.isnan(v) else float(v) for v in waypoint[:2]])

        observation_values.extend([
            float(self.previous_moment_input),
            float(self.previous_force_input)
        ])

        return np.array(observation_values, dtype=np.float32)

    def _parse_output(self, outputs: List) -> Tuple[float, float]:
        """ONNX 모델 출력 파싱 (differential drive 제약 조건 포함)"""
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

            # Differential drive 제약: left=linear+angular, right=linear-angular ∈ [-1,1]
            max_angular = min(1.0 - linear_velocity, linear_velocity + 1.0)
            min_angular = max(-1.0 - linear_velocity, linear_velocity - 1.0)
            angular_velocity = np.clip(angular_velocity, min_angular, max_angular)

        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        return linear_velocity, angular_velocity

    def update_previous_inputs(self, angular_velocity: float, linear_velocity: float) -> None:
        """이전 입력 업데이트 (필터 적용 후 호출, temporal context용)"""
        self.previous_moment_input = angular_velocity
        self.previous_force_input = linear_velocity
