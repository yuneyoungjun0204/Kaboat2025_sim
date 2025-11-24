#!/usr/bin/env python3
"""
IMM-PDAF (Interacting Multiple Model - Probabilistic Data Association Filter) Tracker
- 이미지 기반 객체 추적을 위한 강건한 필터링 시스템
- 해양 환경의 부표 추적에 최적화
- Multiple motion models (NCP, CV, CA, Singer) for buoy dynamics
- PDAF for handling clutter and missed detections
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from scipy.stats import chi2
from scipy.linalg import block_diag

# Depth filtering for temporal smoothing
from ..sensors.depth_filter import ExponentialMovingAverageFilter


class MotionModel(ABC):
    """모션 모델 베이스 클래스"""

    def __init__(self, dim_state: int = 6):
        self.dim_state = dim_state
        self.name = "BaseModel"

    @abstractmethod
    def get_F(self, dt: float) -> np.ndarray:
        """상태 전이 행렬 F"""
        pass

    @abstractmethod
    def get_Q(self, dt: float) -> np.ndarray:
        """프로세스 노이즈 공분산 행렬 Q"""
        pass


class NearlyConstantPosition(MotionModel):
    """Nearly Constant Position (NCP) 모델 - 정지 부표"""

    def __init__(self, sigma_pos: float = 0.1):
        super().__init__()
        self.name = "NCP"
        self.sigma_pos = sigma_pos  # 위치 노이즈 (pixels)

    def get_F(self, dt: float) -> np.ndarray:
        """위치만 유지, 속도/가속도는 감쇠"""
        F = np.eye(6)
        # 속도와 가속도를 0으로 감쇠시킴 (매우 느린 감쇠)
        decay = 0.95
        F[2, 2] = decay
        F[3, 3] = decay
        F[4, 4] = decay
        F[5, 5] = decay
        return F

    def get_Q(self, dt: float) -> np.ndarray:
        """매우 낮은 프로세스 노이즈"""
        q = self.sigma_pos ** 2
        Q = np.diag([q, q, q * 0.1, q * 0.1, q * 0.01, q * 0.01])
        return Q * dt


class ConstantVelocity(MotionModel):
    """Constant Velocity (CV) 모델 - 일정 속도 이동"""

    def __init__(self, sigma_vel: float = 1.5):
        super().__init__()
        self.name = "CV"
        self.sigma_vel = sigma_vel  # 속도 노이즈 (pixels/s)

    def get_F(self, dt: float) -> np.ndarray:
        """등속도 운동"""
        F = np.eye(6)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt
        # 가속도 감쇠
        F[4, 4] = 0.9
        F[5, 5] = 0.9
        return F

    def get_Q(self, dt: float) -> np.ndarray:
        """중간 수준의 프로세스 노이즈"""
        q_vel = self.sigma_vel ** 2

        # Continuous white noise acceleration model
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = np.eye(2) * (dt**3 / 3) * q_vel
        Q[0:2, 2:4] = np.eye(2) * (dt**2 / 2) * q_vel
        Q[2:4, 0:2] = np.eye(2) * (dt**2 / 2) * q_vel
        Q[2:4, 2:4] = np.eye(2) * dt * q_vel
        Q[4:6, 4:6] = np.eye(2) * q_vel * 0.1

        return Q


class ConstantAcceleration(MotionModel):
    """Constant Acceleration (CA) 모델 - 파도에 의한 가속 운동"""

    def __init__(self, sigma_acc: float = 0.8):
        super().__init__()
        self.name = "CA"
        self.sigma_acc = sigma_acc  # 가속도 노이즈 (pixels/s²)

    def get_F(self, dt: float) -> np.ndarray:
        """등가속도 운동"""
        F = np.eye(6)
        F[0, 2] = dt
        F[0, 4] = 0.5 * dt**2
        F[1, 3] = dt
        F[1, 5] = 0.5 * dt**2
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    def get_Q(self, dt: float) -> np.ndarray:
        """높은 프로세스 노이즈 (파도)"""
        q_acc = self.sigma_acc ** 2

        Q = np.zeros((6, 6))
        # Position covariance
        Q[0:2, 0:2] = np.eye(2) * (dt**5 / 20) * q_acc
        Q[0:2, 2:4] = np.eye(2) * (dt**4 / 8) * q_acc
        Q[0:2, 4:6] = np.eye(2) * (dt**3 / 6) * q_acc

        # Velocity covariance
        Q[2:4, 0:2] = np.eye(2) * (dt**4 / 8) * q_acc
        Q[2:4, 2:4] = np.eye(2) * (dt**3 / 3) * q_acc
        Q[2:4, 4:6] = np.eye(2) * (dt**2 / 2) * q_acc

        # Acceleration covariance
        Q[4:6, 0:2] = np.eye(2) * (dt**3 / 6) * q_acc
        Q[4:6, 2:4] = np.eye(2) * (dt**2 / 2) * q_acc
        Q[4:6, 4:6] = np.eye(2) * dt * q_acc

        return Q


class SingerModel(MotionModel):
    """Singer 모델 - 기동(maneuver) 모델"""

    def __init__(self, sigma_maneuver: float = 1.5, tau: float = 3.0):
        super().__init__()
        self.name = "Singer"
        self.sigma_maneuver = sigma_maneuver  # 기동 표준편차 (pixels/s²)
        self.tau = tau  # 기동 상관 시간 (seconds)

    def get_F(self, dt: float) -> np.ndarray:
        """Singer 상태 전이"""
        alpha = np.exp(-dt / self.tau)

        F = np.eye(6)
        F[0, 2] = dt
        F[0, 4] = self.tau * (1 - alpha) * dt - (1 - alpha)**2 * self.tau**2 / 2
        F[1, 3] = dt
        F[1, 5] = self.tau * (1 - alpha) * dt - (1 - alpha)**2 * self.tau**2 / 2
        F[2, 4] = 1 - alpha
        F[3, 5] = 1 - alpha
        F[4, 4] = alpha
        F[5, 5] = alpha

        return F

    def get_Q(self, dt: float) -> np.ndarray:
        """Singer 프로세스 노이즈"""
        alpha = np.exp(-dt / self.tau)
        q = 2 * self.sigma_maneuver**2 / self.tau

        # Simplified Singer Q matrix (for 2D case)
        Q = np.zeros((6, 6))

        q11 = q * (dt - 2 * self.tau * (1 - alpha) + self.tau / 2 * (1 - alpha**2))
        q12 = q * (self.tau * (1 - alpha) - self.tau / 2 * (1 - alpha**2))
        q13 = q * self.tau / 2 * (1 - alpha**2)
        q22 = q * self.tau * (1 - alpha**2)
        q23 = q * (1 - alpha**2)
        q33 = q * (1 - alpha**2)

        # x direction
        Q[0, 0] = q11
        Q[0, 2] = q12
        Q[0, 4] = q13
        Q[2, 0] = q12
        Q[2, 2] = q22
        Q[2, 4] = q23
        Q[4, 0] = q13
        Q[4, 2] = q23
        Q[4, 4] = q33

        # y direction
        Q[1, 1] = q11
        Q[1, 3] = q12
        Q[1, 5] = q13
        Q[3, 1] = q12
        Q[3, 3] = q22
        Q[3, 5] = q23
        Q[5, 1] = q13
        Q[5, 3] = q23
        Q[5, 5] = q33

        return Q


class KalmanFilter:
    """Extended Kalman Filter for single model"""

    def __init__(self, dim_state: int = 6):
        self.dim_state = dim_state
        self.x = np.zeros(dim_state)  # State
        self.P = np.eye(dim_state) * 100.0  # Covariance

    def predict(self, F: np.ndarray, Q: np.ndarray):
        """예측 단계"""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """업데이트 단계"""
        # Innovation
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        I = np.eye(self.dim_state)
        self.P = (I - K @ H) @ self.P

    def get_innovation(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """혁신(innovation) 및 공분산 계산"""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S


class IMMFilter:
    """Interacting Multiple Model Filter"""

    def __init__(self, models: List[MotionModel], transition_probs: np.ndarray):
        """
        Args:
            models: 모션 모델 리스트
            transition_probs: 모델 전이 확률 행렬 (M x M)
        """
        self.models = models
        self.num_models = len(models)
        self.transition_probs = transition_probs

        # 각 모델에 대한 칼만 필터
        self.filters = [KalmanFilter() for _ in range(self.num_models)]

        # 모델 확률 (초기에는 균등 분포)
        self.mu = np.ones(self.num_models) / self.num_models

        # 혼합된 초기 상태/공분산
        self.x_mixed = [np.zeros(6) for _ in range(self.num_models)]
        self.P_mixed = [np.eye(6) * 100.0 for _ in range(self.num_models)]

    def mixing_probabilities(self):
        """Mixing probabilities 계산"""
        # c_j: normalization constant
        c = self.transition_probs.T @ self.mu

        # mu_ij: mixing probability
        mu_ij = np.zeros((self.num_models, self.num_models))
        for i in range(self.num_models):
            for j in range(self.num_models):
                if c[j] > 1e-10:
                    mu_ij[i, j] = self.transition_probs[i, j] * self.mu[i] / c[j]

        # Mixed state and covariance for each model
        for j in range(self.num_models):
            # Mixed state
            self.x_mixed[j] = sum(mu_ij[i, j] * self.filters[i].x for i in range(self.num_models))

            # Mixed covariance
            self.P_mixed[j] = np.zeros((6, 6))
            for i in range(self.num_models):
                diff = self.filters[i].x - self.x_mixed[j]
                self.P_mixed[j] += mu_ij[i, j] * (self.filters[i].P + np.outer(diff, diff))

    def predict(self, dt: float):
        """예측 단계 (모든 모델)"""
        self.mixing_probabilities()

        for j in range(self.num_models):
            # 혼합된 초기 상태로 설정
            self.filters[j].x = self.x_mixed[j].copy()
            self.filters[j].P = self.P_mixed[j].copy()

            # 각 모델로 예측
            F = self.models[j].get_F(dt)
            Q = self.models[j].get_Q(dt)
            self.filters[j].predict(F, Q)

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """업데이트 단계 (모든 모델)"""
        likelihoods = np.zeros(self.num_models)

        for j in range(self.num_models):
            # Innovation
            y, S = self.filters[j].get_innovation(z, H, R)

            # Likelihood (Gaussian PDF)
            det_S = np.linalg.det(S)
            if det_S > 1e-10:
                exp_part = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y)
                likelihoods[j] = (1.0 / np.sqrt((2 * np.pi)**len(z) * det_S)) * exp_part
            else:
                likelihoods[j] = 1e-10

            # Update this filter
            self.filters[j].update(z, H, R)

        # Update model probabilities
        c = self.transition_probs.T @ self.mu
        self.mu = c * likelihoods
        if self.mu.sum() > 1e-10:
            self.mu /= self.mu.sum()
        else:
            self.mu = np.ones(self.num_models) / self.num_models

    def combination(self) -> Tuple[np.ndarray, np.ndarray]:
        """Combined state and covariance"""
        # Combined state
        x_combined = sum(self.mu[j] * self.filters[j].x for j in range(self.num_models))

        # Combined covariance
        P_combined = np.zeros((6, 6))
        for j in range(self.num_models):
            diff = self.filters[j].x - x_combined
            P_combined += self.mu[j] * (self.filters[j].P + np.outer(diff, diff))

        return x_combined, P_combined


class PDAFilter:
    """Probabilistic Data Association Filter"""

    def __init__(self, P_D: float = 0.95, clutter_density: float = 1e-6, gate_threshold: float = 9.21):
        """
        Args:
            P_D: Detection probability
            clutter_density: Clutter spatial density (per pixel²)
            gate_threshold: Chi-square gate threshold (e.g., 9.21 for 99% with df=2)
        """
        self.P_D = P_D
        self.clutter_density = clutter_density
        self.gate_threshold = gate_threshold

    def validation_gate(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """Validation gate using Mahalanobis distance"""
        mahala_dist_sq = innovation.T @ np.linalg.inv(S) @ innovation
        return mahala_dist_sq <= self.gate_threshold

    def association_probabilities(self,
                                  measurements: List[np.ndarray],
                                  innovations: List[np.ndarray],
                                  S: np.ndarray) -> np.ndarray:
        """
        Calculate association probabilities for validated measurements

        Returns:
            beta: Association probabilities (length = 1 + len(measurements))
                  beta[0] = prob(no detection), beta[i>0] = prob(measurement i)
        """
        m = len(measurements)  # Number of validated measurements

        if m == 0:
            # No measurements in gate
            return np.array([1.0])

        # Volume of validation gate
        V = np.pi * np.sqrt(np.linalg.det(S)) * self.gate_threshold

        # Likelihoods for each measurement
        likelihoods = np.zeros(m)
        for i, y in enumerate(innovations):
            det_S = np.linalg.det(S)
            if det_S > 1e-10:
                exp_part = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y)
                likelihoods[i] = (1.0 / np.sqrt((2 * np.pi)**len(y) * det_S)) * exp_part
            else:
                likelihoods[i] = 1e-10

        # Association probabilities
        beta = np.zeros(m + 1)

        # beta[0]: probability of no detection
        beta[0] = (1 - self.P_D) * self.clutter_density * V

        # beta[i]: probability of detection i
        for i in range(m):
            beta[i + 1] = self.P_D * likelihoods[i]

        # Normalize
        if beta.sum() > 1e-10:
            beta /= beta.sum()
        else:
            beta[0] = 1.0

        return beta

    def pdaf_update(self,
                    kf: KalmanFilter,
                    measurements: List[np.ndarray],
                    H: np.ndarray,
                    R: np.ndarray) -> None:
        """
        PDAF update with probabilistic association

        Args:
            kf: Kalman filter to update
            measurements: List of measurements in validation gate
            H: Measurement matrix
            R: Measurement noise covariance
        """
        if not measurements:
            # No measurements - prediction only (already done)
            return

        # Get innovations and innovation covariance
        innovations = []
        y, S = kf.get_innovation(measurements[0], H, R)

        for z in measurements:
            y_i, _ = kf.get_innovation(z, H, R)
            innovations.append(y_i)

        # Association probabilities
        beta = self.association_probabilities(measurements, innovations, S)

        # Combined innovation (weighted average)
        y_combined = sum(beta[i + 1] * innovations[i] for i in range(len(measurements)))

        # Kalman gain
        K = kf.P @ H.T @ np.linalg.inv(S)

        # Update state
        kf.x = kf.x + K @ y_combined

        # Update covariance (accounting for association uncertainty)
        I = np.eye(kf.dim_state)
        P_c = (I - K @ H) @ kf.P  # Standard Kalman update

        # Spread of innovations (association uncertainty)
        P_spread = np.zeros((kf.dim_state, kf.dim_state))
        for i in range(len(measurements)):
            y_diff = innovations[i] - y_combined
            P_spread += beta[i + 1] * K @ np.outer(y_diff, y_diff) @ K.T

        # Combined covariance
        P_D_factor = sum(beta[1:])  # Probability of detection
        kf.P = P_D_factor * P_c + (1 - P_D_factor) * kf.P + P_spread


class Track:
    """Single target track with IMM-PDAF"""

    def __init__(self,
                 track_id: int,
                 label: str,
                 initial_measurement: np.ndarray,
                 models: List[MotionModel],
                 transition_probs: np.ndarray,
                 initial_covariance: float = 100.0,
                 depth_filter_alpha: float = 0.3):
        """
        Args:
            track_id: Unique track ID
            label: Object label (e.g., 'red_cone', 'green_cone', 'blue_buoy')
            initial_measurement: Initial measurement [x, y] or [x, y, depth]
            models: List of motion models
            transition_probs: Model transition probability matrix
            initial_covariance: Initial state covariance
            depth_filter_alpha: EMA smoothing factor for depth (0 < alpha <= 1)
        """
        self.track_id = track_id
        self.label = label
        self.coast_count = 0  # Consecutive frames without measurement
        self.age = 0  # Frames since initialization

        # IMM filter
        self.imm = IMMFilter(models, transition_probs)

        # Initialize state: [x, y, vx, vy, ax, ay]
        x_init = np.zeros(6)
        x_init[0] = initial_measurement[0]
        x_init[1] = initial_measurement[1]

        # Initialize all filters with same state
        for kf in self.imm.filters:
            kf.x = x_init.copy()
            kf.P = np.eye(6) * initial_covariance
            # Higher uncertainty in velocity/acceleration
            kf.P[2:4, 2:4] *= 10.0
            kf.P[4:6, 4:6] *= 100.0

        # PDA filter
        self.pda = PDAFilter()

        # State and covariance (combined from IMM)
        self.x, self.P = self.imm.combination()

        # Depth measurement with temporal EMA filtering
        self.depth = initial_measurement[2] if len(initial_measurement) > 2 else None
        self.depth_filter = ExponentialMovingAverageFilter(alpha=depth_filter_alpha)
        if self.depth is not None:
            self.depth_filter.update(self.depth)

    def predict(self, dt: float):
        """Predict step"""
        self.imm.predict(dt)
        self.x, self.P = self.imm.combination()
        self.age += 1

    def update_pdaf(self, measurements: List[Dict], dt: float, use_depth: bool = False):
        """
        Update with PDAF

        Args:
            measurements: List of detections in gate (each is dict with 'center', 'depth', etc.)
            dt: Time step
            use_depth: Whether to use depth in measurement
        """
        # Measurement matrix
        if use_depth:
            H = np.zeros((3, 6))
            H[0, 0] = 1  # x
            H[1, 1] = 1  # y
            H[2, 0] = 0  # depth (not directly in state, would need separate model)
            # For now, we track depth separately
            use_depth = False  # Disable for this implementation

        H = np.zeros((2, 6))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y

        # Measurement noise
        R = np.diag([5.0, 5.0])  # 5 pixels standard deviation

        if not measurements:
            # No measurements - increase coast count
            self.coast_count += 1
            # Prediction only (already done)
            return

        # Extract measurement vectors
        z_list = [np.array([m['center'][0], m['center'][1]]) for m in measurements]

        # Filter measurements by validation gate
        validated_measurements = []
        validated_dicts = []
        for z, m_dict in zip(z_list, measurements):
            y, S = self.imm.filters[0].get_innovation(z, H, R)  # Use first filter for gating
            if self.pda.validation_gate(y, S):
                validated_measurements.append(z)
                validated_dicts.append(m_dict)

        if not validated_measurements:
            # No validated measurements
            self.coast_count += 1
            return

        # Update each IMM filter with PDAF
        for kf in self.imm.filters:
            self.pda.pdaf_update(kf, validated_measurements, H, R)

        # Update model probabilities (use first validated measurement for likelihood)
        if validated_measurements:
            self.imm.update(validated_measurements[0], H, R)

        # Combine
        self.x, self.P = self.imm.combination()

        # Update depth (weighted average with temporal EMA filtering)
        if validated_dicts:
            y, S = self.imm.filters[0].get_innovation(validated_measurements[0], H, R)
            beta = self.pda.association_probabilities(validated_measurements,
                                                     [self.imm.filters[0].get_innovation(z, H, R)[0]
                                                      for z in validated_measurements],
                                                     S)
            # PDAF weighted average of raw depth measurements
            depth_weighted = sum(beta[i+1] * validated_dicts[i]['depth']
                                for i in range(len(validated_dicts)))

            # Apply temporal EMA low-pass filter to reduce high-frequency noise
            self.depth = self.depth_filter.update(depth_weighted)

        # Reset coast count
        self.coast_count = 0

    def get_state_dict(self) -> Dict:
        """Get current state as dictionary (compatible with detection format)"""
        return {
            'track_id': self.track_id,
            'label': self.label,
            'center': (int(self.x[0]), int(self.x[1])),
            'velocity': (self.x[2], self.x[3]),
            'acceleration': (self.x[4], self.x[5]),
            'depth': self.depth if self.depth is not None else 0.0,
            'covariance': self.P[:2, :2],  # Position covariance only
            'model_probs': self.imm.mu.copy(),
            'coast_count': self.coast_count,
            'age': self.age
        }


class IMMPDAFTracker:
    """Main IMM-PDAF tracker managing multiple tracks"""

    def __init__(self,
                 dt: float = 1/30.0,
                 P_D: float = 0.95,
                 clutter_density: float = 1e-6,
                 max_coast_frames: int = 10,
                 gate_threshold: float = 9.21,
                 depth_filter_alpha: float = 0.3):
        """
        Args:
            dt: Time step (default 1/30 for 30 fps)
            P_D: Detection probability
            clutter_density: Clutter spatial density
            max_coast_frames: Maximum frames without detection before track deletion
            gate_threshold: Chi-square gating threshold
            depth_filter_alpha: EMA smoothing factor for depth filtering (0 < alpha <= 1)
        """
        self.dt = dt
        self.max_coast_frames = max_coast_frames
        self.depth_filter_alpha = depth_filter_alpha

        # Motion models for maritime buoys
        self.models = [
            NearlyConstantPosition(sigma_pos=0.1),
            ConstantVelocity(sigma_vel=1.5),
            ConstantAcceleration(sigma_acc=0.8),
            SingerModel(sigma_maneuver=1.5, tau=3.0)
        ]

        # Model transition probabilities (sticky models)
        self.transition_probs = np.array([
            [0.97, 0.01, 0.01, 0.01],  # NCP
            [0.01, 0.97, 0.01, 0.01],  # CV
            [0.01, 0.01, 0.97, 0.01],  # CA
            [0.01, 0.01, 0.01, 0.97]   # Singer
        ])

        # Tracks
        self.tracks: List[Track] = []
        self.next_track_id = 0

        # PDA parameters
        self.P_D = P_D
        self.clutter_density = clutter_density
        self.gate_threshold = gate_threshold

    def initialize_track(self, detection: Dict) -> Track:
        """Initialize new track from detection"""
        measurement = np.array([
            detection['center'][0],
            detection['center'][1],
            detection.get('depth', 0.0)
        ])

        track = Track(
            track_id=self.next_track_id,
            label=detection['label'],
            initial_measurement=measurement,
            models=self.models,
            transition_probs=self.transition_probs,
            initial_covariance=100.0,
            depth_filter_alpha=self.depth_filter_alpha
        )

        self.next_track_id += 1
        return track

    def predict_tracks(self):
        """Predict all tracks"""
        for track in self.tracks:
            track.predict(self.dt)

    def update_tracks(self, detections: List[Dict]):
        """
        Update tracks with new detections

        Args:
            detections: List of detections from DetectionSystem
        """
        # Group detections by label
        detections_by_label = {}
        for det in detections:
            label = det['label']
            if label not in detections_by_label:
                detections_by_label[label] = []
            detections_by_label[label].append(det)

        # Update existing tracks
        for track in self.tracks:
            label_detections = detections_by_label.get(track.label, [])
            track.update_pdaf(label_detections, self.dt)

        # Create new tracks for unassociated detections
        # (Simple strategy: create track if no existing track for this label)
        for label, dets in detections_by_label.items():
            # Check if we have a track for this label
            has_track = any(t.label == label and t.coast_count < self.max_coast_frames
                           for t in self.tracks)

            if not has_track and dets:
                # Initialize track with highest confidence detection
                best_det = max(dets, key=lambda d: d.get('confidence', 0.0))
                new_track = self.initialize_track(best_det)
                self.tracks.append(new_track)

    def prune_tracks(self):
        """Remove tracks that have been coasting too long"""
        self.tracks = [t for t in self.tracks if t.coast_count < self.max_coast_frames]

    def get_tracked_objects(self) -> List[Dict]:
        """
        Get tracked objects in detection-compatible format

        Returns:
            List of tracked detections with smoothed positions and velocity info
        """
        tracked_objects = []
        for track in self.tracks:
            if track.coast_count < 3:  # Only return recently updated tracks
                obj = track.get_state_dict()
                # Add bbox for compatibility (estimate from covariance)
                std_x = np.sqrt(track.P[0, 0])
                std_y = np.sqrt(track.P[1, 1])
                cx, cy = int(track.x[0]), int(track.x[1])
                w, h = int(std_x * 4), int(std_y * 4)  # ±2σ
                obj['bbox'] = (cx - w//2, cy - h//2, cx + w//2, cy + h//2)
                obj['confidence'] = 0.95 * (1.0 - track.coast_count / self.max_coast_frames)
                tracked_objects.append(obj)

        return tracked_objects

    def reset(self):
        """Reset all tracks"""
        self.tracks = []
        self.next_track_id = 0


# Convenience function for integration
def create_tracker(fps: float = 30.0, **kwargs) -> IMMPDAFTracker:
    """
    Create IMM-PDAF tracker with default parameters

    Args:
        fps: Frame rate (default 30 Hz)
        **kwargs: Additional parameters for IMMPDAFTracker

    Returns:
        Configured IMMPDAFTracker instance
    """
    dt = 1.0 / fps
    return IMMPDAFTracker(dt=dt, **kwargs)
