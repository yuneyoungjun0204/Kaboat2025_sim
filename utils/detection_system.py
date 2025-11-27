#!/usr/bin/env python3
"""
객체 탐지 시스템 모듈
- NanoOWL 기반 객체 탐지
- MiDaS 깊이 필터링 통합
"""

import sys
import torch
import cv2
import numpy as np
from PIL import Image as PILImage
from enum import Enum
from typing import List, Dict, Optional

# config 모듈에서 경로 가져오기
from .config import Constants

# Depth filtering utilities
from .depth_filter import smooth_depth_spatially

# NanoOWL 경로 추가
sys.path.insert(0, str(Constants.Paths.NANOOWL_DIR))
from nanoowl.owl_predictor import OwlPredictor


class MissionType(Enum):
    """미션 타입 정의"""
    PASS_BETWEEN_BUOYS = 1
    CIRCLE_BUOY = 2
    WAYPOINT_FOLLOW = 3
    OBSTACLE_AVOID = 4
    HEADING_ALIGN = 5
    DOCK_MODE = 6
    ROTATION = 7  # 제자리 선회 미션


class DetectionSystem:
    """NanoOWL + MiDaS 통합 탐지 시스템"""

    def __init__(self, depth_estimator, device="cuda", detection_threshold=0.0065,
                 min_box_area=500, max_box_area=80000, min_depth=0.0, max_depth=50.0,
                 spatial_smoothing=True, spatial_kernel_size=5):
        """
        Args:
            depth_estimator: MiDaSHybridDepthEstimator 인스턴스
            device: 디바이스 (cuda/cpu)
            detection_threshold: 탐지 임계값
            min_box_area: 최소 박스 면적
            max_box_area: 최대 박스 면적
            min_depth: 최소 깊이 (미터)
            max_depth: 최대 깊이 (미터)
            spatial_smoothing: 공간적 depth smoothing 활성화 여부
            spatial_kernel_size: spatial smoothing 커널 크기 (홀수 권장)
        """
        self.device = device
        self.depth_estimator = depth_estimator

        # 탐지 파라미터
        self.detection_threshold = detection_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_depth_threshold = min_depth
        self.max_depth_threshold = max_depth

        # Depth smoothing 파라미터
        self.spatial_smoothing = spatial_smoothing
        self.spatial_kernel_size = spatial_kernel_size

        # Depth map 캐싱 (시각화용)
        self.last_depth_map = None

        # 이미지 전처리기 분리
        # image_preprocessor: Detection용 (원본 또는 약간 축소)
        # depth_preprocessor: Depth용 (많이 축소 - 4배)
        self.image_preprocessor = None
        self.depth_preprocessor = None

        # NanoOWL 초기화
        self._init_nanoowl()

    def _init_nanoowl(self):
        """NanoOWL 모델 초기화"""
        model_name = 'google/owlvit-base-patch32'
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None
        )

        # 미션별 탐지 쿼리 정의
        self.detection_queries = {
            MissionType.PASS_BETWEEN_BUOYS: {
                'queries': [
                    "a red cone buoy", "a red conical marker", "a cone-shaped red buoy",
                    "a green cone buoy", "a green conical marker", "a cone-shaped green buoy"
                ],
                'label_mapping': {
                    0: "red_cone", 1: "red_cone", 2: "red_cone",
                    3: "green_cone", 4: "green_cone", 5: "green_cone"
                }
            },
            MissionType.CIRCLE_BUOY: {
                'queries': [
                    "a blue buoy",
                ],
                'label_mapping': {
                    0: "blue_buoy"
                }
            },
            MissionType.DOCK_MODE: {
                'queries': [
                    "a red square"
                    # "a red circle", "a red square", "a red triangle",
                    # "a green circle", "a green square", "a green triangle",
                    # "a blue circle", "a blue square", "a blue triangle",
                    # "a yellow circle", "a yellow square", "a yellow triangle"
                ],
                'label_mapping': {
                    0: "red_square"
                    # 0: "red_circle", 1: "red_square", 2: "red_triangle",
                    # 3: "green_circle", 4: "green_square", 5: "green_triangle",
                    # 6: "blue_circle", 7: "blue_square", 8: "blue_triangle",
                    # 9: "yellow_circle", 10: "yellow_square", 11: "yellow_triangle"
                }
            }
        }

        # 텍스트 인코딩 미리 수행
        for mission_type, query_info in self.detection_queries.items():
            query_info['text_encodings'] = self.predictor.encode_text(query_info['queries'])

    def update_parameters(self, detection_threshold=None, min_box_area=None,
                         max_box_area=None, min_depth=None, max_depth=None):
        """탐지 파라미터 업데이트"""
        if detection_threshold is not None:
            self.detection_threshold = detection_threshold
        if min_box_area is not None:
            self.min_box_area = min_box_area
        if max_box_area is not None:
            self.max_box_area = max_box_area
        if min_depth is not None:
            self.min_depth_threshold = min_depth
        if max_depth is not None:
            self.max_depth_threshold = max_depth

    def set_preprocessors(self, image_preprocessor=None, depth_preprocessor=None):
        """
        전처리기 설정 (Detection용과 Depth용 분리)

        Args:
            image_preprocessor: Detection용 전처리기 (고해상도)
            depth_preprocessor: Depth용 전처리기 (저해상도 - 4배 축소)
        """
        self.image_preprocessor = image_preprocessor
        self.depth_preprocessor = depth_preprocessor

    def detect_objects(self, image: np.ndarray, mission_type: MissionType) -> List[Dict]:
        """
        객체 탐지 수행 (Jetson 최적화)

        Args:
            image: BGR 이미지
            mission_type: 현재 미션 타입

        Returns:
            탐지된 객체 리스트
        """
        if image is None:
            return []

        # 탐지가 필요 없는 미션은 스킵
        if mission_type not in [MissionType.PASS_BETWEEN_BUOYS, MissionType.CIRCLE_BUOY, MissionType.DOCK_MODE]:
            return []

        # 원본 이미지 저장
        original_image = image
        original_h, original_w = image.shape[:2]

        # === 1. Depth 전처리 (저해상도) ===
        depth_image = image
        depth_metadata = None
        if self.depth_preprocessor:
            depth_image, depth_metadata = self.depth_preprocessor.preprocess(image)
            if depth_image is None:
                return []

        # 깊이 맵 추정 (저해상도 이미지 사용)
        depth_map = self.depth_estimator.estimate_depth(depth_image)
        if depth_map is None:
            return []

        # 시각화를 위해 depth_map 저장
        self.last_depth_map = depth_map

        # Depth 스케일 계산 (original → depth map)
        depth_h, depth_w = depth_map.shape[:2]
        depth_scale_x = depth_w / original_w
        depth_scale_y = depth_h / original_h

        # === 2. Detection 전처리 (원본 또는 약간 축소) ===
        detection_image = original_image
        detection_metadata = None
        if self.image_preprocessor:
            detection_image, detection_metadata = self.image_preprocessor.preprocess(original_image)
            if detection_image is None:
                return []

        # Jetson 최적화: PIL 변환 없이 직접 RGB로 변환
        frame_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(frame_rgb)

        # 현재 미션에 맞는 쿼리 선택
        query_info = self.detection_queries.get(mission_type)
        if query_info is None:
            return []

        # NanoOWL 탐지
        output = self.predictor.predict(
            image=image_pil,
            text=query_info['queries'],
            text_encodings=query_info['text_encodings'],
            threshold=self.detection_threshold
        )

        # 결과 파싱
        detections = []
        for i in range(len(output.labels)):
            score = output.scores[i].item()
            bbox = output.boxes[i].detach().cpu().numpy()
            label_idx = output.labels[i].item()

            x1, y1, x2, y2 = [int(b) for b in bbox]

            # === Detection 좌표 → 원본 좌표 변환 ===
            if detection_metadata:
                # Detection 이미지 좌표를 원본 좌표로 복원
                scale_x_inv = 1.0 / detection_metadata['scale_x']
                scale_y_inv = 1.0 / detection_metadata['scale_y']
                x1_orig = int(x1 * scale_x_inv)
                y1_orig = int(y1 * scale_y_inv)
                x2_orig = int(x2 * scale_x_inv)
                y2_orig = int(y2 * scale_y_inv)
                cx_orig = (x1_orig + x2_orig) // 2
                cy_orig = (y1_orig + y2_orig) // 2
            else:
                # Detection이 원본 해상도
                x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2
                cx_orig = (x1 + x2) // 2
                cy_orig = (y1 + y2) // 2

            # 박스 크기 필터링 (원본 좌표 기준)
            area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
            if not (self.min_box_area <= area <= self.max_box_area):
                continue

            # === 원본 좌표 → Depth map 좌표 변환 ===
            cx_depth = int(cx_orig * depth_scale_x)
            cy_depth = int(cy_orig * depth_scale_y)
            cx_depth = max(0, min(depth_map.shape[1] - 1, cx_depth))
            cy_depth = max(0, min(depth_map.shape[0] - 1, cy_depth))

            # Spatial smoothing으로 주변 영역 평균 사용 (노이즈 감소)
            if self.spatial_smoothing:
                depth = smooth_depth_spatially(
                    depth_map, cx_depth, cy_depth, kernel_size=self.spatial_kernel_size
                )
            else:
                depth = depth_map[cy_depth, cx_depth]

            # 깊이 필터링
            if not (self.min_depth_threshold <= depth <= self.max_depth_threshold):
                continue

            label = query_info['label_mapping'].get(label_idx, "unknown")

            # 최종 결과는 원본 해상도 좌표 사용
            detections.append({
                "label": label,
                "confidence": score,
                "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                "center": (cx_orig, cy_orig),
                "depth": float(depth)
            })

        # 클래스별 최고 신뢰도만 선택
        detections = self._select_best_per_class(detections)

        return detections

    def _select_best_per_class(self, detections: List[Dict]) -> List[Dict]:
        """클래스별로 가장 높은 신뢰도의 객체만 선택"""
        if not detections:
            return []

        best_detections = {}
        for det in detections:
            label = det['label']
            if label not in best_detections or det['confidence'] > best_detections[label]['confidence']:
                best_detections[label] = det

        return list(best_detections.values())
