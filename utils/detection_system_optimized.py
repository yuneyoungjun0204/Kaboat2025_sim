#!/usr/bin/env python3
"""
최적화된 객체 탐지 시스템 (Jetson Orin Nano)
- 멀티스레딩으로 Detection과 Depth를 병렬 처리
- Frame skip으로 불필요한 연산 제거
- TensorRT 최적화 지원
- NanoOWL 대신 경량 모델 사용 가능
"""

import sys
import torch
import cv2
import numpy as np
from PIL import Image as PILImage
from enum import Enum
from typing import List, Dict, Optional, Tuple
import threading
import queue
import time

from .config import Constants
from .depth_filter import smooth_depth_spatially


class MissionType(Enum):
    """미션 타입 정의"""
    PASS_BETWEEN_BUOYS = 1
    CIRCLE_BUOY = 2
    WAYPOINT_FOLLOW = 3
    OBSTACLE_AVOID = 4
    HEADING_ALIGN = 5
    DOCK_MODE = 6
    ROTATION = 7


class AsyncWorker:
    """비동기 워커 스레드"""
    def __init__(self, name: str, process_func, max_queue_size=2):
        self.name = name
        self.process_func = process_func
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None

    def start(self):
        """스레드 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """스레드 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _worker_loop(self):
        """워커 루프"""
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = self.process_func(data)

                # 이전 결과 제거하고 최신 결과만 유지
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break

                self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ {self.name} 워커 에러: {e}")

    def submit(self, data):
        """작업 제출 (논블로킹)"""
        try:
            # 큐가 가득 차면 오래된 데이터 제거
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            self.input_queue.put_nowait(data)
        except queue.Full:
            pass

    def get_result(self, timeout=0.001):
        """결과 가져오기 (논블로킹)"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class OptimizedDetectionSystem:
    """
    최적화된 탐지 시스템 (Jetson Orin Nano)

    최적화 기법:
    1. 멀티스레딩: Detection과 Depth를 병렬 처리
    2. Frame skip: 매 N 프레임마다만 처리
    3. ROI 처리: 관심 영역만 처리
    4. 결과 캐싱: 이전 결과 재사용
    5. TensorRT 지원: FP16 최적화
    """

    def __init__(
        self,
        depth_estimator,
        device="cuda",
        detection_threshold=0.0065,
        min_box_area=500,
        max_box_area=80000,
        min_depth=0.0,
        max_depth=50.0,
        spatial_smoothing=True,
        spatial_kernel_size=5,
        detection_frame_skip=2,  # Detection은 2프레임마다
        depth_frame_skip=3,      # Depth는 3프레임마다
        use_async=True,          # 비동기 처리 사용
        use_nanoowl=True,        # NanoOWL 사용 (False면 생략 가능)
        roi_enabled=False,       # ROI 처리 사용
        roi_bounds=None          # ROI 영역 (x1, y1, x2, y2)
    ):
        self.device = device
        self.depth_estimator = depth_estimator
        self.use_nanoowl = use_nanoowl

        # 파라미터
        self.detection_threshold = detection_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_depth_threshold = min_depth
        self.max_depth_threshold = max_depth
        self.spatial_smoothing = spatial_smoothing
        self.spatial_kernel_size = spatial_kernel_size

        # Frame skip
        self.detection_frame_skip = detection_frame_skip
        self.depth_frame_skip = depth_frame_skip
        self.detection_frame_count = 0
        self.depth_frame_count = 0

        # ROI
        self.roi_enabled = roi_enabled
        self.roi_bounds = roi_bounds

        # 캐싱
        self.last_detections = []
        self.last_depth_map = None

        # 비동기 처리
        self.use_async = use_async
        if self.use_async:
            self.depth_worker = AsyncWorker("DepthEstimator", self._estimate_depth_async)
            self.depth_worker.start()

        # NanoOWL 초기화 (선택적)
        if self.use_nanoowl:
            self._init_nanoowl()
        else:
            print("⚠️ NanoOWL 비활성화 - Detection 기능 제한됨")

        print(f"✅ 최적화된 Detection System 초기화 완료")
        print(f"   - Detection frame skip: {detection_frame_skip}")
        print(f"   - Depth frame skip: {depth_frame_skip}")
        print(f"   - 비동기 처리: {use_async}")
        print(f"   - ROI 처리: {roi_enabled}")

    def _init_nanoowl(self):
        """NanoOWL 초기화"""
        try:
            sys.path.insert(0, str(Constants.Paths.NANOOWL_DIR))
            from nanoowl.owl_predictor import OwlPredictor

            model_name = 'google/owlvit-base-patch32'
            self.predictor = OwlPredictor(
                model_name,
                device=self.device,
                image_encoder_engine=None
            )

            # 미션별 쿼리 정의
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
                    'queries': ["a blue buoy"],
                    'label_mapping': {0: "blue_buoy"}
                },
                MissionType.DOCK_MODE: {
                    'queries': ["a red square"],
                    'label_mapping': {0: "red_square"}
                }
            }

            # 텍스트 인코딩 미리 수행
            for mission_type, query_info in self.detection_queries.items():
                query_info['text_encodings'] = self.predictor.encode_text(query_info['queries'])

            print("✅ NanoOWL 초기화 완료")
        except Exception as e:
            print(f"❌ NanoOWL 초기화 실패: {e}")
            self.use_nanoowl = False

    def _estimate_depth_async(self, image):
        """비동기 Depth 추정"""
        return self.depth_estimator.estimate_depth(image)

    def _apply_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """ROI 적용"""
        if not self.roi_enabled or self.roi_bounds is None:
            return image, (0, 0)

        x1, y1, x2, y2 = self.roi_bounds
        roi_image = image[y1:y2, x1:x2]
        return roi_image, (x1, y1)

    def detect_objects(self, image: np.ndarray, mission_type: MissionType) -> List[Dict]:
        """
        최적화된 객체 탐지

        Args:
            image: BGR 이미지
            mission_type: 현재 미션 타입

        Returns:
            탐지된 객체 리스트
        """
        if image is None:
            return []

        # 탐지가 필요 없는 미션은 스킵
        if mission_type not in [
            MissionType.PASS_BETWEEN_BUOYS,
            MissionType.CIRCLE_BUOY,
            MissionType.DOCK_MODE
        ]:
            return []

        # === 1. Depth 처리 (비동기 + Frame skip) ===
        self.depth_frame_count += 1

        if self.depth_frame_count % self.depth_frame_skip == 0:
            if self.use_async:
                # 비동기로 Depth 추정 제출
                self.depth_worker.submit(image.copy())
            else:
                # 동기로 Depth 추정
                self.last_depth_map = self.depth_estimator.estimate_depth(image)

        # 비동기 결과 확인
        if self.use_async:
            depth_result = self.depth_worker.get_result()
            if depth_result is not None:
                self.last_depth_map = depth_result

        depth_map = self.last_depth_map
        if depth_map is None:
            return []

        # === 2. Detection 처리 (Frame skip) ===
        self.detection_frame_count += 1

        if self.detection_frame_count % self.detection_frame_skip != 0:
            # 이전 결과 반환 (depth는 업데이트)
            return self._update_detections_depth(self.last_detections, depth_map)

        if not self.use_nanoowl:
            return []

        # ROI 적용
        process_image, roi_offset = self._apply_roi(image)

        # RGB 변환
        frame_rgb = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(frame_rgb)

        # 쿼리 선택
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

            # ROI 오프셋 적용
            x1 += roi_offset[0]
            x2 += roi_offset[0]
            y1 += roi_offset[1]
            y2 += roi_offset[1]

            # 박스 크기 필터링
            area = (x2 - x1) * (y2 - y1)
            if not (self.min_box_area <= area <= self.max_box_area):
                continue

            # 중심점 계산
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = max(0, min(depth_map.shape[1] - 1, cx))
            cy = max(0, min(depth_map.shape[0] - 1, cy))

            # Depth 추출
            if self.spatial_smoothing:
                depth = smooth_depth_spatially(
                    depth_map, cx, cy, kernel_size=self.spatial_kernel_size
                )
            else:
                depth = depth_map[cy, cx]

            # 깊이 필터링
            if not (self.min_depth_threshold <= depth <= self.max_depth_threshold):
                continue

            label = query_info['label_mapping'].get(label_idx, "unknown")

            detections.append({
                "label": label,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "depth": float(depth)
            })

        # 클래스별 최고 신뢰도만 선택
        detections = self._select_best_per_class(detections)

        # 캐싱
        self.last_detections = detections

        return detections

    def _update_detections_depth(self, detections: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        """이전 Detection 결과의 Depth만 업데이트"""
        if not detections or depth_map is None:
            return detections

        updated = []
        for det in detections:
            det_copy = det.copy()
            cx, cy = det['center']

            cx = max(0, min(depth_map.shape[1] - 1, cx))
            cy = max(0, min(depth_map.shape[0] - 1, cy))

            if self.spatial_smoothing:
                depth = smooth_depth_spatially(
                    depth_map, cx, cy, kernel_size=self.spatial_kernel_size
                )
            else:
                depth = depth_map[cy, cx]

            det_copy['depth'] = float(depth)
            updated.append(det_copy)

        return updated

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

    def update_parameters(
        self,
        detection_threshold=None,
        min_box_area=None,
        max_box_area=None,
        min_depth=None,
        max_depth=None,
        detection_frame_skip=None,
        depth_frame_skip=None
    ):
        """파라미터 업데이트"""
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
        if detection_frame_skip is not None:
            self.detection_frame_skip = detection_frame_skip
        if depth_frame_skip is not None:
            self.depth_frame_skip = depth_frame_skip

    def cleanup(self):
        """리소스 정리"""
        if self.use_async:
            self.depth_worker.stop()
            print("✅ 비동기 워커 종료")


# 하위 호환성
DetectionSystem = OptimizedDetectionSystem
