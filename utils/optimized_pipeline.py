#!/usr/bin/env python3
"""
최적화된 통합 파이프라인 (Jetson Orin Nano)

전체 파이프라인 최적화:
- Detection (NanoOWL) + Depth (MiDaS) + Tracking (IMM-PDAF)
- 멀티스레딩 병렬 처리
- TensorRT FP16 최적화
- Frame skip 및 캐싱
- 성능 모니터링
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional
from collections import deque

from .detection_system_optimized import OptimizedDetectionSystem, MissionType
from .depth_estimation_optimized import OptimizedDepthEstimator
from .imm_pdaf_tracker import IMMPDAFTracker


class PerformanceMonitor:
    """성능 모니터링"""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.detection_times = deque(maxlen=window_size)
        self.depth_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)

    def record(self, detection_time, depth_time, tracking_time, total_time):
        """시간 기록"""
        self.detection_times.append(detection_time)
        self.depth_times.append(depth_time)
        self.tracking_times.append(tracking_time)
        self.total_times.append(total_time)

    def get_stats(self):
        """통계 반환"""
        if not self.total_times:
            return {}

        return {
            'detection_avg_ms': np.mean(self.detection_times) * 1000,
            'depth_avg_ms': np.mean(self.depth_times) * 1000,
            'tracking_avg_ms': np.mean(self.tracking_times) * 1000,
            'total_avg_ms': np.mean(self.total_times) * 1000,
            'fps': 1.0 / np.mean(self.total_times) if np.mean(self.total_times) > 0 else 0,
            'detection_max_ms': np.max(self.detection_times) * 1000,
            'depth_max_ms': np.max(self.depth_times) * 1000,
        }

    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        if stats:
            print(f"\n📊 성능 통계 (최근 {self.window_size}프레임)")
            print(f"   Detection: {stats['detection_avg_ms']:.1f}ms (max: {stats['detection_max_ms']:.1f}ms)")
            print(f"   Depth:     {stats['depth_avg_ms']:.1f}ms (max: {stats['depth_max_ms']:.1f}ms)")
            print(f"   Tracking:  {stats['tracking_avg_ms']:.1f}ms")
            print(f"   Total:     {stats['total_avg_ms']:.1f}ms")
            print(f"   FPS:       {stats['fps']:.1f}")


class OptimizedPipeline:
    """
    최적화된 통합 파이프라인

    구성:
    - OptimizedDetectionSystem (멀티스레딩 + Frame skip)
    - OptimizedDepthEstimator (TensorRT FP16)
    - IMMPDAFTracker (고정밀 추적)
    """

    def __init__(
        self,
        # Depth 설정
        depth_model_type="DPT_Hybrid",
        depth_input_size=256,
        use_tensorrt_depth=False,
        depth_engine_path=None,

        # Detection 설정
        detection_threshold=0.00065,
        detection_frame_skip=2,
        depth_frame_skip=3,
        use_async=True,
        use_nanoowl=True,

        # Tracking 설정
        fps=30.0,
        max_coast_frames=10,

        # 성능 모니터링
        enable_monitoring=True
    ):
        print("=" * 60)
        print("🚀 Jetson Orin Nano 최적화 파이프라인 초기화")
        print("=" * 60)

        # Depth Estimator
        print("\n[1/3] Depth Estimator 초기화...")
        self.depth_estimator = OptimizedDepthEstimator(
            model_type=depth_model_type,
            input_size=depth_input_size,
            use_tensorrt=use_tensorrt_depth,
            engine_path=depth_engine_path
        )

        # Detection System
        print("\n[2/3] Detection System 초기화...")
        self.detection_system = OptimizedDetectionSystem(
            depth_estimator=self.depth_estimator,
            detection_threshold=detection_threshold,
            detection_frame_skip=detection_frame_skip,
            depth_frame_skip=depth_frame_skip,
            use_async=use_async,
            use_nanoowl=use_nanoowl
        )

        # Tracker
        print("\n[3/3] IMM-PDAF Tracker 초기화...")
        self.tracker = IMMPDAFTracker(
            dt=1.0/fps,
            max_coast_frames=max_coast_frames
        )

        # 성능 모니터
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = PerformanceMonitor()

        print("\n" + "=" * 60)
        print("✅ 파이프라인 초기화 완료!")
        print("=" * 60)

    def process_frame(
        self,
        image: np.ndarray,
        mission_type: MissionType,
        return_timing=False
    ) -> Dict:
        """
        프레임 처리

        Args:
            image: BGR 이미지
            mission_type: 현재 미션
            return_timing: 타이밍 정보 반환 여부

        Returns:
            {
                'detections': [...],  # Raw detections
                'tracked_objects': [...],  # Tracked objects
                'timing': {...}  # 타이밍 정보 (옵션)
            }
        """
        t_start = time.time()

        # === Detection ===
        t_det_start = time.time()
        detections = self.detection_system.detect_objects(image, mission_type)
        t_det_end = time.time()
        detection_time = t_det_end - t_det_start

        # === Tracking ===
        t_track_start = time.time()
        self.tracker.predict_tracks()
        self.tracker.update_tracks(detections)
        self.tracker.prune_tracks()
        tracked_objects = self.tracker.get_tracked_objects()
        t_track_end = time.time()
        tracking_time = t_track_end - t_track_start

        t_end = time.time()
        total_time = t_end - t_start

        # 성능 모니터링
        if self.enable_monitoring:
            # Depth time은 비동기라서 정확하지 않을 수 있음
            depth_time = 0.0  # 비동기 처리
            self.monitor.record(detection_time, depth_time, tracking_time, total_time)

        result = {
            'detections': detections,
            'tracked_objects': tracked_objects,
        }

        if return_timing:
            result['timing'] = {
                'detection_ms': detection_time * 1000,
                'tracking_ms': tracking_time * 1000,
                'total_ms': total_time * 1000,
                'fps': 1.0 / total_time if total_time > 0 else 0
            }

        return result

    def get_performance_stats(self):
        """성능 통계 반환"""
        if self.enable_monitoring:
            return self.monitor.get_stats()
        return {}

    def print_performance_stats(self):
        """성능 통계 출력"""
        if self.enable_monitoring:
            self.monitor.print_stats()

    def visualize_results(
        self,
        image: np.ndarray,
        result: Dict,
        show_fps=True
    ) -> np.ndarray:
        """
        결과 시각화

        Args:
            image: 원본 이미지
            result: process_frame 결과
            show_fps: FPS 표시 여부

        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()

        # Tracked objects 그리기
        for obj in result.get('tracked_objects', []):
            bbox = obj.get('bbox', [0, 0, 0, 0])
            label = obj.get('label', 'unknown')
            confidence = obj.get('confidence', 0.0)
            depth = obj.get('depth', 0.0)
            track_id = obj.get('track_id', -1)

            x1, y1, x2, y2 = bbox

            # 색상 선택
            if 'red' in label:
                color = (0, 0, 255)
            elif 'green' in label:
                color = (0, 255, 0)
            elif 'blue' in label:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)

            # 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 텍스트
            text = f"ID{track_id} {label} {confidence:.2f} {depth:.1f}m"
            cv2.putText(vis_image, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS 표시
        if show_fps and 'timing' in result:
            fps = result['timing']['fps']
            total_ms = result['timing']['total_ms']
            text = f"FPS: {fps:.1f} ({total_ms:.1f}ms)"
            cv2.putText(vis_image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        return vis_image

    def export_depth_to_tensorrt(self, onnx_path: str = "depth_model.onnx",
                                  engine_path: str = "depth_model.engine"):
        """Depth 모델을 TensorRT로 변환"""
        print("\n🔥 Depth 모델 TensorRT 변환 시작...")
        print(f"   ONNX: {onnx_path}")
        print(f"   Engine: {engine_path}")

        # 1. ONNX 변환
        self.depth_estimator.export_to_onnx(onnx_path)

        # 2. TensorRT 변환
        self.depth_estimator.export_to_tensorrt(onnx_path, engine_path)

        print("\n✅ TensorRT 변환 완료!")
        print(f"   다음 실행 시 사용:")
        print(f"   pipeline = OptimizedPipeline(")
        print(f"       use_tensorrt_depth=True,")
        print(f"       depth_engine_path='{engine_path}'")
        print(f"   )")

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 리소스 정리 중...")
        self.detection_system.cleanup()
        self.tracker.reset()
        print("✅ 정리 완료")


def create_optimized_pipeline(
    preset="balanced",  # 'fast', 'balanced', 'quality'
    **kwargs
):
    """
    프리셋 기반 파이프라인 생성

    Presets:
    - 'fast': 최대 속도 (낮은 품질)
    - 'balanced': 균형 (권장)
    - 'quality': 최고 품질 (느림)
    """
    presets = {
        'fast': {
            'depth_input_size': 192,
            'detection_frame_skip': 3,
            'depth_frame_skip': 5,
            'detection_threshold': 0.00065,
        },
        'balanced': {
            'depth_input_size': 256,
            'detection_frame_skip': 2,
            'depth_frame_skip': 3,
            'detection_threshold': 0.00065,
        },
        'quality': {
            'depth_input_size': 384,
            'detection_frame_skip': 1,
            'depth_frame_skip': 2,
            'detection_threshold': 0.00065,
        }
    }

    config = presets.get(preset, presets['balanced'])
    config.update(kwargs)

    return OptimizedPipeline(**config)
