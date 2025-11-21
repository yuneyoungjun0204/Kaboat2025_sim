#!/usr/bin/env python3
"""
성능 벤치마크 스크립트

기존 시스템 vs 최적화 시스템 비교:
- Detection 속도
- Depth estimation 속도
- 전체 파이프라인 FPS
- 메모리 사용량
"""

import cv2
import numpy as np
import time
import argparse
from typing import Dict, List
import sys
sys.path.append('/home/yuneyoungjun/vrx_ws/src/vrx/kaboat_backup')

# 기존 시스템
from utils.detection_system_optimized import DetectionSystem, MissionType as OldMissionType
from utils.depth_estimation_optimized import MiDaSHybridDepthEstimator
from utils.imm_pdaf_tracker import IMMPDAFTracker

# 최적화 시스템
from utils.optimized_pipeline import OptimizedPipeline, create_optimized_pipeline, MissionType


def generate_test_image(width=640, height=480):
    """테스트 이미지 생성"""
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return image


def benchmark_old_system(num_frames=100):
    """기존 시스템 벤치마크"""
    print("\n" + "=" * 60)
    print("📊 기존 시스템 벤치마크")
    print("=" * 60)

    # 초기화
    print("\n초기화 중...")
    depth_estimator = MiDaSHybridDepthEstimator()
    detection_system = DetectionSystem(depth_estimator)
    tracker = IMMPDAFTracker(dt=1/30.0)

    # 테스트 이미지
    test_image = generate_test_image()

    # 워밍업
    print("워밍업 중... (10 프레임)")
    for _ in range(10):
        detections = detection_system.detect_objects(test_image, OldMissionType.PASS_BETWEEN_BUOYS)
        tracker.predict_tracks()
        tracker.update_tracks(detections)

    # 벤치마크
    print(f"\n벤치마크 시작... ({num_frames} 프레임)")
    times = {
        'detection': [],
        'tracking': [],
        'total': []
    }

    for i in range(num_frames):
        t_start = time.time()

        # Detection
        t_det_start = time.time()
        detections = detection_system.detect_objects(test_image, OldMissionType.PASS_BETWEEN_BUOYS)
        t_det_end = time.time()

        # Tracking
        t_track_start = time.time()
        tracker.predict_tracks()
        tracker.update_tracks(detections)
        tracked = tracker.get_tracked_objects()
        t_track_end = time.time()

        t_end = time.time()

        times['detection'].append(t_det_end - t_det_start)
        times['tracking'].append(t_track_end - t_track_start)
        times['total'].append(t_end - t_start)

        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{num_frames} 프레임")

    # 통계
    stats = {
        'detection_avg_ms': np.mean(times['detection']) * 1000,
        'detection_max_ms': np.max(times['detection']) * 1000,
        'detection_min_ms': np.min(times['detection']) * 1000,
        'tracking_avg_ms': np.mean(times['tracking']) * 1000,
        'total_avg_ms': np.mean(times['total']) * 1000,
        'total_max_ms': np.max(times['total']) * 1000,
        'fps': 1.0 / np.mean(times['total'])
    }

    print("\n" + "=" * 60)
    print("📈 기존 시스템 결과")
    print("=" * 60)
    print(f"Detection:")
    print(f"  평균: {stats['detection_avg_ms']:.1f}ms")
    print(f"  최소: {stats['detection_min_ms']:.1f}ms")
    print(f"  최대: {stats['detection_max_ms']:.1f}ms")
    print(f"\nTracking:")
    print(f"  평균: {stats['tracking_avg_ms']:.1f}ms")
    print(f"\nTotal:")
    print(f"  평균: {stats['total_avg_ms']:.1f}ms")
    print(f"  최대: {stats['total_max_ms']:.1f}ms")
    print(f"  FPS: {stats['fps']:.1f}")

    return stats


def benchmark_optimized_system(num_frames=100, preset='balanced', use_tensorrt=False):
    """최적화 시스템 벤치마크"""
    print("\n" + "=" * 60)
    print(f"📊 최적화 시스템 벤치마크 (preset: {preset})")
    print("=" * 60)

    # 초기화
    print("\n초기화 중...")
    pipeline = create_optimized_pipeline(
        preset=preset,
        use_tensorrt_depth=use_tensorrt,
        enable_monitoring=True
    )

    # 테스트 이미지
    test_image = generate_test_image()

    # 워밍업
    print("워밍업 중... (10 프레임)")
    for _ in range(10):
        result = pipeline.process_frame(test_image, MissionType.PASS_BETWEEN_BUOYS)

    # 벤치마크
    print(f"\n벤치마크 시작... ({num_frames} 프레임)")
    times = {
        'detection': [],
        'tracking': [],
        'total': []
    }

    for i in range(num_frames):
        result = pipeline.process_frame(
            test_image,
            MissionType.PASS_BETWEEN_BUOYS,
            return_timing=True
        )

        if 'timing' in result:
            times['detection'].append(result['timing']['detection_ms'] / 1000)
            times['tracking'].append(result['timing']['tracking_ms'] / 1000)
            times['total'].append(result['timing']['total_ms'] / 1000)

        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{num_frames} 프레임")

    # 통계
    stats = {
        'detection_avg_ms': np.mean(times['detection']) * 1000,
        'detection_max_ms': np.max(times['detection']) * 1000,
        'detection_min_ms': np.min(times['detection']) * 1000,
        'tracking_avg_ms': np.mean(times['tracking']) * 1000,
        'total_avg_ms': np.mean(times['total']) * 1000,
        'total_max_ms': np.max(times['total']) * 1000,
        'fps': 1.0 / np.mean(times['total'])
    }

    print("\n" + "=" * 60)
    print(f"📈 최적화 시스템 결과 (preset: {preset})")
    print("=" * 60)
    print(f"Detection:")
    print(f"  평균: {stats['detection_avg_ms']:.1f}ms")
    print(f"  최소: {stats['detection_min_ms']:.1f}ms")
    print(f"  최대: {stats['detection_max_ms']:.1f}ms")
    print(f"\nTracking:")
    print(f"  평균: {stats['tracking_avg_ms']:.1f}ms")
    print(f"\nTotal:")
    print(f"  평균: {stats['total_avg_ms']:.1f}ms")
    print(f"  최대: {stats['total_max_ms']:.1f}ms")
    print(f"  FPS: {stats['fps']:.1f}")

    pipeline.cleanup()

    return stats


def compare_results(old_stats: Dict, new_stats: Dict):
    """결과 비교"""
    print("\n" + "=" * 60)
    print("📊 성능 비교")
    print("=" * 60)

    metrics = [
        ('Detection 평균', 'detection_avg_ms', 'ms'),
        ('Detection 최대', 'detection_max_ms', 'ms'),
        ('Tracking 평균', 'tracking_avg_ms', 'ms'),
        ('Total 평균', 'total_avg_ms', 'ms'),
        ('FPS', 'fps', '')
    ]

    for name, key, unit in metrics:
        old_val = old_stats[key]
        new_val = new_stats[key]

        if key == 'fps':
            # FPS는 높을수록 좋음
            improvement = ((new_val - old_val) / old_val) * 100
            symbol = "📈" if improvement > 0 else "📉"
        else:
            # 시간은 낮을수록 좋음
            improvement = ((old_val - new_val) / old_val) * 100
            symbol = "📈" if improvement > 0 else "📉"

        print(f"\n{name}:")
        print(f"  기존:   {old_val:.1f}{unit}")
        print(f"  최적화: {new_val:.1f}{unit}")
        print(f"  개선:   {symbol} {improvement:+.1f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='성능 벤치마크')
    parser.add_argument('--frames', type=int, default=100, help='테스트 프레임 수')
    parser.add_argument('--preset', type=str, default='balanced',
                       choices=['fast', 'balanced', 'quality'],
                       help='최적화 프리셋')
    parser.add_argument('--tensorrt', action='store_true',
                       help='TensorRT 사용 (변환 필요)')
    parser.add_argument('--skip-old', action='store_true',
                       help='기존 시스템 벤치마크 스킵')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 Jetson Orin Nano 성능 벤치마크")
    print("=" * 60)
    print(f"프레임 수: {args.frames}")
    print(f"프리셋: {args.preset}")
    print(f"TensorRT: {args.tensorrt}")

    # 기존 시스템
    old_stats = None
    if not args.skip_old:
        try:
            old_stats = benchmark_old_system(args.frames)
        except Exception as e:
            print(f"\n❌ 기존 시스템 벤치마크 실패: {e}")

    # 최적화 시스템
    new_stats = None
    try:
        new_stats = benchmark_optimized_system(
            args.frames,
            preset=args.preset,
            use_tensorrt=args.tensorrt
        )
    except Exception as e:
        print(f"\n❌ 최적화 시스템 벤치마크 실패: {e}")
        import traceback
        traceback.print_exc()

    # 비교
    if old_stats and new_stats:
        compare_results(old_stats, new_stats)

    print("\n✅ 벤치마크 완료!\n")


if __name__ == "__main__":
    main()
