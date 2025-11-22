#!/usr/bin/env python3
"""
Ultra 최적화 벤치마크 스크립트
============================

기존 시스템 vs Ultra 최적화 시스템 성능 비교

측정 항목:
- 평균 FPS
- 평균 처리 시간
- 메모리 사용량
- CUDA 메모리 사용량
- 전체 처리 속도 향상률
"""

import sys
import time
import numpy as np
import cv2
import torch
import psutil
import argparse
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from utils.depth_estimation_optimized import OptimizedDepthEstimator
from utils.depth_estimation_ultra import UltraDepthEstimator, create_ultra_depth_estimator
from utils.jetson_optimizer import setup_jetson


class BenchmarkRunner:
    """벤치마크 실행기"""

    def __init__(self, num_frames=100, warmup_frames=10):
        self.num_frames = num_frames
        self.warmup_frames = warmup_frames

        # 테스트 이미지 생성
        self.test_images = self._generate_test_images()

        print(f"\n{'='*80}")
        print("🏁 Ultra 최적화 벤치마크")
        print(f"{'='*80}")
        print(f"  테스트 프레임: {num_frames}")
        print(f"  Warmup 프레임: {warmup_frames}")
        print(f"  이미지 해상도: 640x480")
        print(f"{'='*80}\n")

    def _generate_test_images(self, count=100):
        """테스트 이미지 생성"""
        print("📸 테스트 이미지 생성 중...")
        images = []
        for _ in range(count):
            # 랜덤 이미지 (실제 카메라 이미지와 유사)
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images.append(img)
        print(f"  ✓ {count}개 이미지 생성 완료\n")
        return images

    def benchmark_estimator(self, estimator, name: str):
        """Depth Estimator 벤치마크"""
        print(f"\n{'='*80}")
        print(f"🔍 벤치마크: {name}")
        print(f"{'='*80}\n")

        # 초기 메모리
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        if torch.cuda.is_available():
            cuda_mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        print(f"🔥 Warmup ({self.warmup_frames} frames)...")
        for i in range(self.warmup_frames):
            _ = estimator.estimate_depth(self.test_images[i % len(self.test_images)])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("  ✓ Warmup 완료\n")

        # 벤치마크
        print(f"⏱️  벤치마크 실행 ({self.num_frames} frames)...")
        times = []

        for i in range(self.num_frames):
            t_start = time.time()

            depth_map = estimator.estimate_depth(
                self.test_images[i % len(self.test_images)]
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.time()
            times.append((t_end - t_start) * 1000)  # ms

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.num_frames} frames")

        # 통계 계산
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        fps = 1000.0 / avg_time

        # 메모리 사용량
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        if torch.cuda.is_available():
            cuda_mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            cuda_mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            cuda_mem_used = cuda_mem_after - cuda_mem_before

        # 결과 출력
        print(f"\n{'='*80}")
        print(f"📊 {name} - 벤치마크 결과")
        print(f"{'='*80}")
        print(f"  평균 시간:    {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  최소 시간:    {min_time:.2f} ms")
        print(f"  최대 시간:    {max_time:.2f} ms")
        print(f"  P95 시간:     {p95_time:.2f} ms")
        print(f"  P99 시간:     {p99_time:.2f} ms")
        print(f"  평균 FPS:     {fps:.2f}")
        print(f"\n  메모리 사용:  {mem_used:.1f} MB")

        if torch.cuda.is_available():
            print(f"  CUDA 메모리:  {cuda_mem_used:.1f} MB")
            print(f"  CUDA Peak:    {cuda_mem_peak:.1f} MB")

        print(f"{'='*80}\n")

        return {
            'name': name,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'p95_time_ms': p95_time,
            'p99_time_ms': p99_time,
            'fps': fps,
            'mem_used_mb': mem_used,
            'cuda_mem_used_mb': cuda_mem_used if torch.cuda.is_available() else 0,
            'cuda_mem_peak_mb': cuda_mem_peak if torch.cuda.is_available() else 0,
        }

    def compare_results(self, baseline_result, optimized_result):
        """결과 비교"""
        print(f"\n{'='*80}")
        print("🏆 성능 비교 결과")
        print(f"{'='*80}\n")

        speedup = baseline_result['avg_time_ms'] / optimized_result['avg_time_ms']
        fps_improvement = (
            (optimized_result['fps'] - baseline_result['fps']) /
            baseline_result['fps'] * 100
        )

        print(f"  속도 향상:      {speedup:.2f}x")
        print(f"  FPS 향상:       {fps_improvement:.1f}%")
        print(f"")
        print(f"  Baseline FPS:   {baseline_result['fps']:.2f}")
        print(f"  Optimized FPS:  {optimized_result['fps']:.2f}")
        print(f"")
        print(f"  Baseline 평균:  {baseline_result['avg_time_ms']:.2f} ms")
        print(f"  Optimized 평균: {optimized_result['avg_time_ms']:.2f} ms")
        print(f"  시간 단축:      {baseline_result['avg_time_ms'] - optimized_result['avg_time_ms']:.2f} ms")

        print(f"\n{'='*80}\n")

        return {
            'speedup': speedup,
            'fps_improvement_pct': fps_improvement
        }


def main():
    parser = argparse.ArgumentParser(description='Ultra 최적화 벤치마크')
    parser.add_argument('--frames', type=int, default=100,
                       help='테스트 프레임 수')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup 프레임 수')
    parser.add_argument('--input-size', type=int, default=256,
                       help='입력 이미지 크기')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Baseline 벤치마크 스킵')

    args = parser.parse_args()

    # Jetson 최적화 적용
    if torch.cuda.is_available():
        print("🚀 Jetson 시스템 최적화 적용 중...\n")
        setup_jetson()

    # 벤치마크 러너 생성
    runner = BenchmarkRunner(num_frames=args.frames, warmup_frames=args.warmup)

    results = {}

    # === 1. Baseline (기존 최적화) ===
    if not args.skip_baseline:
        print("\n" + "="*80)
        print("📌 Baseline: 기존 최적화 시스템")
        print("="*80)

        baseline = OptimizedDepthEstimator(
            model_type="DPT_Hybrid",
            input_size=args.input_size,
            use_tensorrt=False,
            device="cuda"
        )

        results['baseline'] = runner.benchmark_estimator(baseline, "Baseline")

        # 메모리 정리
        del baseline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(2)

    # === 2. Ultra 최적화 (Balanced) ===
    print("\n" + "="*80)
    print("🚀 Ultra 최적화: Balanced Preset")
    print("="*80)

    ultra_balanced = create_ultra_depth_estimator(
        preset='balanced',
        input_size=args.input_size,
        enable_profiling=False
    )

    results['ultra_balanced'] = runner.benchmark_estimator(
        ultra_balanced, "Ultra (Balanced)"
    )

    # 메모리 정리
    del ultra_balanced
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    time.sleep(2)

    # === 3. Ultra 최적화 (Max Speed) ===
    print("\n" + "="*80)
    print("⚡ Ultra 최적화: Max Speed Preset")
    print("="*80)

    ultra_fast = create_ultra_depth_estimator(
        preset='fast',
        input_size=192,  # 더 작은 입력
        enable_profiling=False
    )

    results['ultra_fast'] = runner.benchmark_estimator(
        ultra_fast, "Ultra (Max Speed)"
    )

    # === 4. 결과 비교 ===
    if not args.skip_baseline:
        print("\n" + "="*80)
        print("📈 Baseline vs Ultra (Balanced)")
        print("="*80)
        runner.compare_results(results['baseline'], results['ultra_balanced'])

        print("\n" + "="*80)
        print("📈 Baseline vs Ultra (Max Speed)")
        print("="*80)
        runner.compare_results(results['baseline'], results['ultra_fast'])

    # === 5. 최종 요약 ===
    print("\n" + "="*80)
    print("📊 최종 요약")
    print("="*80)

    for name, result in results.items():
        print(f"\n{result['name']}:")
        print(f"  FPS:           {result['fps']:.2f}")
        print(f"  평균 시간:      {result['avg_time_ms']:.2f} ms")
        print(f"  메모리:        {result['mem_used_mb']:.1f} MB")

    print("\n" + "="*80 + "\n")

    # === 6. 권장 사항 ===
    print("💡 권장 사항:")
    print("")

    if results['ultra_balanced']['fps'] > 15:
        print("  ✅ Ultra (Balanced) 프리셋 사용 권장")
        print("     - 우수한 성능과 품질의 균형")
        print(f"     - {results['ultra_balanced']['fps']:.1f} FPS 달성")
    elif results['ultra_fast']['fps'] > 20:
        print("  ✅ Ultra (Max Speed) 프리셋 사용 권장")
        print("     - 최대 속도 우선")
        print(f"     - {results['ultra_fast']['fps']:.1f} FPS 달성")
    else:
        print("  ⚠️  추가 최적화 고려:")
        print("     - TensorRT INT8 변환")
        print("     - 입력 해상도 감소 (192 또는 128)")
        print("     - Frame skip 증가")

    print("")


if __name__ == "__main__":
    main()
