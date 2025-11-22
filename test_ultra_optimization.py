#!/usr/bin/env python3
"""
Ultra 최적화 빠른 테스트
======================

Ultra 최적화 시스템을 빠르게 테스트하는 스크립트
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.depth_estimation_ultra import create_ultra_depth_estimator
from utils.jetson_optimizer import setup_jetson


def main():
    print("\n" + "="*80)
    print("🚀 Ultra 최적화 시스템 빠른 테스트")
    print("="*80 + "\n")

    # Jetson 최적화
    if torch.cuda.is_available():
        setup_jetson()

    # Ultra Depth Estimator 생성
    print("📦 Ultra Depth Estimator 초기화 중...\n")

    estimator = create_ultra_depth_estimator(
        preset='balanced',  # 'fast', 'balanced', 'quality'
        enable_profiling=True
    )

    # 테스트 이미지 생성
    print("📸 테스트 이미지 생성 중...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("  ✓ 640x480 테스트 이미지 생성 완료\n")

    # Warmup
    print("🔥 Warmup (10 frames)...")
    for _ in range(10):
        _ = estimator.estimate_depth(test_image)
    print("  ✓ Warmup 완료\n")

    # 테스트
    print("⏱️  성능 테스트 (30 frames)...")
    for i in range(30):
        depth_map = estimator.estimate_depth(test_image)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/30 frames")

    print("  ✓ 테스트 완료\n")

    # 통계 출력
    estimator.print_performance_stats()

    # 시각화 (선택)
    print("\n💡 시각화를 보려면 실제 이미지로 테스트하세요:")
    print("   python test_ultra_optimization.py --image your_image.jpg")
    print("")

    # 권장 사항
    stats = estimator.get_performance_stats()
    fps = stats.get('fps', 0)

    print("="*80)
    print("💡 권장 사항:")
    print("="*80)

    if fps > 20:
        print("  ✅ 우수한 성능! 현재 설정 유지 권장")
        print(f"     - {fps:.1f} FPS 달성")
    elif fps > 15:
        print("  ✅ 양호한 성능! 실시간 처리 가능")
        print(f"     - {fps:.1f} FPS 달성")
    elif fps > 10:
        print("  ⚠️  보통 성능. 더 빠른 처리를 위해:")
        print("     - 'fast' 프리셋 사용")
        print("     - 입력 해상도 감소 (192)")
    else:
        print("  ⚠️  추가 최적화 필요:")
        print("     - 'fast' 프리셋 사용")
        print("     - Frame skip 증가")
        print("     - TensorRT INT8 변환 고려")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
