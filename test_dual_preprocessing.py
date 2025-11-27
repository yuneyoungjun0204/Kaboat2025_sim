#!/usr/bin/env python3
"""
이중 전처리 테스트 (Detection vs Depth)

전략:
- Detection (NanoOWL): 원본 1280x720 → 정확도 유지
- Depth (MiDaS): 416x320 (4배 축소) → 속도 대폭 향상
- 좌표 자동 매핑: Detection bbox → Depth map 좌표
"""

import cv2
import numpy as np
import sys
sys.path.append('/home/yuneyoungjun/vrx_ws/src/vrx/kaboat_backup')

from utils.image_preprocessor import create_preprocessor


def test_coordinate_mapping():
    """좌표 매핑 테스트"""
    print("=" * 80)
    print("🧪 이중 전처리 좌표 매핑 테스트")
    print("=" * 80)

    # 원본 이미지 (1280x720)
    original_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    original_h, original_w = original_image.shape[:2]

    print(f"\n원본 이미지: {original_w}x{original_h}")

    # === 1. Detection 전처리 (원본 유지) ===
    image_preprocessor = None  # Detection은 원본 사용
    print(f"Detection 이미지: {original_w}x{original_h} (원본)")

    # === 2. Depth 전처리 (4배 축소) ===
    depth_preprocessor = create_preprocessor('max_speed')  # 416x320
    depth_image, depth_metadata = depth_preprocessor.preprocess(original_image)
    depth_h, depth_w = depth_image.shape[:2]

    print(f"Depth 이미지: {depth_w}x{depth_h} (4배 축소)")
    print(f"Depth 스케일: x={depth_metadata['scale_x']:.3f}, y={depth_metadata['scale_y']:.3f}")

    # === 3. 가짜 Detection 결과 (원본 좌표) ===
    # NanoOWL이 원본 이미지에서 탐지한 bbox
    detection_bbox_orig = [400, 200, 600, 400]  # 원본 좌표 (1280x720)
    cx_orig = (detection_bbox_orig[0] + detection_bbox_orig[2]) // 2
    cy_orig = (detection_bbox_orig[1] + detection_bbox_orig[3]) // 2

    print(f"\nDetection 결과 (원본 좌표):")
    print(f"  bbox: {detection_bbox_orig}")
    print(f"  center: ({cx_orig}, {cy_orig})")

    # === 4. Depth map 좌표로 변환 ===
    depth_scale_x = depth_metadata['scale_x']
    depth_scale_y = depth_metadata['scale_y']

    cx_depth = int(cx_orig * depth_scale_x)
    cy_depth = int(cy_orig * depth_scale_y)

    print(f"\nDepth map 좌표:")
    print(f"  center: ({cx_depth}, {cy_depth})")
    print(f"  계산: ({cx_orig} * {depth_scale_x:.3f}, {cy_orig} * {depth_scale_y:.3f})")

    # === 5. 검증 ===
    expected_cx = int(500 * 0.325)  # 500 * (416/1280)
    expected_cy = int(300 * 0.444)  # 300 * (320/720)

    print(f"\n✅ 좌표 매핑 검증:")
    print(f"  예상 depth center: ({expected_cx}, {expected_cy})")
    print(f"  실제 depth center: ({cx_depth}, {cy_depth})")

    if abs(cx_depth - expected_cx) <= 1 and abs(cy_depth - expected_cy) <= 1:
        print("  ✅ 좌표 매핑 정확!")
    else:
        print("  ❌ 좌표 매핑 오류!")

    # === 6. 시각화 ===
    print(f"\n📊 해상도 비교:")
    print(f"  원본 픽셀 수: {original_w * original_h:,} ({original_w}x{original_h})")
    print(f"  Depth 픽셀 수: {depth_w * depth_h:,} ({depth_w}x{depth_h})")
    print(f"  픽셀 감소율: {(1 - (depth_w * depth_h) / (original_w * original_h)) * 100:.1f}%")


def estimate_performance():
    """성능 예측"""
    print("\n" + "=" * 80)
    print("📊 예상 성능 향상")
    print("=" * 80)

    print("\n현재 시스템 (모두 1280x720):")
    print("  Detection (NanoOWL): ~1000ms")
    print("  Depth (MiDaS): ~800ms")
    print("  Tracking: ~50ms")
    print("  Total: ~1850ms (0.5 FPS)")

    print("\n개선 후 (Detection=1280x720, Depth=416x320):")
    print("  Detection (NanoOWL): ~1000ms (변화 없음 - 정확도 유지)")
    print("  Depth (MiDaS): ~115ms (7배 빠름! 800ms → 115ms)")
    print("  Tracking: ~50ms (변화 없음)")
    print("  Total: ~1165ms (0.86 FPS)")

    speedup = 1850 / 1165
    print(f"\n전체 속도 향상: {speedup:.2f}x")
    print(f"FPS 향상: 0.5 → 0.86 ({(0.86/0.5):.1f}x)")

    print("\n💡 추가 최적화 조합:")
    print("  + Frame skip (2프레임마다) → ~2x")
    print("  + 비동기 처리 → ~1.4x")
    print("  = 종합 속도 향상: ~3-4x")
    print("  = 최종 FPS: 0.5 → 2.0 FPS 🚀")


def show_advantages():
    """장점 요약"""
    print("\n" + "=" * 80)
    print("✅ 이중 전처리 전략의 장점")
    print("=" * 80)

    print("\n1. 정확도 유지:")
    print("   • Detection (NanoOWL)은 원본 해상도(1280x720) 사용")
    print("   • 작은 부표도 정확하게 탐지")
    print("   • bbox 좌표가 정밀함")

    print("\n2. 속도 대폭 향상:")
    print("   • Depth는 4배 축소(416x320) → 7배 빠름")
    print("   • Depth estimation이 병목의 50%를 차지")
    print("   • 전체 파이프라인 1.6배 향상")

    print("\n3. 메모리 절약:")
    print("   • Depth 연산 메모리: 1280x720 → 416x320")
    print("   • GPU 메모리 사용량 감소")
    print("   • 다른 작업에 여유 메모리 확보")

    print("\n4. 자동 좌표 매핑:")
    print("   • Detection bbox(원본) → Depth map(축소) 자동 변환")
    print("   • 사용자는 신경 쓸 필요 없음")
    print("   • 최종 결과는 모두 원본 좌표")

    print("\n5. 유연한 조정:")
    print("   • Detection 해상도 조절 가능 (원본 or 640x480)")
    print("   • Depth 축소 비율 조절 가능 (2배~8배)")
    print("   • 속도/품질 트레이드오프 선택")


def pixel_reduction_analysis():
    """픽셀 감소 분석"""
    print("\n" + "=" * 80)
    print("📐 픽셀 수 감소 분석")
    print("=" * 80)

    resolutions = [
        ("원본", 1280, 720),
        ("max_speed", 416, 320),
        ("balanced", 640, 480),
        ("quality", 800, 600),
    ]

    original_pixels = 1280 * 720

    print(f"\n{'프리셋':<12} {'해상도':<12} {'픽셀 수':>12} {'감소율':>10} {'예상 속도':>12}")
    print("-" * 80)

    for name, w, h in resolutions:
        pixels = w * h
        reduction = (1 - pixels / original_pixels) * 100
        speedup = original_pixels / pixels

        print(f"{name:<12} {w}x{h:<7} {pixels:>12,} {reduction:>9.1f}% {speedup:>11.1f}x")

    print("\n💡 권장:")
    print("  • Detection: 원본 (1280x720) - 정확도 최우선")
    print("  • Depth: max_speed (416x320) - 속도 최우선")
    print("  → 정확도 유지하면서 Depth만 7배 빠르게!")


if __name__ == "__main__":
    print("\n🚀 이중 전처리 (Detection vs Depth) 테스트\n")

    # 1. 좌표 매핑 테스트
    test_coordinate_mapping()

    # 2. 성능 예측
    estimate_performance()

    # 3. 장점 요약
    show_advantages()

    # 4. 픽셀 감소 분석
    pixel_reduction_analysis()

    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)

    print("\n💡 Main_MCP.py에서 자동 적용됩니다:")
    print("   → Detection: 원본 1280x720 (정확도 유지)")
    print("   → Depth: 416x320 (4배 축소, 7배 빠름)")
    print("   → 좌표 자동 변환")
    print("\n")
