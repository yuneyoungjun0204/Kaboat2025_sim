#!/usr/bin/env python3
"""
TensorRT 엔진 변환 스크립트
Depth 모델을 ONNX → TensorRT FP16 엔진으로 변환

사용법:
    python3 convert_to_tensorrt.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.depth_estimation_optimized import OptimizedDepthEstimator

def main():
    print("=" * 70)
    print("🔥 TensorRT 엔진 변환 스크립트")
    print("=" * 70)
    
    # 설정
    input_size = 256
    onnx_path = "depth_model.onnx"
    engine_path = "depth_model.engine"
    
    print(f"\n📋 설정:")
    print(f"   - Input Size: {input_size}x{input_size}")
    print(f"   - ONNX 경로: {onnx_path}")
    print(f"   - Engine 경로: {engine_path}")
    print()
    
    # 1. Depth Estimator 생성
    print("📦 Depth Estimator 생성 중...")
    estimator = OptimizedDepthEstimator(
        model_type="DPT_Hybrid",
        input_size=input_size,
        use_tensorrt=False,  # PyTorch 모드로 변환
        device="cuda"
    )
    print("✅ Depth Estimator 생성 완료\n")
    
    # 2. ONNX 변환
    print("=" * 70)
    print("📤 1단계: ONNX 변환")
    print("=" * 70)
    try:
        estimator.export_to_onnx(onnx_path)
        print(f"✅ ONNX 변환 완료: {onnx_path}\n")
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        return
    
    # 3. TensorRT 엔진 생성 안내
    print("=" * 70)
    print("🔥 2단계: TensorRT 엔진 생성")
    print("=" * 70)
    print("\n다음 명령을 실행하세요:\n")
    print(f"trtexec --onnx={onnx_path} \\")
    print(f"        --saveEngine={engine_path} \\")
    print(f"        --fp16 \\")
    print(f"        --workspace=2048 \\")
    print(f"        --verbose")
    print("\n또는 Python으로 변환하려면:")
    print(f"  estimator.export_to_tensorrt('{onnx_path}', '{engine_path}')")
    print("\n" + "=" * 70)
    print("✅ 변환 준비 완료!")
    print("=" * 70)
    
    # Python TensorRT 변환 시도
    print("\n🐍 Python TensorRT 변환 시도 중...")
    try:
        estimator.export_to_tensorrt(onnx_path, engine_path)
        print(f"\n✅ TensorRT 엔진 생성 완료: {engine_path}")
        print(f"\n💡 이제 system_factory.py가 자동으로 이 엔진을 사용합니다!")
    except ImportError:
        print("⚠️ TensorRT Python API 미설치")
        print("   → trtexec 명령을 사용하세요 (위 명령 참고)")
    except Exception as e:
        print(f"⚠️ Python TensorRT 변환 실패: {e}")
        print("   → trtexec 명령을 사용하세요 (위 명령 참고)")

if __name__ == "__main__":
    main()



