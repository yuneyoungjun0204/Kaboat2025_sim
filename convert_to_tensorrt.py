#!/usr/bin/env python3
"""
Depth 모델을 TensorRT FP16/INT8로 변환
Jetson Orin Nano에서 5-10배 속도 향상!
"""

import torch
import torch.onnx
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import argparse


def export_pytorch_to_onnx(model_type="DPT_Hybrid", input_size=256, onnx_path="depth_model.onnx"):
    """PyTorch 모델을 ONNX로 변환"""
    print(f"\n{'='*70}")
    print("📦 1단계: PyTorch → ONNX 변환")
    print(f"{'='*70}")

    # MiDaS 모델 로드
    print(f"  → {model_type} 모델 로딩...")
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    model.eval()
    model.cuda()

    # 더미 입력
    dummy_input = torch.randn(1, 3, input_size, input_size).cuda()

    # ONNX 변환
    print(f"  → ONNX로 변환 중 ({onnx_path})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Static shape for max performance
    )

    print(f"  ✅ ONNX 변환 완료: {onnx_path}")
    return onnx_path


class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    """간단한 INT8 캘리브레이터"""

    def __init__(self, input_size=256, cache_file="calibration.cache"):
        super().__init__()
        self.input_size = input_size
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0

        # 더미 캘리브레이션 데이터 (실제로는 실제 이미지 사용 권장)
        self.num_batches = 10
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * input_size * input_size * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index < self.num_batches:
            # 더미 데이터 생성 (실제로는 실제 이미지 사용)
            batch = np.random.randn(self.batch_size, 3, self.input_size, self.input_size).astype(np.float32)
            cuda.memcpy_htod(self.device_input, batch.ravel())
            self.current_index += 1
            return [int(self.device_input)]
        else:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_tensorrt_engine(onnx_path, engine_path, precision="fp16", batch_size=1, input_size=256):
    """ONNX 모델을 TensorRT 엔진으로 변환"""
    print(f"\n{'='*70}")
    print(f"🔥 2단계: ONNX → TensorRT {precision.upper()} 변환")
    print(f"{'='*70}")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # Builder 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 파싱
    print(f"  → ONNX 파일 파싱 중...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('  ❌ ONNX 파싱 실패:')
            for error in range(parser.num_errors):
                print(f"     {parser.get_error(error)}")
            return None

    print(f"  ✅ ONNX 파싱 완료")

    # Config 설정
    config = builder.create_builder_config()

    # 메모리 한도 (Jetson Orin Nano: 8GB, 안전하게 4GB 사용)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Precision 설정
    if precision == "fp16":
        print(f"  → FP16 모드 활성화 (2-3배 속도 향상)")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        print(f"  → INT8 모드 활성화 (4-8배 속도 향상)")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # INT8 fallback용

        # INT8 캘리브레이터 설정
        print(f"  → INT8 캘리브레이션 진행 중...")
        calibrator = SimpleCalibrator(input_size=input_size)
        config.int8_calibrator = calibrator
        print(f"  ✅ 캘리브레이터 설정 완료")

    # 엔진 빌드
    print(f"  → TensorRT 엔진 빌드 중 (5-15분 소요)...")
    print(f"     이 시간 동안 Jetson이 느려질 수 있습니다.")
    if precision == "int8":
        print(f"     INT8은 캘리브레이션으로 인해 더 오래 걸립니다.")

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("  ❌ 엔진 빌드 실패!")
        return None

    # 엔진 저장
    print(f"  → 엔진 저장 중: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"  ✅ TensorRT 엔진 생성 완료!")
    print(f"     파일 크기: {Path(engine_path).stat().st_size / (1024**2):.1f} MB")

    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Depth 모델 TensorRT 변환")
    parser.add_argument('--model', type=str, default='DPT_Hybrid',
                       choices=['DPT_Hybrid', 'MiDaS_small', 'DPT_Large'],
                       help='MiDaS 모델 타입')
    parser.add_argument('--input-size', type=int, default=256,
                       help='입력 이미지 크기 (256 권장)')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='정밀도 (fp16 권장)')
    parser.add_argument('--onnx-path', type=str, default='depth_model.onnx',
                       help='ONNX 파일 경로')
    parser.add_argument('--engine-path', type=str, default='depth_model_fp16.engine',
                       help='TensorRT 엔진 파일 경로')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 Depth 모델 TensorRT 변환 시작")
    print("="*70)
    print(f"  Model: {args.model}")
    print(f"  Input Size: {args.input_size}x{args.input_size}")
    print(f"  Precision: {args.precision.upper()}")
    print("="*70)

    # 1. PyTorch → ONNX
    onnx_path = export_pytorch_to_onnx(
        model_type=args.model,
        input_size=args.input_size,
        onnx_path=args.onnx_path
    )

    # 2. ONNX → TensorRT
    engine_path = build_tensorrt_engine(
        onnx_path=onnx_path,
        engine_path=args.engine_path,
        precision=args.precision,
        input_size=args.input_size
    )

    if engine_path:
        print(f"\n{'='*70}")
        print("✅ 변환 완료!")
        print(f"{'='*70}")
        print(f"  TensorRT 엔진: {engine_path}")
        print(f"\n💡 사용 방법:")
        print(f"  코드에서 다음과 같이 사용하세요:")
        print(f"")
        print(f"  estimator = create_ultra_depth_estimator(")
        print(f"      preset='balanced',")
        print(f"      use_tensorrt=True,")
        print(f"      engine_path='{engine_path}'")
        print(f"  )")
        print(f"{'='*70}\n")
    else:
        print("\n❌ 변환 실패!")


if __name__ == "__main__":
    main()
