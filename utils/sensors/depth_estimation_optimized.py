"""
최적화된 깊이 추정 모듈 (Jetson Orin Nano)
- TensorRT FP16 최적화
- ONNX 변환 및 TensorRT 엔진 생성
- 입력 해상도 최적화 (256x256)
- FP16 precision for 2x speedup
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from pathlib import Path
import os


class OptimizedDepthEstimator:
    """
    TensorRT 최적화된 MiDaS Depth Estimator

    최적화 기법:
    1. TensorRT FP16 사용 (2배 속도 향상)
    2. 입력 해상도 감소 (256x256)
    3. torch.inference_mode() 사용
    4. CUDA 최적화 활성화
    5. 메모리 최적화
    """

    def __init__(
        self,
        model_type="DPT_Hybrid",  # DPT_Hybrid or MiDaS_small
        input_size=256,  # 256 for speed, 384 for quality
        use_tensorrt=False,  # TensorRT 사용 (변환 필요)
        engine_path=None,  # TensorRT engine 경로
        device="cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.use_tensorrt = use_tensorrt
        self.model_type = model_type

        print(f"🚀 Optimized Depth Estimator 초기화...")
        print(f"   - Device: {self.device}")
        print(f"   - Input size: {input_size}x{input_size}")
        print(f"   - TensorRT: {use_tensorrt}")

        if use_tensorrt and engine_path and Path(engine_path).exists():
            self._load_tensorrt_engine(engine_path)
        else:
            self._load_pytorch_model()

        print(f"✅ Depth Estimator 초기화 완료!")

    def _load_pytorch_model(self):
        """PyTorch 모델 로드 (TensorRT 변환 전)"""
        print(f"📦 PyTorch {self.model_type} 모델 로딩 중...")

        # MiDaS 모델 로드
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Jetson 최적화
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print("   ✓ CUDA 최적화 활성화")

        # 입력 전처리
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"   ✓ PyTorch 모델 로드 완료")

    def _load_tensorrt_engine(self, engine_path: str):
        """TensorRT 엔진 로드"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            print(f"🔥 TensorRT 엔진 로딩: {engine_path}")

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            with open(engine_path, "rb") as f:
                engine_data = f.read()

            runtime = trt.Runtime(TRT_LOGGER)
            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self.trt_context = self.trt_engine.create_execution_context()

            # 입출력 바인딩
            self.trt_inputs = []
            self.trt_outputs = []
            self.trt_bindings = []

            for i in range(self.trt_engine.num_bindings):
                binding_name = self.trt_engine.get_binding_name(i)
                size = trt.volume(self.trt_engine.get_binding_shape(i))
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))

                device_mem = cuda.mem_alloc(size * dtype.itemsize)
                self.trt_bindings.append(int(device_mem))

                if self.trt_engine.binding_is_input(i):
                    self.trt_inputs.append({'name': binding_name, 'mem': device_mem, 'size': size, 'dtype': dtype})
                else:
                    self.trt_outputs.append({'name': binding_name, 'mem': device_mem, 'size': size, 'dtype': dtype})

            # 입력 전처리
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.cuda_stream = cuda.Stream()
            print(f"   ✓ TensorRT 엔진 로드 완료")

        except ImportError:
            print("⚠️ TensorRT 미설치 - PyTorch 모드로 전환")
            self.use_tensorrt = False
            self._load_pytorch_model()
        except Exception as e:
            print(f"❌ TensorRT 로드 실패: {e}")
            print("   → PyTorch 모드로 전환")
            self.use_tensorrt = False
            self._load_pytorch_model()

    def export_to_onnx(self, onnx_path: str = "depth_model.onnx"):
        """PyTorch 모델을 ONNX로 변환"""
        if self.use_tensorrt:
            print("⚠️ 이미 TensorRT 모드입니다")
            return

        print(f"📤 ONNX 모델 변환 중: {onnx_path}")

        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"✅ ONNX 변환 완료: {onnx_path}")
        print(f"   → TensorRT 변환:")
        print(f"      trtexec --onnx={onnx_path} --saveEngine=depth_model.engine --fp16 --workspace=2048")

    def export_to_tensorrt(self, onnx_path: str, engine_path: str = "depth_model.engine"):
        """ONNX를 TensorRT 엔진으로 변환"""
        try:
            import tensorrt as trt

            print(f"🔥 TensorRT 엔진 생성 중...")

            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # ONNX 파�ing
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("ONNX 파싱 실패")

            # Builder 설정
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

            # FP16 최적화
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("   ✓ FP16 모드 활성화")

            # DLA 사용 (Jetson 전용)
            if builder.num_DLA_cores > 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = 0
                print(f"   ✓ DLA 코어 0 사용")

            # 엔진 빌드
            print("   ⏳ 엔진 빌드 중... (시간이 걸릴 수 있습니다)")
            serialized_engine = builder.build_serialized_network(network, config)

            # 저장
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            print(f"✅ TensorRT 엔진 생성 완료: {engine_path}")

        except ImportError:
            print("❌ TensorRT가 설치되지 않았습니다")
            print("   설치: pip install tensorrt")
        except Exception as e:
            print(f"❌ TensorRT 변환 실패: {e}")

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """깊이 추정 (최적화)"""
        try:
            if self.use_tensorrt:
                return self._estimate_depth_tensorrt(image)
            else:
                return self._estimate_depth_pytorch(image)

        except Exception as e:
            print(f"❌ 깊이 추정 오류: {e}")
            return None

    def _estimate_depth_pytorch(self, image: np.ndarray) -> np.ndarray:
        """PyTorch로 깊이 추정"""
        # RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # 전처리
        input_tensor = self.transform(pil_image).to(self.device)
        input_batch = input_tensor.unsqueeze(0)

        # 추론 (최적화 모드)
        with torch.inference_mode():
            prediction = self.model(input_batch)

            # Bilinear interpolation (빠름)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        # numpy 변환
        depth_map = prediction.cpu().numpy()

        # 정규화 (0-1)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_map

    def _estimate_depth_tensorrt(self, image: np.ndarray) -> np.ndarray:
        """TensorRT로 깊이 추정"""
        import pycuda.driver as cuda

        # 전처리
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).cpu().numpy()

        # Host to Device
        cuda.memcpy_htod_async(self.trt_inputs[0]['mem'], input_tensor, self.cuda_stream)

        # 추론
        self.trt_context.execute_async_v2(bindings=self.trt_bindings, stream_handle=self.cuda_stream.handle)

        # Device to Host
        output = np.empty(self.trt_outputs[0]['size'], dtype=self.trt_outputs[0]['dtype'])
        cuda.memcpy_dtoh_async(output, self.trt_outputs[0]['mem'], self.cuda_stream)
        self.cuda_stream.synchronize()

        # 후처리
        depth_map = output.reshape(self.input_size, self.input_size)
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 정규화
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_map

    def get_depth_at_point(self, depth_map, x, y):
        """특정 좌표의 깊이 값 반환"""
        if depth_map is None:
            return None

        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return float(depth_map[y, x])
        return None


# 하위 호환성
MiDaSHybridDepthEstimator = OptimizedDepthEstimator
