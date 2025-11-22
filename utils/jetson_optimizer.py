#!/usr/bin/env python3
"""
Jetson Orin Nano 런타임 최적화
- CUDA 최적화
- PyTorch 최적화
- 메모리 최적화
- Jetson 파워 모드 설정 (nvpmodel, jetson_clocks)
"""

import os
import torch
import gc
import subprocess
import shutil


class JetsonOptimizer:
    """Jetson Orin Nano 런타임 최적화"""

    @staticmethod
    def setup_cuda_optimizations():
        """CUDA 최적화 설정"""
        print("🚀 CUDA 최적화 설정 중...")

        # CUDA 환경 변수 설정
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 비동기 실행
        os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache'
        os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB

        # cuDNN 최적화
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True  # 최적 알고리즘 자동 선택
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False  # 속도 우선
            print("  ✅ cuDNN 벤치마크 활성화")

        # TF32 활성화 (Ampere 아키텍처)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ✅ TF32 활성화 (Ampere)")

        # CUDA 캐시 디렉토리 생성
        os.makedirs('/tmp/cuda_cache', exist_ok=True)

        print("  ✅ CUDA 최적화 완료")

    @staticmethod
    def setup_pytorch_optimizations():
        """PyTorch 최적화 설정"""
        print("🔧 PyTorch 최적화 설정 중...")

        # 추론 모드 (Autograd 비활성화)
        torch.set_grad_enabled(False)
        print("  ✅ Gradient 비활성화 (추론 모드)")

        # JIT 최적화
        torch.jit.enable_onednn_fusion(True)
        print("  ✅ JIT OneDNN fusion 활성화")

        # 메모리 할당자 최적화
        if torch.cuda.is_available():
            # 메모리 할당 전략 설정
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            print("  ✅ CUDA 메모리 할당 최적화")

        print("  ✅ PyTorch 최적화 완료")

    @staticmethod
    def optimize_memory():
        """메모리 최적화"""
        print("💾 메모리 최적화 중...")

        # Python garbage collection
        gc.collect()

        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  ✅ CUDA 캐시 정리 완료")

        print("  ✅ 메모리 최적화 완료")

    @staticmethod
    def setup_power_mode():
        """Jetson 파워 모드 설정 (MAXN 모드)"""
        print("⚡ Jetson 파워 모드 설정 중...")
        
        # nvpmodel 명령어 확인
        nvpmodel_path = shutil.which('nvpmodel')
        jetson_clocks_path = shutil.which('jetson_clocks')
        
        if not nvpmodel_path:
            print("  ⚠️ nvpmodel을 찾을 수 없습니다. 수동으로 실행하세요: sudo nvpmodel -m 0")
        else:
            try:
                # 현재 모드 확인
                result = subprocess.run(
                    ['nvpmodel', '-q'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"  📊 현재 전력 모드:\n{result.stdout}")
                
                # MAXN 모드 설정 (모드 0)
                # sudo 권한이 필요할 수 있으므로 시도
                result = subprocess.run(
                    ['sudo', 'nvpmodel', '-m', '0'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print("  ✅ MAXN 모드 (최대 성능) 활성화 완료")
                else:
                    print(f"  ⚠️ nvpmodel 설정 실패 (sudo 권한 필요할 수 있음): {result.stderr}")
                    print("  💡 수동 실행: sudo nvpmodel -m 0")
            except subprocess.TimeoutExpired:
                print("  ⚠️ nvpmodel 명령 타임아웃")
            except Exception as e:
                print(f"  ⚠️ nvpmodel 실행 오류: {e}")
                print("  💡 수동 실행: sudo nvpmodel -m 0")
        
        if not jetson_clocks_path:
            print("  ⚠️ jetson_clocks를 찾을 수 없습니다. 수동으로 실행하세요: sudo jetson_clocks")
        else:
            try:
                # jetson_clocks 실행 (최대 클럭 설정)
                result = subprocess.run(
                    ['sudo', 'jetson_clocks'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print("  ✅ Jetson clocks (최대 클럭) 활성화 완료")
                else:
                    print(f"  ⚠️ jetson_clocks 실행 실패 (sudo 권한 필요할 수 있음): {result.stderr}")
                    print("  💡 수동 실행: sudo jetson_clocks")
            except subprocess.TimeoutExpired:
                print("  ⚠️ jetson_clocks 명령 타임아웃")
            except Exception as e:
                print(f"  ⚠️ jetson_clocks 실행 오류: {e}")
                print("  💡 수동 실행: sudo jetson_clocks")
        
        print("  ✅ 파워 모드 설정 완료")

    @staticmethod
    def print_device_info():
        """디바이스 정보 출력"""
        print("\n" + "=" * 60)
        print("📊 Jetson 디바이스 정보")
        print("=" * 60)

        if torch.cuda.is_available():
            print(f"CUDA Available: True")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Device Count: {torch.cuda.device_count()}")

            # 메모리 정보
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3

            print(f"Total Memory: {total_memory:.2f} GB")
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Cached: {cached:.2f} GB")
            print(f"Free: {total_memory - cached:.2f} GB")

            # cuDNN 정보
            if torch.backends.cudnn.is_available():
                print(f"cuDNN Available: True")
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
                print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        else:
            print("CUDA Available: False (CPU 모드)")

        print("=" * 60 + "\n")

    @staticmethod
    def apply_all_optimizations():
        """모든 최적화 적용"""
        print("\n" + "=" * 60)
        print("🚀 Jetson Orin Nano 최적화 시작")
        print("=" * 60 + "\n")

        # 파워 모드 설정 (먼저 실행)
        JetsonOptimizer.setup_power_mode()
        print()
        JetsonOptimizer.setup_cuda_optimizations()
        print()
        JetsonOptimizer.setup_pytorch_optimizations()
        print()
        JetsonOptimizer.optimize_memory()
        print()
        JetsonOptimizer.print_device_info()

        print("=" * 60)
        print("✅ Jetson 최적화 완료!")
        print("=" * 60 + "\n")


def setup_jetson():
    """Jetson 최적화 설정 (간편 함수)"""
    JetsonOptimizer.apply_all_optimizations()


if __name__ == "__main__":
    # 테스트
    setup_jetson()
