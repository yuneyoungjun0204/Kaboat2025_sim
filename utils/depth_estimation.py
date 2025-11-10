"""
깊이 추정 모듈
- MiDaS Hybrid 모델을 사용한 실시간 깊이 맵 추정
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

class MiDaSHybridDepthEstimator:
    """MiDaS Hybrid 모델을 사용한 깊이 추정기 (캐싱 최적화)"""

    def __init__(self, cache_frames: int = 5):
        """
        Args:
            cache_frames: 깊이 맵 캐시 유지 프레임 수 (기본값: 5, 100Hz에서 20Hz로)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MiDaS Hybrid 모델을 {self.device}에서 실행합니다.")

        # MiDaS Hybrid 모델 로드
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Jetson 최적화: torch 추론 최적화
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # 자동 최적화
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 사용

        # Jetson 최적화: 해상도 축소 (384 -> 256)로 처리 속도 약 2배 향상
        self.transform_hybrid = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Jetson 최적화: 깊이 맵 캐싱 (GPU 사용량 80% 절감)
        self.cache_frames = cache_frames
        self.cached_depth_map = None
        self.cache_age = cache_frames  # 초기에는 즉시 계산하도록

        print(f"MiDaS Hybrid 모델 로드 완료! (캐시: {cache_frames} frames)")

    def estimate_depth(self, image):
        """
        이미지에서 깊이 맵 추정 (캐싱 최적화)

        Jetson 최적화:
        - 캐시 히트 시 GPU 추론 생략 (80% GPU 절감)
        - 기본 10프레임마다 갱신 (100Hz → 10Hz)
        """
        # 캐시 확인: 유효 기간 내면 캐시 반환
        if self.cache_age < self.cache_frames and self.cached_depth_map is not None:
            self.cache_age += 1
            return self.cached_depth_map

        try:
            # Jetson 최적화: 직접 RGB 변환 후 PIL로
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            # 이미지 전처리
            input_tensor = self.transform_hybrid(pil_image).to(self.device)
            input_batch = input_tensor.unsqueeze(0)

            # Jetson 최적화: torch.inference_mode() 사용 (no_grad보다 빠름)
            with torch.inference_mode():
                prediction = self.model(input_batch)
                # Jetson 최적화: bilinear 모드 사용 (bicubic보다 빠름)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            # 깊이 맵을 numpy 배열로 변환
            depth_map = prediction.cpu().numpy()

            # 깊이 맵 정규화 (0-1 범위)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            # 캐시 갱신
            self.cached_depth_map = depth_map
            self.cache_age = 0

            return depth_map

        except Exception as e:
            print(f"깊이 추정 오류: {e}")
            # 캐시가 있으면 반환, 없으면 None
            return self.cached_depth_map
    
    def get_depth_at_point(self, depth_map, x, y):
        """특정 좌표에서의 깊이 값 반환"""
        if depth_map is None:
            return None
        
        # 좌표가 이미지 범위 내에 있는지 확인
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return float(depth_map[y, x])
        return None
