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
    """MiDaS Hybrid 모델을 사용한 깊이 추정기"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MiDaS Hybrid 모델을 {self.device}에서 실행합니다.")
        
        # MiDaS Hybrid 모델 로드
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Hybrid 모델용 전처리 변환
        self.transform_hybrid = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("MiDaS Hybrid 모델 로드 완료!")
    
    def estimate_depth(self, image):
        """이미지에서 깊이 맵 추정"""
        try:
            # OpenCV 이미지를 PIL 이미지로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # 이미지 전처리
            input_tensor = self.transform_hybrid(pil_image).to(self.device)
            input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가
            
            # 깊이 추정
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # 깊이 맵을 numpy 배열로 변환
            depth_map = prediction.cpu().numpy()
            
            # 깊이 맵 정규화 (0-1 범위)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            return depth_map
            
        except Exception as e:
            print(f"깊이 추정 오류: {e}")
            return None
    
    def get_depth_at_point(self, depth_map, x, y):
        """특정 좌표에서의 깊이 값 반환"""
        if depth_map is None:
            return None
        
        # 좌표가 이미지 범위 내에 있는지 확인
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return float(depth_map[y, x])
        return None
