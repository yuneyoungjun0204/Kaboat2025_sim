#!/usr/bin/env python3
"""
고급 Blob Detector 모듈
- HSV 색상 필터링
- Adaptive Threshold
- 깊이 기반 필터링
- 파란색 부표 전용 감지
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

class AdvancedBlobDetector:
    def __init__(self):
        """고급 Blob Detector 초기화"""
        self.blob_detector = cv2.SimpleBlobDetector_create()
        
        # 파란색 HSV 범위 (기본값)
        self.blue_hsv_ranges = {
            'lower': np.array([100, 50, 50]),   # 파란색 하한
            'upper': np.array([130, 255, 255])  # 파란색 상한
        }
        
        # Blob 파라미터 (기본값)
        self.blob_params = {
            'min_area': 100,
            'max_area': 50000,
            'min_circularity': 0.3,
            'min_convexity': 0.5,
            'min_inertia_ratio': 0.1
        }
        
        # 깊이 필터링 파라미터
        self.depth_filter = {
            'min_depth': 0.5,  # 0.5m
            'max_depth': 15.0  # 15m
        }
    
    def update_blob_parameters(self, **kwargs):
        """Blob 감지 파라미터 업데이트"""
        self.blob_params.update(kwargs)
        
        # SimpleBlobDetector 파라미터 설정
        params = cv2.SimpleBlobDetector_Params()
        
        # 면적 필터링
        params.filterByArea = True
        params.minArea = self.blob_params['min_area']
        params.maxArea = self.blob_params['max_area']
        
        # 원형도 필터링
        params.filterByCircularity = True
        params.minCircularity = self.blob_params['min_circularity']
        
        # 볼록성 필터링
        params.filterByConvexity = True
        params.minConvexity = self.blob_params['min_convexity']
        
        # 관성비 필터링
        params.filterByInertia = True
        params.minInertiaRatio = self.blob_params['min_inertia_ratio']
        
        # 색상 필터링 (흰색 객체만)
        params.filterByColor = True
        params.blobColor = 255
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def update_blue_hsv_range(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray):
        """파란색 HSV 범위 업데이트"""
        self.blue_hsv_ranges['lower'] = lower_hsv
        self.blue_hsv_ranges['upper'] = upper_hsv
    
    def update_depth_filter(self, min_depth: float, max_depth: float):
        """깊이 필터링 범위 업데이트"""
        self.depth_filter['min_depth'] = min_depth
        self.depth_filter['max_depth'] = max_depth
    
    def create_blue_mask(self, image: np.ndarray) -> np.ndarray:
        """파란색 마스크 생성"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 파란색 범위 마스크
        blue_mask = cv2.inRange(hsv_image, 
                               self.blue_hsv_ranges['lower'], 
                               self.blue_hsv_ranges['upper'])
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        return blue_mask
    
    def create_adaptive_mask(self, image: np.ndarray, 
                           block_size: int = 11, c_param: int = 2) -> np.ndarray:
        """Adaptive Threshold 마스크 생성"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Threshold
        binary_mask = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_param
        )
        
        return binary_mask
    
    def detect_blobs(self, mask: np.ndarray) -> List[cv2.KeyPoint]:
        """마스크에서 blob 감지"""
        keypoints = self.blob_detector.detect(mask)
        return keypoints
    
    def get_depth_at_point(self, depth_map: Optional[np.ndarray], 
                          x: int, y: int) -> Optional[float]:
        """특정 좌표에서의 깊이 값 반환"""
        if depth_map is None:
            return None
            
        h, w = depth_map.shape
        if 0 <= x < w and 0 <= y < h:
            return float(depth_map[y, x])
        return None
    
    def filter_by_depth(self, keypoints: List[cv2.KeyPoint], 
                       depth_map: Optional[np.ndarray]) -> List[cv2.KeyPoint]:
        """깊이 기반 필터링"""
        if depth_map is None:
            return keypoints
        
        filtered_keypoints = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            depth = self.get_depth_at_point(depth_map, x, y)
            
            if depth is not None:
                if self.depth_filter['min_depth'] <= depth <= self.depth_filter['max_depth']:
                    filtered_keypoints.append(kp)
            else:
                # 깊이 정보가 없으면 일단 포함
                filtered_keypoints.append(kp)
        
        return filtered_keypoints
    
    def analyze_blob_color(self, image: np.ndarray, keypoint: cv2.KeyPoint, 
                          roi_radius: int = 10) -> Dict:
        """Blob의 색상 분석"""
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        
        # ROI 영역 설정
        h, w = image.shape[:2]
        y_start = max(0, y - roi_radius)
        y_end = min(h, y + roi_radius)
        x_start = max(0, x - roi_radius)
        x_end = min(w, x + roi_radius)
        
        roi = image[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            return {'color': 'unknown', 'confidence': 0.0}
        
        # HSV로 변환
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 평균 색상 계산
        avg_h, avg_s, avg_v, _ = cv2.mean(roi_hsv)
        
        # 색상 판별
        color_info = {
            'hsv': (avg_h, avg_s, avg_v),
            'color': 'unknown',
            'confidence': 0.0
        }
        
        # 파란색 판별
        if (100 <= avg_h <= 130) and avg_s > 50 and avg_v > 50:
            color_info['color'] = 'blue'
            color_info['confidence'] = min(avg_s / 255.0, 1.0)
        # 빨간색 판별
        elif ((avg_h >= 0 and avg_h < 10) or (avg_h >= 170 and avg_h <= 180)) and avg_s > 50 and avg_v > 50:
            color_info['color'] = 'red'
            color_info['confidence'] = min(avg_s / 255.0, 1.0)
        # 초록색 판별
        elif (35 <= avg_h < 85) and avg_s > 50 and avg_v > 50:
            color_info['color'] = 'green'
            color_info['confidence'] = min(avg_s / 255.0, 1.0)
        
        return color_info
    
    def detect_blue_buoys(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None,
                         mode: str = 'hsv') -> List[Dict]:
        """파란색 부표 감지 (메인 함수)"""
        detections = []
        
        # 마스크 생성
        if mode == 'hsv':
            mask = self.create_blue_mask(image)
        elif mode == 'adaptive':
            mask = self.create_adaptive_mask(image)
        else:
            raise ValueError("mode는 'hsv' 또는 'adaptive'여야 합니다")
        
        # Blob 감지
        keypoints = self.detect_blobs(mask)
        
        # 깊이 필터링
        keypoints = self.filter_by_depth(keypoints, depth_map)
        
        # 각 blob 분석
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            # 색상 분석
            color_info = self.analyze_blob_color(image, kp)
            
            # 파란색만 선택
            if color_info['color'] == 'blue' and color_info['confidence'] > 0.3:
                # 깊이 정보
                depth = self.get_depth_at_point(depth_map, x, y)
                
                # Detection 객체 생성
                detection = {
                    'center': (x, y),
                    'color': 'blue',
                    'size': size,
                    'depth': depth,
                    'confidence': color_info['confidence'],
                    'keypoint': kp,
                    'bbox': (x - size//2, y - size//2, size, size)
                }
                
                detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """감지 결과 시각화"""
        result_image = image.copy()
        
        for detection in detections:
            x, y = detection['center']
            size = detection['size']
            depth = detection['depth']
            confidence = detection['confidence']
            
            # 바운딩 박스 그리기
            radius = size // 2
            cv2.circle(result_image, (x, y), radius, (255, 0, 0), 2)  # 파란색 원
            
            # 라벨 표시
            label = f"BLUE ({confidence:.2f})"
            cv2.putText(result_image, label, (x - 50, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 좌표 표시
            coord_text = f"({x}, {y})"
            cv2.putText(result_image, coord_text, (x - 30, y + radius + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 깊이 정보 표시
            if depth is not None:
                depth_text = f"Depth: {depth:.2f}m"
                cv2.putText(result_image, depth_text, (x - 40, y + radius + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            else:
                cv2.putText(result_image, "Depth: N/A", (x - 30, y + radius + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return result_image
