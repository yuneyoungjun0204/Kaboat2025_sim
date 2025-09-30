"""
객체 검출 모듈
- CV 처리를 통한 이미지에서의 객체 검출
- 색상 필터링과 형태 분석을 통한 부표 탐지
"""

import cv2
import numpy as np
from .color_filtering import ColorFilter

class BlobDetector:
    """HSV 색상 기반 부표 탐지기"""
    
    def __init__(self):
        self.color_filter = ColorFilter()
        
        # 형태 필터링 파라미터
        self.min_area = 100
        self.max_area = 10000
        self.min_circularity = 0.3
    
    def detect_blobs(self, image, depth_map=None, min_depth_threshold=0.035, max_depth_threshold=0.4):
        """이미지에서 부표 탐지 (깊이 필터링 포함)"""
        detections = []
        
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 빨간색 부표 탐지
        red_mask = self.color_filter.create_red_mask(hsv)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in red_contours:
            if self._is_valid_blob(contour):
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 깊이 필터링 적용
                if depth_map is not None:
                    depth_value = self._get_depth_at_point(depth_map, center_x, center_y)
                    if depth_value is not None:
                        # 깊이 범위 체크
                        if depth_value < min_depth_threshold or depth_value > max_depth_threshold:
                            continue  # 깊이 범위를 벗어나면 제외
                
                detections.append({
                    'color': 'red',
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour)
                })
        
        # 초록색 부표 탐지
        green_mask = self.color_filter.create_green_mask(hsv)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in green_contours:
            if self._is_valid_blob(contour):
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 깊이 필터링 적용
                if depth_map is not None:
                    depth_value = self._get_depth_at_point(depth_map, center_x, center_y)
                    if depth_value is not None:
                        # 깊이 범위 체크
                        if depth_value < min_depth_threshold or depth_value > max_depth_threshold:
                            continue  # 깊이 범위를 벗어나면 제외
                
                detections.append({
                    'color': 'green',
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour)
                })
        
        return detections
    
    def _get_depth_at_point(self, depth_map, x, y):
        """특정 좌표에서의 깊이 값 반환"""
        if depth_map is None:
            return None
        
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return float(depth_map[y, x])
        return None
    
    def _is_valid_blob(self, contour):
        """유효한 부표인지 확인"""
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return False
        
        # 원형도 계산
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity >= self.min_circularity
    
    def update_detection_parameters(self, min_area=None, max_area=None, min_circularity=None):
        """탐지 파라미터 업데이트"""
        if min_area is not None:
            self.min_area = min_area
        if max_area is not None:
            self.max_area = max_area
        if min_circularity is not None:
            self.min_circularity = min_circularity
    
    def get_detection_parameters(self):
        """현재 탐지 파라미터 반환"""
        return {
            'min_area': self.min_area,
            'max_area': self.max_area,
            'min_circularity': self.min_circularity
        }
