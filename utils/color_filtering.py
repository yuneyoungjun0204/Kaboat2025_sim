"""
색상 필터링 모듈
- HSV 색상 공간에서 특정 색상 범위 필터링
- 빨간색, 초록색 부표 탐지를 위한 색상 마스크 생성
"""

import cv2
import numpy as np

class ColorFilter:
    """HSV 색상 기반 필터링 클래스"""
    
    def __init__(self):
        # HSV 색상 범위 (부표 색상)
        self.red_lower = np.array([0, 50, 50])
        self.red_upper = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
    
    def create_red_mask(self, hsv_image):
        """빨간색 마스크 생성"""
        # 빨간색 범위 1 (0-10도)
        red_mask1 = cv2.inRange(hsv_image, self.red_lower, self.red_upper)
        # 빨간색 범위 2 (170-180도) - 색상 원형 구조로 인한 처리
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        # 두 마스크 합치기
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        return red_mask
    
    def create_green_mask(self, hsv_image):
        """초록색 마스크 생성"""
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        return green_mask
    
    def create_color_mask(self, hsv_image, color):
        """지정된 색상의 마스크 생성"""
        if color.lower() == 'red':
            return self.create_red_mask(hsv_image)
        elif color.lower() == 'green':
            return self.create_green_mask(hsv_image)
        else:
            raise ValueError(f"지원하지 않는 색상: {color}")
    
    def update_color_ranges(self, color, lower, upper):
        """색상 범위 업데이트"""
        if color.lower() == 'red':
            self.red_lower = lower
            self.red_upper = upper
        elif color.lower() == 'green':
            self.green_lower = lower
            self.green_upper = upper
        else:
            raise ValueError(f"지원하지 않는 색상: {color}")
    
    def get_color_ranges(self, color):
        """색상 범위 반환"""
        if color.lower() == 'red':
            return self.red_lower, self.red_upper, self.red_lower2, self.red_upper2
        elif color.lower() == 'green':
            return self.green_lower, self.green_upper
        else:
            raise ValueError(f"지원하지 않는 색상: {color}")
