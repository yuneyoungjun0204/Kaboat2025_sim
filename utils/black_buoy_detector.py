#!/usr/bin/env python3
"""
검정색 부표 탐지 모듈
- 검정색 HSV 범위 탐지
- 깊이 필터링
- Blob 검출
"""

import cv2
import numpy as np
from typing import List, Tuple


class BlackBuoyDetector:
    """검정색 부표 탐지기"""
    
    def __init__(self):
        # 검정색 HSV 범위
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])  # V값 60 이하
        
        # Blob 파라미터
        self.min_area = 200
        self.max_area = 10000
        self.min_circularity = 0.3
        self.min_convexity = 0.5
        self.min_inertia_ratio = 0.2
    
    def update_detection_parameters(self, min_area=None, max_area=None, 
                                   min_circularity=None, min_convexity=None,
                                   min_inertia_ratio=None, v_threshold=None):
        """탐지 파라미터 업데이트"""
        if min_area is not None:
            self.min_area = min_area
        if max_area is not None:
            self.max_area = max_area
        if min_circularity is not None:
            self.min_circularity = min_circularity
        if min_convexity is not None:
            self.min_convexity = min_convexity
        if min_inertia_ratio is not None:
            self.min_inertia_ratio = min_inertia_ratio
        if v_threshold is not None:
            self.upper_black[2] = v_threshold
    
    def detect_black_buoys(self, image: np.ndarray, depth_map: np.ndarray,
                          min_depth: float, max_depth: float) -> List:
        """
        검정색 부표 탐지
        
        Args:
            image: BGR 이미지
            depth_map: 깊이 맵
            min_depth: 최소 깊이 임계값
            max_depth: 최대 깊이 임계값
            
        Returns:
            탐지된 부표 리스트
        """
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 검정색 마스크
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        
        # 깊이 필터링
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_mask = ((depth_normalized > min_depth) & (depth_normalized < max_depth)).astype(np.uint8) * 255
        
        # 최종 마스크 (검정색 AND 깊이 범위)
        final_mask = cv2.bitwise_and(mask, depth_mask)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 면적 필터
            if self.min_area < area < self.max_area:
                # 원형도 계산
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity >= self.min_circularity:
                        # 볼록도 계산
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            convexity = area / hull_area
                            
                            if convexity >= self.min_convexity:
                                # 중심점 계산
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    # 깊이값 추출
                                    if 0 <= cy < depth_normalized.shape[0] and 0 <= cx < depth_normalized.shape[1]:
                                        depth = depth_normalized[cy, cx]
                                    else:
                                        depth = 0.0
                                    
                                    # Detection 객체 생성
                                    detection = type('Detection', (), {
                                        'center': (cx, cy),
                                        'area': area,
                                        'color': 'black',
                                        'depth': depth,
                                        'circularity': circularity,
                                        'convexity': convexity,
                                        'contour': contour
                                    })()
                                    detections.append(detection)
        
        return detections
    
    def visualize_black_buoys(self, image: np.ndarray, tracks: List, detections: List,
                             control_mode: str = "navigation") -> np.ndarray:
        """
        검정색 부표 시각화
        
        Args:
            image: 원본 이미지
            tracks: 추적 중인 부표들
            detections: 탐지된 부표들
            control_mode: 제어 모드
            
        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()
        
        # 탐지된 부표들 표시 (흰색 박스)
        for detection in detections:
            cx, cy = detection.center
            cv2.rectangle(vis_image, (cx-30, cy-30), (cx+30, cy+30), (255, 255, 255), 2)
            cv2.putText(vis_image, "BLACK", (cx-25, cy-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if hasattr(detection, 'depth'):
                cv2.putText(vis_image, f"{detection.depth:.3f}m", (cx-25, cy+50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 추적 중인 부표들 표시 (노란색 박스 + 추적 정보)
        valid_tracks = [t for t in tracks if hasattr(t, 'confidence') and t.confidence > 0.3]
        
        for track in valid_tracks:
            cx, cy = track.center
            # 추적 박스 (노란색, 굵게)
            cv2.rectangle(vis_image, (cx-35, cy-35), (cx+35, cy+35), (0, 255, 255), 3)
            
            # 추적 ID
            cv2.putText(vis_image, f"ID:{track.track_id}", (cx-30, cy-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 신뢰도
            cv2.putText(vis_image, f"Conf:{track.confidence:.2f}", (cx-30, cy+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 깊이
            if hasattr(track, 'depth') and track.depth is not None:
                cv2.putText(vis_image, f"{track.depth:.3f}m", (cx-30, cy+75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Navigation 모드: 두 부표 사이 중점 표시
        if control_mode == "navigation" and len(valid_tracks) >= 2:
            t1, t2 = valid_tracks[0], valid_tracks[1]
            mid_x = (t1.center[0] + t2.center[0]) // 2
            mid_y = (t1.center[1] + t2.center[1]) // 2
            
            # 중점 마커 (보라색 큰 원)
            cv2.circle(vis_image, (mid_x, mid_y), 15, (255, 0, 255), -1)
            cv2.circle(vis_image, (mid_x, mid_y), 18, (255, 255, 255), 2)
            cv2.putText(vis_image, "TARGET", (mid_x-35, mid_y-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # 두 부표를 연결하는 선
            cv2.line(vis_image, t1.center, t2.center, (255, 0, 255), 2)
        
        # Approach 모드: 목표 부표 강조
        elif control_mode == "approach" and len(valid_tracks) >= 1:
            # 가장 신뢰도 높은 부표 또는 화면 중앙에 가까운 부표 선택
            target_track = max(valid_tracks, key=lambda t: t.confidence)
            cx, cy = target_track.center
            
            # 목표 부표 강조 (빨간색 큰 박스)
            cv2.rectangle(vis_image, (cx-45, cy-45), (cx+45, cy+45), (0, 0, 255), 4)
            cv2.putText(vis_image, "TARGET BUOY", (cx-55, cy-55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_image

