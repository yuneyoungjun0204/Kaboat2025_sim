"""
시각화 모듈
- 추적 결과 시각화
- 깊이 맵 시각화
- 통계 정보 표시
"""

import cv2
import numpy as np

class Visualizer:
    """시각화 클래스"""
    
    def __init__(self):
        # 창 이름들
        self.combined_window = "VRX Robot Control System"
        self.control_window = "VRX Control Panel"
        
        # 창 생성
        cv2.namedWindow(self.combined_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        
        # 창 크기 설정
        cv2.resizeWindow(self.combined_window, 1300, 540)  # 가로 1300, 세로 540
        cv2.resizeWindow(self.control_window, 400, 600)    # 가로 400, 세로 600
        
        # 창 위치 설정 (화면에 나란히 배치)
        cv2.moveWindow(self.combined_window, 100, 100)     # 메인 창
        cv2.moveWindow(self.control_window, 1450, 100)     # 컨트롤 창 (오른쪽)
        
    def visualize_tracking_results(self, image, tracks, detections, frame_count, control_mode, target_color):
        """추적 결과 시각화"""
        # 원본 이미지 복사
        image_with_tracks = image.copy()
        
        # 모든 활성 트랙 표시
        for track in tracks:
            center_x, center_y = track.center
            x, y, w, h = track.bbox
            color = track.color
            
            # 색상 설정
            if color == 'red':
                bbox_color = (0, 0, 255)
                text_color = (0, 0, 255)
            else:
                bbox_color = (0, 255, 0)
                text_color = (0, 255, 0)
            
            # 바운딩 박스 그리기
            cv2.rectangle(image_with_tracks, (x, y), (x + w, y + h), bbox_color, 2)
            
            # 중심점 그리기
            cv2.circle(image_with_tracks, (center_x, center_y), 6, bbox_color, -1)
            
            # 라벨 그리기
            label = f"{color.upper()}_ID{track.track_id}_C{track.confidence:.2f}"
            cv2.putText(image_with_tracks, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # 궤적 그리기
            if len(track.trajectory) > 1:
                trajectory_points = list(track.trajectory)
                for i in range(1, len(trajectory_points)):
                    pt1 = (int(trajectory_points[i-1][0]), int(trajectory_points[i-1][1]))
                    pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
                    thickness = max(1, int(3 * (i / len(trajectory_points))))
                    cv2.line(image_with_tracks, pt1, pt2, bbox_color, thickness)
        
        # 제어 모드 정보 표시
        mode_text = f"Mode: {control_mode.upper()} | Target: {target_color.upper()}"
        cv2.putText(image_with_tracks, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_with_tracks, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 통계 정보 표시
        stats_text = f"Tracks: {len(tracks)} | Detections: {len(detections)} | Frame: {frame_count}"
        cv2.putText(image_with_tracks, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image_with_tracks, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return image_with_tracks
    
    def visualize_depth_map(self, depth_map, tracks, detections):
        """깊이 맵 시각화"""
        if depth_map is None:
            return None
        
        # 깊이 맵 대비 극대화
        depth_min = np.percentile(depth_map, 5)
        depth_max = np.percentile(depth_map, 95)
        
        depth_clipped = np.clip(depth_map, depth_min, depth_max)
        depth_vis = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.equalizeHist(depth_vis)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        
        # 깊이 맵 정보 표시
        legend_text = "MiDaS Hybrid Depth Map: Blue=Near, Red=Far"
        cv2.putText(depth_colored, legend_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_colored, legend_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 깊이 통계 정보
        depth_min_val = np.min(depth_map)
        depth_max_val = np.max(depth_map)
        depth_mean_val = np.mean(depth_map)
        
        stats_text = f"Depth Range: {depth_min_val:.3f} - {depth_max_val:.3f} (Mean: {depth_mean_val:.3f})"
        cv2.putText(depth_colored, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(depth_colored, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 트랙 정보 표시
        for track in tracks:
            center_x, center_y = track.center
            color = track.color
            
            if color == 'red':
                track_color = (0, 0, 255)
            else:
                track_color = (0, 255, 0)
            
            cv2.circle(depth_colored, (center_x, center_y), 8, track_color, -1)
            cv2.circle(depth_colored, (center_x, center_y), 8, (255, 255, 255), 2)
            
            if track.depth is not None:
                track_info = f"ID{track.track_id}_D{track.depth:.3f}"
                cv2.putText(depth_colored, track_info, (center_x + 10, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 후보 탐지 표시
        for detection in detections:
            center_x, center_y = detection['center']
            color = detection['color']
            
            if color == 'red':
                color_circle = (0, 0, 255)
                color_text = (255, 255, 255)
                label = "R"
            else:
                color_circle = (0, 255, 0)
                color_text = (255, 255, 255)
                label = "G"
            
            cv2.circle(depth_colored, (center_x, center_y), 6, color_circle, -1)
            cv2.putText(depth_colored, label, (center_x + 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)
        
        return depth_colored
    
    def show_images(self, main_image, depth_image=None, show_depth=True):
        """이미지를 하나의 창에서 subplot으로 표시"""
        if show_depth and depth_image is not None:
            # 두 이미지를 나란히 결합
            combined_image = self.combine_images_side_by_side(main_image, depth_image)
        else:
            # 깊이 맵이 없으면 메인 이미지만 표시
            combined_image = main_image
        
        # 통합 이미지 표시
        cv2.imshow(self.combined_window, combined_image)
        cv2.waitKey(1)
    
    def combine_images_side_by_side(self, main_image, depth_image):
        """두 이미지를 나란히 결합하여 subplot 형태로 표시"""
        # 화면에 맞는 크기로 조정 (더 작은 크기로 조정)
        target_height = 480
        target_width = 640
        
        # 메인 이미지 크기 조정
        main_resized = cv2.resize(main_image, (target_width, target_height))
        
        # 깊이 이미지 크기 조정
        depth_resized = cv2.resize(depth_image, (target_width, target_height))
        
        # 두 이미지를 나란히 결합 (가로로 연결)
        combined_width = target_width * 2 + 10  # 10px 간격
        combined_height = target_height + 60    # 제목 공간 추가
        
        # 결합된 이미지 생성 (검은 배경)
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # 메인 이미지를 왼쪽에 배치
        combined_image[60:60+target_height, :target_width] = main_resized
        
        # 깊이 이미지를 오른쪽에 배치
        combined_image[60:60+target_height, target_width+10:] = depth_resized
        
        # 구분선 그리기 (세로선)
        cv2.line(combined_image, (target_width + 5, 60), (target_width + 5, combined_height), (100, 100, 100), 2)
        
        # 제목 바 생성 (상단)
        cv2.rectangle(combined_image, (0, 0), (combined_width, 60), (50, 50, 50), -1)
        cv2.rectangle(combined_image, (0, 0), (combined_width, 60), (200, 200, 200), 2)
        
        # 메인 제목
        cv2.putText(combined_image, "VRX Robot Control System", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 서브 제목들
        cv2.putText(combined_image, "Main Tracking View", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(combined_image, "Depth Map View", (target_width + 20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        # 이미지 테두리 추가
        cv2.rectangle(combined_image, (0, 60), (target_width, 60+target_height), (200, 200, 200), 2)
        cv2.rectangle(combined_image, (target_width+10, 60), (combined_width, 60+target_height), (200, 200, 200), 2)
        
        return combined_image
    
    def cleanup(self):
        """정리"""
        cv2.destroyAllWindows()
