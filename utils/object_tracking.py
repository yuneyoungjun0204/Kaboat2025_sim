"""
객체 추적 모듈
- 검출된 객체들의 추적 관리
- IMM (Interacting Multiple Model) + MMPDAM (Multi-Model Probabilistic Data Association Method)
"""

import time
import math
from collections import deque

class Track:
    """개별 추적 객체"""
    
    def __init__(self, track_id, initial_detection, depth_value):
        self.track_id = track_id
        
        # Detection 객체의 속성에 접근
        if hasattr(initial_detection, 'color'):
            self.color = initial_detection.color
            self.center = initial_detection.center
            self.bbox = getattr(initial_detection, 'bbox', None)
            self.area = initial_detection.area
        else:
            self.color = initial_detection['color']
            self.center = initial_detection['center']
            self.bbox = initial_detection['bbox']
            self.area = initial_detection['area']
        
        self.detection_history = deque(maxlen=10)
        self.detection_history.append(initial_detection)
        self.depth = depth_value
        self.confidence = 0.8 if depth_value is not None else 0.5
        
        # 추적 상태
        self.missed_frames = 0
        self.total_frames = 1
        self.last_update = time.time()
        
        # 모션 모델 (간단한 칼만 필터)
        self.velocity = [0.0, 0.0]  # [vx, vy]
        self.predicted_center = self.center
        
        # 추적 궤적 저장 (시각화용)
        self.trajectory = deque(maxlen=20)  # 최근 20개 위치 저장
        self.trajectory.append(self.center)
        
        # 추적 품질 지표
        self.stability_score = 1.0  # 추적 안정성 점수
        self.depth_consistency = 1.0  # 깊이 일관성 점수
        
    def update(self, detection, depth_value):
        """추적 업데이트"""
        # 이전 중심점 저장
        prev_center = self.center
        
        # 새로운 탐지로 업데이트
        # Detection 객체의 속성에 접근
        if hasattr(detection, 'center'):
            self.center = detection.center
            self.bbox = getattr(detection, 'bbox', None)
            self.area = detection.area
        else:
            self.center = detection['center']
            self.bbox = detection['bbox']
            self.area = detection['area']
        self.depth = depth_value
        self.detection_history.append(detection)
        
        # 궤적에 새 위치 추가
        self.trajectory.append(self.center)
        
        # 속도 계산
        self.velocity = [
            self.center[0] - prev_center[0],
            self.center[1] - prev_center[1]
        ]
        
        # 추적 안정성 점수 계산 (위치 변화량 기반)
        movement = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if movement < 5:  # 작은 움직임은 안정적
            self.stability_score = min(1.0, self.stability_score + 0.05)
        else:  # 큰 움직임은 불안정
            self.stability_score = max(0.1, self.stability_score - 0.1)
        
        # 깊이 일관성 점수 계산
        if depth_value is not None and self.depth is not None:
            depth_change = abs(depth_value - self.depth)
            if depth_change < 0.1:  # 깊이 변화가 작으면 일관적
                self.depth_consistency = min(1.0, self.depth_consistency + 0.05)
            else:  # 깊이 변화가 크면 불일치
                self.depth_consistency = max(0.1, self.depth_consistency - 0.1)
        
        # 신뢰도 업데이트 (안정성과 깊이 일관성 반영)
        base_confidence = 0.8 if depth_value is not None else 0.5
        stability_factor = self.stability_score * 0.4
        depth_factor = self.depth_consistency * 0.3
        self.confidence = base_confidence + stability_factor + depth_factor
        self.confidence = min(1.0, max(0.1, self.confidence))
        
        self.missed_frames = 0
        self.total_frames += 1
        self.last_update = time.time()
        
        # 다음 프레임 예측
        self.predicted_center = (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1]
        )
    
    def predict(self):
        """다음 위치 예측"""
        # 간단한 선형 예측
        predicted_center = (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1]
        )
        
        # 신뢰도 감소 (모션 모델에 따라)
        motion_models = {
            'stationary': 0.95,    # 정지 모델
            'drifting': 0.9,       # 표류 모델
            'wave': 0.85,          # 파도 모델
            'boat_induced': 0.8    # 보트 유도 모델
        }
        
        # 현재 모션 상태에 따른 신뢰도 조정
        speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed < 2:
            model_factor = motion_models['stationary']
        elif speed < 5:
            model_factor = motion_models['drifting']
        elif speed < 10:
            model_factor = motion_models['wave']
        else:
            model_factor = motion_models['boat_induced']
        
        self.confidence *= model_factor
        self.missed_frames += 1
        
        return predicted_center

class MultiTargetTracker:
    """다중 표적 추적기 (IMM + MMPDAM)"""
    
    def __init__(self, max_tracks=10, max_missed_frames=5, gate_threshold=50.0):
        self.max_tracks = max_tracks
        self.max_missed_frames = max_missed_frames
        self.gate_threshold = gate_threshold
        self.min_association_prob = 0.3
        
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        
    def update(self, detections, depth_map):
        """추적 업데이트"""
        self.frame_count += 1
        
        # 기존 트랙 예측
        for track in self.tracks:
            track.predict()
        
        # 데이터 연결 (MMPDAM)
        associations = self._associate_detections_to_tracks(detections, depth_map)
        
        # 트랙 업데이트
        self._update_tracks(associations, detections, depth_map)
        
        # 새로운 트랙 생성
        self._create_new_tracks(detections, depth_map, associations)
        
        # 오래된 트랙 제거
        self._remove_old_tracks()
        
        return self.tracks
    
    def _associate_detections_to_tracks(self, detections, depth_map):
        """탐지와 트랙 연결 (MMPDAM)"""
        associations = {}
        used_detections = set()
        
        # 각 트랙에 대해 최적의 탐지 찾기
        for track in self.tracks:
            best_detection = None
            best_score = -1
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                # 거리 기반 점수 계산
                # Detection 객체의 center 속성에 접근
                if hasattr(detection, 'center'):
                    detection_center = detection.center
                else:
                    detection_center = detection['center']
                
                distance = math.sqrt(
                    (detection_center[0] - track.predicted_center[0])**2 +
                    (detection_center[1] - track.predicted_center[1])**2
                )
                
                # 게이트 테스트
                if distance > self.gate_threshold:
                    continue
                
                # 색상 일치 확인
                detection_color = detection.color if hasattr(detection, 'color') else detection['color']
                if detection_color != track.color:
                    continue
                
                # 깊이 정보로 점수 보정
                depth_value = self._get_depth_at_point(depth_map, detection_center[0], detection_center[1])
                depth_score = 1.0 if depth_value is not None else 0.5
                
                # 최종 점수 계산
                score = (1.0 / (1.0 + distance)) * depth_score * track.confidence
                
                if score > best_score and score > self.min_association_prob:
                    best_score = score
                    best_detection = i
            
            if best_detection is not None:
                associations[track.track_id] = best_detection
                used_detections.add(best_detection)
        
        return associations
    
    def _update_tracks(self, associations, detections, depth_map):
        """트랙 업데이트"""
        for track in self.tracks:
            if track.track_id in associations:
                detection_idx = associations[track.track_id]
                detection = detections[detection_idx]
                # Detection 객체의 center 속성에 접근
                if hasattr(detection, 'center'):
                    detection_center = detection.center
                else:
                    detection_center = detection['center']
                depth_value = self._get_depth_at_point(depth_map, detection_center[0], detection_center[1])
                track.update(detection, depth_value)
    
    def _create_new_tracks(self, detections, depth_map, associations):
        """새로운 트랙 생성"""
        used_detections = set(associations.values())
        
        for i, detection in enumerate(detections):
            if i in used_detections:
                continue
            
            if len(self.tracks) >= self.max_tracks:
                break
            
            # Detection 객체의 center 속성에 접근
            if hasattr(detection, 'center'):
                detection_center = detection.center
            else:
                detection_center = detection['center']
            depth_value = self._get_depth_at_point(depth_map, detection_center[0], detection_center[1])
            new_track = Track(self.next_track_id, detection, depth_value)
            self.tracks.append(new_track)
            self.next_track_id += 1
    
    def _remove_old_tracks(self):
        """오래된 트랙 제거"""
        self.tracks = [track for track in self.tracks 
                      if track.missed_frames < self.max_missed_frames]
    
    def _get_depth_at_point(self, depth_map, x, y):
        """특정 좌표에서의 깊이 값 반환"""
        if depth_map is None:
            return None
        
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return float(depth_map[y, x])
        return None
    
    def get_best_tracks(self):
        """최고 신뢰도 트랙 반환"""
        red_tracks = [track for track in self.tracks if track.color == 'red']
        green_tracks = [track for track in self.tracks if track.color == 'green']
        
        best_red = max(red_tracks, key=lambda t: t.confidence) if red_tracks else None
        best_green = max(green_tracks, key=lambda t: t.confidence) if green_tracks else None
        
        return best_red, best_green
    
    def update_tracking_parameters(self, max_tracks=None, max_missed_frames=None, gate_threshold=None, min_association_prob=None):
        """추적 파라미터 업데이트"""
        if max_tracks is not None:
            self.max_tracks = max_tracks
        if max_missed_frames is not None:
            self.max_missed_frames = max_missed_frames
        if gate_threshold is not None:
            self.gate_threshold = gate_threshold
        if min_association_prob is not None:
            self.min_association_prob = min_association_prob
