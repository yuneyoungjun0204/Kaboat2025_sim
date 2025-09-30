"""
트랙바 제어 모듈
- GUI 트랙바를 통한 실시간 파라미터 조정
- 모든 제어 파라미터를 통합 관리
"""

import cv2

class TrackbarController:
    """트랙바 제어기"""
    
    def __init__(self):
        self.window_name = "VRX Control Panel"
        
        # 트랙바 설정
        self.setup_trackbars()
        
        # 기본값 설정
        self.set_default_values()
    
    def setup_trackbars(self):
        """트랙바 설정"""
        cv2.namedWindow(self.window_name)
        
        # 제어 모드 트랙바
        cv2.createTrackbar("Control_Mode", self.window_name, 0, 1, self.nothing)  # 0: navigation, 1: approach
        cv2.createTrackbar("Target_Color", self.window_name, 1, 1, self.nothing)  # 0: red, 1: green
        cv2.createTrackbar("Rotation_Direction", self.window_name, 1, 1, self.nothing)  # 0: 시계방향, 1: 반시계방향
        
        # 탐지 파라미터
        cv2.createTrackbar("Min_Depth_Threshold", self.window_name, 35, 100, self.nothing)
        cv2.createTrackbar("Max_Depth_Threshold", self.window_name, 400, 500, self.nothing)
        
        # Blob Detector 파라미터
        cv2.createTrackbar("Min_Area", self.window_name, 100, 5000, self.nothing)
        cv2.createTrackbar("Max_Area", self.window_name, 10000, 50000, self.nothing)
        cv2.createTrackbar("Min_Circularity", self.window_name, 30, 100, self.nothing)
        
        # 추적 파라미터
        cv2.createTrackbar("Max_Tracks", self.window_name, 10, 20, self.nothing)
        cv2.createTrackbar("Max_Missed_Frames", self.window_name, 5, 20, self.nothing)
        cv2.createTrackbar("Gate_Threshold", self.window_name, 50, 200, self.nothing)
        cv2.createTrackbar("Min_Association_Prob", self.window_name, 30, 100, self.nothing)
        
        # 제어 파라미터
        cv2.createTrackbar("Max_Speed", self.window_name, 1000, 2000, self.nothing)
        cv2.createTrackbar("Min_Speed", self.window_name, 300, 1000, self.nothing)
        cv2.createTrackbar("Base_Speed", self.window_name, 150, 500, self.nothing)
        cv2.createTrackbar("Max_Turn_Thrust", self.window_name, 100, 250, self.nothing)
        
        # PID 파라미터
        cv2.createTrackbar("Steering_Kp", self.window_name, 10, 50, self.nothing)  # 0.1-5.0
        cv2.createTrackbar("Approach_Kp", self.window_name, 8, 50, self.nothing)   # 0.1-5.0
        
        # 시각화 옵션
        cv2.createTrackbar("Show_Depth", self.window_name, 1, 1, self.nothing)
    
    def set_default_values(self):
        """기본값 설정"""
        # 기본값들 설정
        cv2.setTrackbarPos("Control_Mode", self.window_name, 0)  # navigation
        cv2.setTrackbarPos("Target_Color", self.window_name, 1)  # green
        cv2.setTrackbarPos("Rotation_Direction", self.window_name, 1)  # 시계방향
        cv2.setTrackbarPos("Min_Depth_Threshold", self.window_name, 35)  # 0.035
        cv2.setTrackbarPos("Max_Depth_Threshold", self.window_name, 400)  # 0.4
        cv2.setTrackbarPos("Min_Area", self.window_name, 100)
        cv2.setTrackbarPos("Max_Area", self.window_name, 10000)
        cv2.setTrackbarPos("Min_Circularity", self.window_name, 30)
        cv2.setTrackbarPos("Max_Tracks", self.window_name, 10)
        cv2.setTrackbarPos("Max_Missed_Frames", self.window_name, 5)
        cv2.setTrackbarPos("Gate_Threshold", self.window_name, 50)
        cv2.setTrackbarPos("Min_Association_Prob", self.window_name, 30)
        cv2.setTrackbarPos("Max_Speed", self.window_name, 1000)
        cv2.setTrackbarPos("Min_Speed", self.window_name, 300)
        cv2.setTrackbarPos("Base_Speed", self.window_name, 150)
        cv2.setTrackbarPos("Max_Turn_Thrust", self.window_name, 100)
        cv2.setTrackbarPos("Steering_Kp", self.window_name, 10)  # 1.0
        cv2.setTrackbarPos("Approach_Kp", self.window_name, 8)   # 0.8
        cv2.setTrackbarPos("Show_Depth", self.window_name, 1)
    
    def nothing(self, val):
        """트랙바 콜백 함수"""
        pass
    
    def get_control_parameters(self):
        """제어 파라미터 읽기"""
        control_mode = cv2.getTrackbarPos("Control_Mode", self.window_name)
        target_color = cv2.getTrackbarPos("Target_Color", self.window_name)
        rotation_direction = cv2.getTrackbarPos("Rotation_Direction", self.window_name)
        
        return {
            'control_mode': "approach" if control_mode == 1 else "navigation",
            'target_color': "green" if target_color == 1 else "red",
            'rotation_direction': 2 if rotation_direction == 1 else 1
        }
    
    def get_detection_parameters(self):
        """탐지 파라미터 읽기"""
        min_depth_threshold = cv2.getTrackbarPos("Min_Depth_Threshold", self.window_name) / 1000.0
        max_depth_threshold = cv2.getTrackbarPos("Max_Depth_Threshold", self.window_name) / 1000.0
        
        return {
            'min_depth_threshold': min_depth_threshold,
            'max_depth_threshold': max_depth_threshold
        }
    
    def get_blob_detector_parameters(self):
        """Blob Detector 파라미터 읽기"""
        min_area = cv2.getTrackbarPos("Min_Area", self.window_name)
        max_area = cv2.getTrackbarPos("Max_Area", self.window_name)
        min_circularity = cv2.getTrackbarPos("Min_Circularity", self.window_name) / 100.0
        
        return {
            'min_area': min_area,
            'max_area': max_area,
            'min_circularity': min_circularity
        }
    
    def get_tracking_parameters(self):
        """추적 파라미터 읽기"""
        max_tracks = cv2.getTrackbarPos("Max_Tracks", self.window_name)
        max_missed_frames = cv2.getTrackbarPos("Max_Missed_Frames", self.window_name)
        gate_threshold = cv2.getTrackbarPos("Gate_Threshold", self.window_name)
        min_association_prob = cv2.getTrackbarPos("Min_Association_Prob", self.window_name) / 100.0
        
        return {
            'max_tracks': max_tracks,
            'max_missed_frames': max_missed_frames,
            'gate_threshold': gate_threshold,
            'min_association_prob': min_association_prob
        }
    
    def get_navigation_parameters(self):
        """네비게이션 파라미터 읽기"""
        max_speed = cv2.getTrackbarPos("Max_Speed", self.window_name)
        min_speed = cv2.getTrackbarPos("Min_Speed", self.window_name)
        base_speed = cv2.getTrackbarPos("Base_Speed", self.window_name)
        max_turn_thrust = cv2.getTrackbarPos("Max_Turn_Thrust", self.window_name)
        
        return {
            'max_speed': max_speed,
            'min_speed': min_speed,
            'base_speed': base_speed,
            'max_turn_thrust': max_turn_thrust
        }
    
    def get_pid_parameters(self):
        """PID 파라미터 읽기"""
        steering_kp = cv2.getTrackbarPos("Steering_Kp", self.window_name) / 10.0
        approach_kp = cv2.getTrackbarPos("Approach_Kp", self.window_name) / 10.0
        
        return {
            'steering_kp': steering_kp,
            'approach_kp': approach_kp
        }
    
    def get_visualization_parameters(self):
        """시각화 파라미터 읽기"""
        show_depth = cv2.getTrackbarPos("Show_Depth", self.window_name)
        
        return {
            'show_depth': show_depth == 1
        }
    
    def get_all_parameters(self):
        """모든 파라미터 읽기"""
        return {
            'control': self.get_control_parameters(),
            'detection': self.get_detection_parameters(),
            'blob_detector': self.get_blob_detector_parameters(),
            'tracking': self.get_tracking_parameters(),
            'navigation': self.get_navigation_parameters(),
            'pid': self.get_pid_parameters(),
            'visualization': self.get_visualization_parameters()
        }
