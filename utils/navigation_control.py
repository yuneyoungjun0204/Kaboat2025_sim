"""
네비게이션 제어 모듈
- PID 제어기를 통한 정밀한 제어
- 부표 간 네비게이션과 객체 접근 제어
"""

import time
import math

class PIDController:
    """PID 제어기"""
    
    def __init__(self, kp=1.5, ki=0.05, kd=0.4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def calculate(self, error):
        """PID 출력 계산"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01  # 최소 시간 간격
        
        # 비례 항
        proportional = self.kp * error
        
        # 적분 항
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # 미분 항
        derivative = self.kd * (error - self.previous_error) / dt
        
        # PID 출력
        output = proportional + integral + derivative
        
        # 상태 업데이트
        self.previous_error = error
        self.last_time = current_time
        
        return output
    
    def update_parameters(self, kp=None, ki=None, kd=None):
        """PID 파라미터 업데이트"""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

class NavigationController:
    """네비게이션 제어기"""
    
    def __init__(self, image_width=1280, image_height=720):
        self.image_width = image_width
        self.image_height = image_height
        
        # PID 제어기들
        self.steering_pid = PIDController(kp=1.0, ki=0.001, kd=0.2)
        self.approach_pid = PIDController(kp=0.8, ki=0.001, kd=0.4)
        
        # 제어 파라미터
        self.max_speed = 800.0
        self.min_speed = 300.0
        self.max_turn_thrust = 500.0
        self.base_speed = 150.0
        
        # 네비게이션 모드
        self.mode = "navigation"  # "navigation" 또는 "approach"
        
    def calculate_steering_command(self, error):
        """조향 명령 계산"""
        # 오차 정규화 (이미지 너비의 절반으로 나누어 -1~1 범위로)
        normalized_error = error / (self.image_width / 2)
        
        # PID 제어기로 조향 명령 계산
        steering_command = self.steering_pid.calculate(normalized_error)
        
        # 조향 명령 제한
        steering_command = max(-1.0, min(1.0, steering_command))
        
        return steering_command
    
    def calculate_approach_command(self, error):
        """접근 명령 계산"""
        # 오차 정규화
        normalized_error = error / (self.image_width / 2)
        
        # PID 제어기로 접근 명령 계산
        approach_command = self.approach_pid.calculate(normalized_error)
        
        # 접근 명령 제한
        approach_command = max(-1.0, min(1.0, approach_command))
        
        return approach_command
    
    def calculate_adaptive_speed(self, turn_thrust):
        """적응적 속도 계산 (회전 각도에 따라 속도 조절)"""
        # 회전 비율 계산 (0~1)
        turn_ratio = abs(turn_thrust) / self.max_turn_thrust
        turn_ratio = min(1.0, turn_ratio)  # 1.0으로 제한
        
        # 선형적으로 속도 감소
        adaptive_speed = self.max_speed - (self.max_speed - self.min_speed) * turn_ratio + 30
        
        return adaptive_speed
    
    def calculate_rotation_speed(self, turn_angle):
        """각도에 따른 적응형 속도 계산 (각도가 클수록 속도 감소)"""
        # 각도를 절댓값으로 변환 (0~180도)
        abs_angle = abs(turn_angle)
        
        # 각도가 클수록 속도 감소 (선형적)
        # 0도: 기본 속도, 90도: 최소 속도
        if abs_angle >= 90:
            return self.min_speed
        else:
            speed_ratio = 1.0 - (abs_angle / 90.0)
            adaptive_speed = self.min_speed + (self.base_speed - self.min_speed) * speed_ratio
            return max(self.min_speed, adaptive_speed)
    
    def calculate_rotation_target(self, rotation_direction, object_x):
        """회전 방향에 따른 목표 x 좌표 계산 (거리에 따라 동적 조정)"""
        if rotation_direction == 1:  # 시계방향
            # 시계방향: 1200 - 100x (멀수록 크게, 가까울수록 작게)
            target_x = 1240 - 3000 * object_x
            # 범위 제한 (640~1200)
            return max(940, min(1200, target_x))
        else:  # 반시계방향
            # 반시계방향: 40 + 100x (멀수록 크게, 가까울수록 작게)
            target_x = 3000 * object_x
            # 범위 제한 (40~640)
            return max(340, min(640, target_x))
    
    def navigation_control(self, red_buoy_x, green_buoy_x, red_buoy_y, green_buoy_y):
        """부표 간 네비게이션 제어"""
        # 부표 간 중점 계산
        center_x = (red_buoy_x + green_buoy_x) / 2
        center_y = (red_buoy_y + green_buoy_y) / 2
        
        # 목표 위치 (이미지 중앙)
        target_x = self.image_width / 2
        
        # 오차 계산
        error = target_x - center_x
        
        # 조향 명령 계산
        steering_command = self.calculate_steering_command(error)
        
        # 회전 추력 계산
        turn_thrust = steering_command * self.max_turn_thrust
        
        # 적응적 속도 계산
        forward_thrust = self.calculate_adaptive_speed(turn_thrust)
        
        # 스러스터 명령 계산 (VRX 환경에 맞게 조정)
        left_command = forward_thrust - turn_thrust
        right_command = forward_thrust + turn_thrust
        
        return left_command, right_command, error, steering_command, forward_thrust, turn_thrust
    
    def approach_control(self, target_x, target_y, target_depth, 
                        approach_distance=0.05, slow_distance=0.03, stop_distance=0.02,
                        rotation_direction=1):
        """객체 접근 제어"""
        # stop_distance 기준 충족 시 회전 시작
        if target_depth >= stop_distance:
            # 회전 모드: 고깔을 기준으로 일정한 방향으로 회전
            rotation_target_x = self.calculate_rotation_target(rotation_direction, target_depth)
            error = rotation_target_x - target_x
            
            # 조향 명령 계산
            steering_command = self.calculate_approach_command(error)
            
            # 회전 추력 계산
            turn_thrust = steering_command * self.max_turn_thrust
            
            # 각도에 따른 적응형 속도 계산
            turn_angle = abs(steering_command * 90)  # 조향 명령을 각도로 변환
            forward_thrust = self.calculate_rotation_speed(turn_angle)
            
            # 스러스터 명령 계산
            left_command = forward_thrust - turn_thrust
            right_command = forward_thrust + turn_thrust
            
            mode = "rotation"
            target_x_output = rotation_target_x
            
        else:
            # 접근 모드: 객체에 접근
            error = self.image_width / 2 - target_x
            steering_command = self.calculate_approach_command(error)
            turn_thrust = steering_command * self.max_turn_thrust
            forward_thrust = self.base_speed * 0.5  # 접근 시 천천히
            
            # 스러스터 명령 계산
            left_command = forward_thrust + turn_thrust
            right_command = forward_thrust - turn_thrust
            
            mode = "approach"
            target_x_output = self.image_width / 2
        
        return left_command, right_command, error, steering_command, forward_thrust, turn_thrust, mode, target_x_output
    
    def update_control_parameters(self, max_speed=None, min_speed=None, max_turn_thrust=None, 
                                base_speed=None, steering_kp=None, approach_kp=None):
        """제어 파라미터 업데이트"""
        if max_speed is not None:
            self.max_speed = max_speed
        if min_speed is not None:
            self.min_speed = min_speed
        if max_turn_thrust is not None:
            self.max_turn_thrust = max_turn_thrust
        if base_speed is not None:
            self.base_speed = base_speed
        if steering_kp is not None:
            self.steering_pid.update_parameters(kp=steering_kp)
        if approach_kp is not None:
            self.approach_pid.update_parameters(kp=approach_kp)
