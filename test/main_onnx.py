#!/usr/bin/env python3
"""
VRX 강화학습 ONNX 모델 기반 선박 제어 시스템
- Ray-48130414.onnx 모델을 사용하여 목표 전진값, 목표 선회값 출력
- LiDAR, IMU, GPS 데이터를 입력으로 사용
- ROS2를 통해 스러스터 제어
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import LaserScan, NavSatFix, Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import time
from collections import deque
from utils import SensorDataManager

class VRXONNXController(Node):
    """VRX ONNX 모델 기반 제어 노드"""
    
    def __init__(self):
        super().__init__('vrx_onnx_controller')
        
        # 로그 레벨을 DEBUG로 설정
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        # ONNX 모델 로드
        self.model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/Ray-48130414.onnx'
        
        # 실행 공급자 확인 및 설정
        available_providers = ort.get_available_providers()
        self.get_logger().info(f'사용 가능한 실행 공급자: {available_providers}')
        
        # GPU 사용 가능한 경우 GPU 우선 사용 (TensorRT > CUDA > CPU)
        if 'TensorrtExecutionProvider' in available_providers:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.get_logger().info('🚀 GPU(TensorRT) 실행 공급자 사용 - 최고 성능!')
        elif 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.get_logger().info('🚀 GPU(CUDA) 실행 공급자 사용')
        else:
            providers = ['CPUExecutionProvider']
            self.get_logger().info('💻 CPU 실행 공급자 사용')
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # 실제 사용된 실행 공급자 확인
        used_providers = self.session.get_providers()
        self.get_logger().info(f'실제 사용된 실행 공급자: {used_providers}')
        self.get_logger().info(f'✅ ONNX 모델 로드 완료: {self.model_path}')
        
        # 모델 입력/출력 정보 출력
        for i, input_info in enumerate(self.session.get_inputs()):
            self.get_logger().info(f'📥 입력 {i}: {input_info.name}, 형태: {input_info.shape}, 타입: {input_info.type}')
        
        for i, output_info in enumerate(self.session.get_outputs()):
            self.get_logger().info(f'📤 출력 {i}: {output_info.name}, 형태: {output_info.shape}, 타입: {output_info.type}')
        
        # 모델이 제대로 로드되었는지 테스트
        self.get_logger().info('🧪 모델 테스트 시작...')
        try:
            # 더미 입력으로 모델 테스트
            dummy_input = np.random.randn(1, 422).astype(np.float32)
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            test_outputs = self.session.run([output_name], {input_name: dummy_input})
            test_output = test_outputs[0]
            self.get_logger().info(f'✅ 모델 테스트 성공: 출력 형태={test_output.shape}, 값={test_output}')
        except Exception as e:
            self.get_logger().error(f'❌ 모델 테스트 실패: {e}')
        
        # 센서 데이터 관리자
        self.sensor_manager = SensorDataManager()
        
        # 기준점 설정 상태
        self.reference_point_set = False
        
        # TurtleBot 스타일 스케일링 사용 여부 (필요시 True로 변경)
        self.use_turtlebot_scaling = False
        self.v_scale = 0.1    # TurtleBot linear velocity scale
        self.w_scale = -0.3   # TurtleBot angular velocity scale
        
        # ROS2 서브스크라이버
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
            self.lidar_callback,
            10
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/wamv/sensors/gps/gps/fix',
            self.gps_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/wamv/sensors/imu/imu/data',
            self.imu_callback,
            10
        )
        
        # ROS2 퍼블리셔 (스러스터 제어)
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        
        # 데이터 저장
        self.lidar_data = None
        self.gps_data = None
        self.imu_data = None
        
        # 이전 명령값 저장 (Unity: moment_input, force_input)
        self.prev_moment_input = 0.0  # 직전 moment_input (선회 명령)
        self.prev_force_input = 0.0   # 직전 force_input (전진 명령)
        
        # Stacked 데이터를 위한 히스토리 (2번의 211개 데이터)
        self.data_history = deque(maxlen=2)  # 최근 2번의 데이터 저장
        
        # 웨이포인트 설정 (기준점 (0,0) 기준 상대 좌표)
        self.waypoints = [
            [50.0, 0.0],    # 첫 번째 웨이포인트 (동쪽 50m)
            [100.0, 0.0],   # 두 번째 웨이포인트 (동쪽 100m)
            [100.0, 150.0], # 세 번째 웨이포인트 (동쪽 100m, 북쪽 150m)
        ]
        self.current_waypoint_idx = 0
        
        # 제어 타이머 (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # 데이터 수신 상태
        self.data_ready = {
            'lidar': False,
            'gps': False,
            'imu': False
        }
        
        self.get_logger().info('🚢 VRX ONNX 강화학습 제어 시스템 시작!')
        self.get_logger().info(f'📍 웨이포인트: {self.waypoints}')
        self.get_logger().info('📊 모델 입력: Stacked 2번의 211개 데이터 (LiDAR 201 + 센서 10)')
    
    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백"""
        self.lidar_data = self.sensor_manager.process_lidar_data(msg)
        self.data_ready['lidar'] = True
        
        # 디버깅 정보
        if self.lidar_data['valid_count'] > 0:
            self.get_logger().debug(f'LiDAR: {self.lidar_data["valid_count"]}개 포인트 수신')
    
    def gps_callback(self, msg):
        """GPS 데이터 콜백"""
        self.gps_data = self.sensor_manager.process_gps_data(msg)
        if self.gps_data is not None:
            self.data_ready['gps'] = True
            
            # 첫 번째 GPS 데이터 기준점 설정 확인
            if not self.reference_point_set:
                self.reference_point_set = True
                self.get_logger().info(f'📍 기준점 설정 완료: (0, 0) - 첫 번째 GPS 위치')
            
            self.get_logger().debug(f'GPS: X={self.gps_data["utm_x"]:.2f}m, Y={self.gps_data["utm_y"]:.2f}m (기준점 기준)')
    
    def imu_callback(self, msg):
        """IMU 데이터 콜백"""
        self.imu_data = self.sensor_manager.process_imu_data(msg)
        self.data_ready['imu'] = True
        
        # YAW rate 계산 (이전 값과의 차이)
        if not hasattr(self, 'prev_yaw'):
            self.prev_yaw = self.imu_data['yaw_rad']
            self.prev_time = time.time()
            self.yaw_rate = 0.0
        else:
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt > 0:
                yaw_diff = self.imu_data['yaw_rad'] - self.prev_yaw
                # 각도 차이를 -π ~ π 범위로 정규화
                while yaw_diff > np.pi:
                    yaw_diff -= 2 * np.pi
                while yaw_diff < -np.pi:
                    yaw_diff += 2 * np.pi
                
                self.yaw_rate = yaw_diff / dt
                self.prev_yaw = self.imu_data['yaw_rad']
                self.prev_time = current_time
        
        self.get_logger().debug(f'IMU: Heading={self.imu_data["yaw_degrees"]:.1f}°, YAW Rate={self.yaw_rate:.3f} rad/s')
    
    def get_next_waypoint(self):
        """다음 웨이포인트 반환"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        else:
            # 모든 웨이포인트 완료 시 마지막 웨이포인트 반환
            return self.waypoints[-1]
    
    def check_waypoint_reached(self, current_pos, waypoint, threshold=10.0):
        """웨이포인트 도달 여부 확인"""
        distance = np.sqrt((current_pos[0] - waypoint[0])**2 + (current_pos[1] - waypoint[1])**2)
        return distance < threshold
    
    def prepare_model_input(self):
        """ONNX 모델 입력 데이터 준비 (Stacked: 2번의 211개 데이터)"""
        # 안전장치 제거 - 모델 입력이 맞지 않아도 오류 발생시키지 않음
        
        # 1. LiDAR 데이터 (-100도 ~ 100도, 201개)
        lidar_ranges = self.lidar_data['ranges']
        lidar_angles = self.lidar_data['angles']
        
        # -100도 ~ 100도 범위의 LiDAR 데이터 필터링
        angle_mask = (lidar_angles >= np.radians(-100)) & (lidar_angles <= np.radians(100))
        filtered_ranges = lidar_ranges[angle_mask]
        filtered_angles = lidar_angles[angle_mask]
        
        # 201개로 리샘플링 (안전장치 제거)
        target_angles = np.linspace(np.radians(-100), np.radians(100), 201)
        lidar_input = np.interp(target_angles, filtered_angles, filtered_ranges)
        
        # 2. Heading 데이터 (도 단위)
        heading_input = self.imu_data['yaw_degrees']
        
        # 3. YAW rate (rad/s)
        yaw_rate_input = self.yaw_rate
        
        # 4. Position (x, y)
        position_input = [self.gps_data['utm_x'], self.gps_data['utm_y']]
        
        # 5. 다음 웨이포인트 (x, y)
        next_waypoint = self.get_next_waypoint()
        waypoint_input = next_waypoint
        
        # 6. 이전 명령값 2개 (Unity: moment_input, force_input)
        prev_commands_input = [self.prev_moment_input, self.prev_force_input]
        
        # Unity 관측 구조에 맞게 211개 배열로 구성
        current_data = np.concatenate([
            lidar_input,           # 201개 (LiDAR 거리)
            [heading_input],       # 1개 (Agent forward direction Y-rotation)
            [yaw_rate_input],      # 1개 (IMU Angular Velocity Heading)
            position_input,        # 2개 (현재 에이전트 위치 X, Z)
            waypoint_input,        # 2개 (현재 웨이포인트 위치 X, Z)
            [waypoint_input[0], waypoint_input[1]],  # 2개 (다음 웨이포인트 위치 X, Z - 현재와 동일)
            prev_commands_input    # 2개 (직전 명령 moment_input, force_input)
        ]).astype(np.float32)
        
        # 히스토리에 현재 데이터 추가
        self.data_history.append(current_data)
        
        # Stacked 입력 생성 (안전장치 제거)
        # 데이터가 2개 미만이어도 강제로 처리
        if len(self.data_history) >= 2:
            stacked_input = np.concatenate([
                self.data_history[0],  # 첫 번째 211개 데이터
                self.data_history[1]   # 두 번째 211개 데이터
            ]).astype(np.float32)
        else:
            # 데이터가 1개뿐이면 복제하여 사용
            stacked_input = np.concatenate([
                self.data_history[0],  # 첫 번째 211개 데이터
                self.data_history[0]   # 동일한 데이터 복제
            ]).astype(np.float32)
        
        return stacked_input.reshape(1, -1)  # 배치 차원 추가
    
    def control_loop(self):
        """메인 제어 루프"""
        # 모델 입력 데이터 준비 (안전장치 제거)
        model_input = self.prepare_model_input()
            
            # 디버깅: 모델 입력 형태 확인
        self.get_logger().info(f'모델 입력 형태: {model_input.shape}, 크기: {model_input.size}')
        self.get_logger().info(f'모델 입력 범위: min={model_input.min():.3f}, max={model_input.max():.3f}, mean={model_input.mean():.3f}')
        
        # ONNX 모델 추론
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        outputs = self.session.run([output_name], {input_name: model_input})
        model_output = outputs[0]  # 모델 출력
        
        # 모든 출력 구조 분석
        self.get_logger().info(f'전체 outputs 길이: {len(outputs)}')
        for i, output in enumerate(outputs):
            self.get_logger().info(f'output[{i}]: 형태={output.shape}, 차원={output.ndim}')
        
        # 디버깅: 모델 출력 형태 확인
        self.get_logger().debug(f'모델 출력 형태: {model_output.shape}, 차원: {model_output.ndim}')
        
        # Unity 모델 출력 해석 (다양한 출력 구조 지원)
        self.get_logger().info(f'모델 출력 상세: 형태={model_output.shape}, 차원={model_output.ndim}, 값={model_output}')
        
        # TurtleBot 스타일 직접 출력 사용
        if len(outputs) > 2:
            moment_input = outputs[2][0][0] * self.w_scale  # angular velocity
            force_input = outputs[2][0][1] * self.v_scale   # linear velocity
        else:
            moment_input = 0.0
            force_input = 0.0
        
        # Unity 범위 제한 적용 (모델 출력에 맞게 조정)
        moment_input = np.clip(moment_input, -3.0, 3.0)      # moment_input 범위 확장
        force_input = np.clip(force_input, -1.0, 1.0)        # force_input 범위 확장
        
        # TurtleBot 스타일 스케일링 적용 (선택적)
        if self.use_turtlebot_scaling:
            moment_input *= self.w_scale  # angular velocity scale
            force_input *= self.v_scale   # linear velocity scale
            self.get_logger().info(f'TurtleBot 스케일링 적용: w_scale={self.w_scale}, v_scale={self.v_scale}')
        
        # Unity 스러스터 제어 로직 적용
        left_thrust, right_thrust = self.calculate_unity_thruster_commands(moment_input, force_input)
        
        # 스러스터 명령 발행
        self.publish_thruster_commands(left_thrust, right_thrust)
        
        # 이전 명령값 업데이트 (Unity 형식)
        self.prev_moment_input = moment_input
        self.prev_force_input = force_input
        
        # 웨이포인트 도달 확인 및 디버깅 정보
        current_pos = [self.gps_data['utm_x'], self.gps_data['utm_y']]
        current_waypoint = self.get_next_waypoint()
        distance_to_waypoint = np.sqrt((current_pos[0] - current_waypoint[0])**2 + (current_pos[1] - current_waypoint[1])**2)
        
        if self.check_waypoint_reached(current_pos, current_waypoint):
            self.current_waypoint_idx += 1
            self.get_logger().info(f'🎯 웨이포인트 {self.current_waypoint_idx-1} 도달! 다음: {self.get_next_waypoint()}')
        
        self.get_logger().info(
            f'위치: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | '
            f'웨이포인트: ({current_waypoint[0]:.1f}, {current_waypoint[1]:.1f}) | '
            f'거리: {distance_to_waypoint:.1f}m | '
            f'스러스터: L={left_thrust:.1f}, R={right_thrust:.1f}'
        )
            
    
    def calculate_unity_thruster_commands(self, moment_input, force_input):
        """Unity 모델의 moment_input, force_input을 좌우 스러스터 명령으로 변환"""
        # 모델 출력 범위에 맞게 정규화 (3.0 -> 1.0으로 스케일링)
        normalized_moment = moment_input / 3.0  # -3.0~3.0 -> -1.0~1.0
        normalized_force = force_input / 1.0    # -1.0~1.0 -> -1.0~1.0
        
        # Unity 스러스터 계산 로직
        target_throttle1 = normalized_moment + normalized_force
        target_throttle2 = -normalized_moment + normalized_force
        
        # Saturation logic (Unity 코드와 동일)
        saturation = 0
        if target_throttle1 > 1.0:
            saturation = target_throttle1 - 1.0
            target_throttle1 -= saturation
            target_throttle2 -= saturation
        elif target_throttle2 < -1.0:
            saturation = target_throttle2 + 1.0
            target_throttle1 -= saturation
            target_throttle2 -= saturation
        elif target_throttle2 > 1.0:
            saturation = target_throttle2 - 1.0
            target_throttle2 -= saturation
            target_throttle1 -= saturation
        elif target_throttle1 < -1.0:
            saturation = target_throttle1 + 1.0
            target_throttle2 -= saturation
            target_throttle1 -= saturation
        
        # VRX 스러스터 출력으로 변환 (-200 ~ 200)
        left_thrust = target_throttle1 * 200.0
        right_thrust = target_throttle2 * 200.0
        
        # 최종 범위 제한
        left_thrust = np.clip(left_thrust, -200.0, 200.0)
        right_thrust = np.clip(right_thrust, -200.0, 200.0)
        
        return left_thrust, right_thrust
    
    def publish_thruster_commands(self, left_thrust, right_thrust):
        """스러스터 명령 발행"""
        left_msg = Float64()
        left_msg.data = left_thrust
        self.left_thrust_pub.publish(left_msg)
        
        right_msg = Float64()
        right_msg.data = right_thrust
        self.right_thrust_pub.publish(right_msg)
    
    def set_waypoints(self, waypoints):
        """웨이포인트 설정"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.get_logger().info(f'📍 웨이포인트 업데이트: {self.waypoints}')
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        # 스러스터 정지
        self.publish_thruster_commands(0.0, 0.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VRXONNXController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
