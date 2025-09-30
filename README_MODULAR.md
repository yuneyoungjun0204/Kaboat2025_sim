# VRX 로봇 제어 시스템 - 모듈화 버전

## 📋 개요

기존의 3개 파일(`blob_depth_detector_hybrid_multi_target_tracking.py`, `buoy_navigation_controller_hybrid.py`, `object_approach_controller.py`)을 기능별로 모듈화하여 재사용 가능한 컴포넌트로 구성한 통합 시스템입니다.

## 🏗️ 모듈 구조

### `utils/` 디렉토리
- **`depth_estimation.py`**: MiDaS Hybrid 모델을 사용한 깊이 맵 추정
- **`color_filtering.py`**: HSV 색상 공간에서 특정 색상 범위 필터링
- **`object_detection.py`**: CV 처리를 통한 이미지에서의 객체 검출
- **`object_tracking.py`**: 검출된 객체들의 추적 관리 (IMM + MMPDAM)
- **`navigation_control.py`**: PID 제어기를 통한 네비게이션 및 접근 제어
- **`thruster_control.py`**: ROS2를 통한 스러스터 명령 전송
- **`visualization.py`**: 추적 결과 및 깊이 맵 시각화
- **`trackbar_control.py`**: GUI 트랙바를 통한 실시간 파라미터 조정
- **`matplotlib_visualizer.py`**: Matplotlib 기반 배 위치 및 장애물 시각화

### 메인 파일
- **`main.py`**: 모든 모듈을 통합하여 사용하는 메인 노드 (간소화 버전)
- **`run_vrx_system.py`**: 시스템 실행을 위한 래퍼 스크립트
- **`run_matplotlib_visualization.py`**: Matplotlib 시각화 실행 스크립트
- **`simple_plot.py`**: 간단한 VRX 데이터 시각화 (LiDAR, GPS, IMU)
- **`run_simple_plot.py`**: 간단한 시각화 실행 스크립트
- **`vrx_plot_system.py`**: VRX 네비게이션 플롯 시스템 (클릭으로 waypoint 설정)

## 🚀 사용법

### 1. 새로운 통합 시스템 실행
```bash
# 방법 1: 직접 실행
vrx_main

# 방법 2: 래퍼 스크립트 사용
vrx_run

# 방법 3: Matplotlib 시각화 실행
vrx_plot

# 방법 4: 간단한 VRX 데이터 시각화 (LiDAR, GPS, IMU)
vrx_simple

# 방법 5: VRX 네비게이션 플롯 시스템 (클릭으로 waypoint 설정)
vrx_nav
```

### 2. Subplot 시각화 사용법
시스템 실행 후 다음과 같은 창이 나타납니다:

#### **VRX Robot Control System 창** (메인 시각화)
- **왼쪽**: Main Tracking View - 추적 결과, 바운딩 박스, 궤적
- **오른쪽**: Depth Map View - MiDaS 깊이 맵 (TURBO 컬러맵)
- **상단**: 제목 바 - 현재 제어 모드 및 시스템 정보
- **크기**: 1300x540 픽셀

#### **VRX Control Panel 창** (파라미터 조정)
- 15개 트랙바로 실시간 파라미터 조정
- 카테고리별로 그룹화된 파라미터들
- **크기**: 400x600 픽셀

### 3. Matplotlib 시각화 사용법
`vrx_plot` 명령어로 실행하면 다음과 같은 시각화가 나타납니다:

#### **VRX Robot Navigation Visualization 창** (Matplotlib)
- **왼쪽 subplot**: 직교좌표계 (Cartesian)
  - 배의 현재 위치 (파란색 점)
  - 배의 헤딩 방향 (파란색 화살표)
  - 배의 이동 궤적 (파란색 선)
  - 주변 장애물 위치 (빨간색/초록색 점)
  - 장애물 정보 (타입, 깊이)

- **오른쪽 subplot**: 원형좌표계 (Polar)
  - 현재 헤딩 방향 (빨간색 화살표)
  - 목표 헤딩 방향 (초록색 화살표)
  - 장애물의 상대적 방향 (색상별 화살표)
  - 방향 표시 (N, E, S, W)
  - 거리별 원형 그리드

#### **실시간 업데이트**
- 100ms 간격으로 자동 업데이트
- ROS2 토픽에서 실시간 데이터 수신
- 애니메이션으로 부드러운 시각화

### 4. 간단한 VRX 데이터 시각화 사용법
`vrx_simple` 명령어로 실행하면 다음과 같은 시각화가 나타납니다:

#### **VRX Simple Data Visualization 창** (4개 Subplot)
- **상단 왼쪽**: LiDAR 스캔 데이터
  - 거리별 스캔 포인트 표시
  - 실시간 거리 정보 업데이트
  - 유효한 범위만 필터링하여 표시

- **상단 오른쪽**: GPS 위치 정보
  - 위도/경도 좌표 표시
  - 실시간 위치 마커
  - 자동 축 범위 조정

- **하단 왼쪽**: IMU 헤딩 정보
  - Yaw 각도 실시간 표시
  - 시간에 따른 헤딩 변화
  - 최근 100개 포인트 표시

- **하단 오른쪽**: 위치 궤적 추적
  - GPS 기반 이동 경로
  - 현재 위치 마커
  - 실시간 궤적 업데이트

#### **사용하는 ROS2 토픽**
- `/wamv/sensors/lidars/lidar_wamv_sensor/scan` (LaserScan)
- `/wamv/sensors/gps/gps/fix` (NavSatFix)
- `/wamv/sensors/imu/imu/data` (Imu)

#### **특징**
- **안정성**: 스레드 안전한 데이터 접근
- **실시간**: 100ms 간격 자동 업데이트
- **간단함**: 복잡한 이미지 처리 없이 기본 센서 데이터만 사용
- **4개 뷰**: LiDAR, GPS, IMU, 궤적을 동시에 모니터링

### 5. VRX 네비게이션 플롯 시스템 사용법
`vrx_nav` 명령어로 실행하면 다음과 같은 고급 네비게이션 시각화가 나타납니다:

#### **VRX Robot Navigation System 창**
- **LiDAR 데이터**: 파란색 점으로 주변 장애물 표시
- **로봇 위치**: 빨간색 점으로 현재 위치 표시
- **현재 헤딩**: 초록색 화살표로 로봇의 현재 방향 표시
- **Waypoint**: 클릭으로 설정한 목표 지점 (주황색 별)
- **비용 함수**: 보라색 선으로 각 방향의 비용 표시
- **최적 헤딩**: 노란색 선으로 계산된 최적 방향 표시
- **목표까지의 경로**: 주황색 점선으로 목표까지의 직선 경로

#### **클릭 인터페이스**
- **마우스 클릭**: 지도에서 클릭하면 waypoint 설정 팝업 표시
- **OK 버튼**: 클릭한 위치를 waypoint로 설정
- **Cancel 버튼**: waypoint 설정 취소
- **Reset 버튼**: 모든 waypoint 초기화

#### **비용 함수 및 경로 계획**
- **안전 구역 계산**: 5m 이내 위험, 10m 이내 주의 구역
- **비용 함수**: 거리, 안전성, 목표 방향을 고려한 비용 계산
- **최적 헤딩**: 비용이 최소인 방향으로 자동 계산
- **실시간 업데이트**: 100ms마다 비용 함수 및 최적 경로 재계산

#### **사용하는 ROS2 토픽**
- **서브스크라이브**:
  - `/wamv/sensors/lidars/lidar_wamv_sensor/scan` (LaserScan) - LiDAR 데이터
  - `/wamv/sensors/gps/gps/fix` (NavSatFix) - GPS 위치
  - `/wamv/sensors/imu/imu/data` (Imu) - IMU 센서 데이터
- **퍼블리시**:
  - `/vrx/waypoint` (PointStamped) - 설정된 waypoint

#### **특징**
- **인터랙티브**: 마우스 클릭으로 직관적인 waypoint 설정
- **실시간 경로 계획**: 비용 함수 기반 최적 경로 계산
- **안전성 고려**: 장애물 회피를 위한 안전 구역 계산
- **시각적 피드백**: 모든 정보를 한 화면에서 확인 가능

### 6. 기존 단축어 (참고용)
```bash
vrx2        # 다중 표적 추적 시스템 (이전 버전)
vrxcontrol  # 부표 네비게이션 컨트롤러 (이전 버전)
approach    # 객체 접근 컨트롤러 (이전 버전)
```

## 🎛️ 제어 모드

### Navigation 모드 (부표 간 네비게이션)
- 두 개의 부표(빨간색, 초록색) 사이를 통과하는 네비게이션
- 부표의 중점을 이미지 중앙에 위치시키도록 제어
- 양쪽 부표가 모두 탐지되어야 활성화

### Approach 모드 (객체 접근 및 회전)
- 선택된 색상의 부표에 접근
- 일정 거리에서 고깔 회전 수행
- 시계방향/반시계방향 선택 가능

## 🎚️ 트랙바 컨트롤 (확장된 버전)

### 제어 모드
- **Control_Mode**: Navigation(0) / Approach(1)
- **Target_Color**: Red(0) / Green(1) 
- **Rotation_Direction**: 시계방향(0) / 반시계방향(1)

### 탐지 파라미터
- **Min_Depth_Threshold**: 최소 깊이 임계값 (0.001-0.1)
- **Max_Depth_Threshold**: 최대 깊이 임계값 (0.001-0.5)

### Blob Detector 파라미터
- **Min_Area**: 최소 면적 (0-5000)
- **Max_Area**: 최대 면적 (0-50000)
- **Min_Circularity**: 최소 원형도 (0.0-1.0)

### 추적 파라미터
- **Max_Tracks**: 최대 트랙 수 (1-20)
- **Max_Missed_Frames**: 최대 누락 프레임 (1-20)
- **Gate_Threshold**: 게이트 임계값 (10-200)
- **Min_Association_Prob**: 최소 연결 확률 (0.0-1.0)

### 제어 파라미터
- **Max_Speed**: 최대 속도 (0-2000)
- **Min_Speed**: 최소 속도 (0-1000)
- **Base_Speed**: 기본 속도 (0-500)
- **Max_Turn_Thrust**: 최대 회전 추력 (10-250)

### PID 파라미터
- **Steering_Kp**: 조향 PID Kp (0.1-5.0)
- **Approach_Kp**: 접근 PID Kp (0.1-5.0)

### 시각화
- **Show_Depth**: 깊이 맵 표시 On/Off (Subplot 형태로 메인 창에 통합)

## 📡 ROS2 토픽

### 퍼블리시
- `/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions` (Float32MultiArray)
- `/detection/status` (String)
- `/wamv/thrusters/left/thrust` (Float64)
- `/wamv/thrusters/right/thrust` (Float64)
- `/approach/target_x` (Float64)

### 서브스크라이브
- `/wamv/sensors/cameras/front_left_camera_sensor/image_raw` (Image)
- `/wamv/pose` (PoseStamped) - **Matplotlib 시각화용**
- `/blob_depth_detector_hybrid_multi_target/buoy_tracking/positions` (Float32MultiArray) - **Matplotlib 시각화용**

## 🔧 모듈별 기능

### 1. 깊이 추정 (MiDaSHybridDepthEstimator)
- Intel MiDaS Hybrid 모델 사용
- 실시간 깊이 맵 생성
- GPU/CPU 자동 감지

### 2. 색상 필터링 (ColorFilter)
- HSV 색상 공간에서 빨간색/초록색 마스크 생성
- 색상 범위 동적 조정 가능

### 3. 객체 검출 (BlobDetector)
- 색상 마스크 + 형태 분석
- 깊이 기반 필터링
- 원형도 기반 유효성 검사

### 4. 객체 추적 (MultiTargetTracker)
- IMM (Interacting Multiple Model) 알고리즘
- MMPDAM (Multi-Model Probabilistic Data Association Method)
- 신뢰도 기반 최적 트랙 선택

### 5. 네비게이션 제어 (NavigationController)
- PID 제어기를 통한 정밀 제어
- 적응적 속도 제어
- 회전 방향별 목표 위치 계산

### 6. 스러스터 제어 (ThrusterController)
- ROS2를 통한 스러스터 명령 전송
- 비상 정지 기능
- 상태 정보 퍼블리시

### 7. 시각화 (Visualizer) - **NEW Subplot 기능**
- **Subplot 통합**: 메인 추적과 깊이 맵을 하나의 창에서 나란히 표시
- **자동 창 배치**: 최적화된 창 위치 및 크기 설정
- **실시간 정보 표시**: 제어 모드, 트랙 정보, 통계 등
- **깔끔한 UI**: 제목 바, 구분선, 테두리로 가독성 향상

### 8. 트랙바 제어 (TrackbarController)
- 15개 파라미터를 통합 관리
- 실시간 파라미터 조정
- 카테고리별 파라미터 그룹화

### 9. Matplotlib 시각화 (MatplotlibVisualizer) - **NEW**
- **실시간 배 위치 추적**: ROS2 `/wamv/pose` 토픽에서 위치 및 헤딩 정보 수신
- **장애물 정보 시각화**: 추적된 부표의 위치와 깊이 정보 표시
- **이중 좌표계**: 직교좌표계와 원형좌표계 동시 표시
- **실시간 애니메이션**: 100ms 간격으로 부드러운 업데이트
- **궤적 표시**: 배의 이동 경로 히스토리 표시
- **상대적 방향**: 장애물의 배 기준 상대적 방향 표시

## 🆚 기존 시스템과의 차이점

### 장점
- ✅ **완전한 모듈화**: 모든 기능이 독립적인 모듈로 분리
- ✅ **간소화된 main.py**: 핵심 로직만 유지하여 가독성 향상
- ✅ **통합 트랙바**: 모든 파라미터를 하나의 창에서 조정
- ✅ **Subplot 시각화**: 메인 추적과 깊이 맵을 하나의 창에서 나란히 표시
- ✅ **독립적 스러스터 제어**: ROS2 명령 전송 모듈화
- ✅ **실시간 파라미터 조정**: 15개 이상의 파라미터 실시간 조정
- ✅ **기능별 독립적 테스트**: 각 모듈을 개별적으로 테스트 가능
- ✅ **최적화된 창 배치**: 자동 창 위치 및 크기 설정

### main.py 간소화 결과
- **이전**: 900+ 라인의 복잡한 단일 파일
- **현재**: 150 라인의 간결한 통합 파일
- **모듈 수**: 8개의 독립적인 기능 모듈
- **트랙바**: 15개 파라미터를 통합 관리

### 호환성
- ✅ 기존 ROS2 토픽 구조 유지
- ✅ 기존 단축어 호환
- ✅ 동일한 시뮬레이션 환경 지원

## 🐛 문제 해결

### Import 오류
```bash
# 가상환경 활성화 확인
source vrx_env/bin/activate

# 의존성 설치 확인
pip install torch torchvision opencv-python numpy rclpy
```

### 시뮬레이션 연결 오류
```bash
# ROS2 환경 확인
echo $ROS_DOMAIN_ID
source /opt/ros/humble/setup.bash
```

### 성능 문제
- 트랙바에서 깊이 임계값 조정
- 탐지 파라미터 최적화
- GPU 사용 여부 확인

## 📝 개발 노트

### 모듈 추가 시
1. `utils/` 디렉토리에 새 모듈 추가
2. `utils/__init__.py`에 import 추가
3. `main.py`에서 통합 사용

### 파라미터 튜닝
- 각 모듈의 파라미터는 독립적으로 조정 가능
- 트랙바를 통한 실시간 조정 지원
- 설정값 저장/로드 기능 추가 가능
