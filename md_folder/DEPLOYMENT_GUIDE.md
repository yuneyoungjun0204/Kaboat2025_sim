# VRX 자율주행 시스템 배포 가이드

이 가이드는 완전히 깨끗한 컴퓨터(Ubuntu 22.04)에서 VRX 자율주행 시스템을 처음부터 설치하고 실제 보트에서 테스트하기 위한 완전한 단계별 가이드입니다.

## 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [기본 환경 설치](#기본-환경-설치)
3. [ROS2 Humble 설치](#ros2-humble-설치)
4. [Python 환경 설정](#python-환경-설정)
5. [딥러닝 프레임워크 설치](#딥러닝-프레임워크-설치)
6. [NanoOWL 설치](#nanoowl-설치)
7. [MiDaS Hybrid 설정](#midas-hybrid-설정)
8. [VRX 워크스페이스 설정](#vrx-워크스페이스-설정)
9. [시스템 검증](#시스템-검증)
10. [실행 방법](#실행-방법)
11. [문제 해결](#문제-해결)

---

## 시스템 요구사항

### 하드웨어
- **CPU**: 4코어 이상 권장
- **RAM**: 8GB 이상 (16GB 권장)
- **GPU**: NVIDIA GPU (CUDA 지원) 권장
  - NanoOWL 및 MiDaS는 GPU 없이도 작동하지만 실시간 성능을 위해 GPU 필요
  - 최소 4GB VRAM 권장
- **저장공간**: 20GB 이상 여유 공간

### 소프트웨어
- **OS**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **ROS2**: Humble Hawksbill
- **Python**: 3.10+
- **CUDA**: 11.8+ (GPU 사용 시)

---

## 기본 환경 설치

### 1. 시스템 업데이트

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 필수 도구 설치

```bash
# 개발 도구
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    software-properties-common

# Python 개발 도구
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv
```

### 3. NVIDIA GPU 드라이버 설치 (GPU 사용 시)

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# 설치되지 않았다면:
sudo ubuntu-drivers autoinstall
sudo reboot

# 재부팅 후 확인
nvidia-smi
```

### 4. CUDA Toolkit 설치 (GPU 사용 시)

```bash
# CUDA 11.8 설치 (PyTorch 2.2.2 호환)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-11-8

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# CUDA 설치 확인
nvcc --version
```

---

## ROS2 Humble 설치

### 1. ROS2 저장소 추가

```bash
# UTF-8 로케일 설정
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS2 GPG 키 추가
sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# ROS2 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
```

### 2. ROS2 Humble 설치

```bash
# ROS2 Humble Desktop (전체 설치)
sudo apt install -y ros-humble-desktop

# 개발 도구
sudo apt install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep

# rosdep 초기화
sudo rosdep init
rosdep update
```

### 3. ROS2 환경 설정

```bash
# ROS2 환경 자동 로드
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 설치 확인
ros2 --help
```

### 4. 필수 ROS2 패키지 설치

```bash
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-ros
```

---

## Python 환경 설정

### 1. Python 가상환경 생성 (선택사항이지만 권장)

```bash
# 가상환경 생성
python3 -m venv ~/vrx_env

# 가상환경 활성화
source ~/vrx_env/bin/activate

# 가상환경 자동 활성화 설정 (선택)
echo "source ~/vrx_env/bin/activate" >> ~/.bashrc
```

### 2. pip 업그레이드

```bash
pip install --upgrade pip setuptools wheel
```

---

## 딥러닝 프레임워크 설치

### 1. PyTorch 설치 (CUDA 지원)

GPU가 있는 경우:
```bash
pip3 install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

CPU만 사용하는 경우:
```bash
pip3 install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

### 2. PyTorch 설치 확인

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

예상 출력:
```
PyTorch version: 2.2.2
CUDA available: True  # GPU가 있고 올바르게 설정된 경우
```

---

## NanoOWL 설치

NanoOWL은 CLIP 기반 제로샷 객체 탐지를 위한 경량 모델입니다.

### 1. NanoOWL 클론

```bash
# VRX 환경 디렉토리 생성
mkdir -p ~/vrx_ws/src/vrx/vrx_env
cd ~/vrx_ws/src/vrx/vrx_env

# NanoOWL 클론
git clone https://github.com/NVIDIA-AI-IOT/nanoowl.git
cd nanoowl
```

### 2. NanoOWL 종속성 설치

```bash
# NanoOWL 요구사항 설치
pip3 install -e .

# 또는 수동으로:
pip3 install transformers==4.44.2
pip3 install huggingface-hub==0.24.6
pip3 install tokenizers==0.19.1
pip3 install safetensors==0.4.5
```

### 3. NanoOWL 모델 다운로드

```bash
# Python에서 모델 자동 다운로드 (최초 실행 시)
python3 -c "
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('google/owlvit-base-patch32')
tokenizer = AutoTokenizer.from_pretrained('google/owlvit-base-patch32')
print('NanoOWL models downloaded successfully!')
"
```

### 4. NanoOWL 설치 확인

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/$USER/vrx_ws/src/vrx/vrx_env/nanoowl')
from nanoowl.owl_predictor import OwlPredictor
print('NanoOWL import successful!')
"
```

---

## MiDaS Hybrid 설정

MiDaS는 단안 카메라 깊이 추정을 위한 모델입니다. PyTorch Hub를 통해 자동으로 다운로드됩니다.

### 1. MiDaS 종속성 설치

```bash
pip3 install timm==0.9.16
```

### 2. MiDaS 모델 사전 다운로드 (선택사항)

```bash
python3 -c "
import torch
# MiDaS DPT_Hybrid 모델 다운로드
model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
print('MiDaS Hybrid model downloaded successfully!')
"
```

모델은 `~/.cache/torch/hub/` 에 저장됩니다.

### 3. MiDaS 설치 확인

```bash
python3 -c "
import torch
model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
print(f'MiDaS model loaded: {type(model).__name__}')
"
```

---

## VRX 워크스페이스 설정

### 1. 워크스페이스 구조 생성

```bash
# 워크스페이스 디렉토리
mkdir -p ~/vrx_ws/src/vrx
cd ~/vrx_ws/src/vrx
```

### 2. Scripts_git 클론 또는 복사

기존 저장소에서 클론:
```bash
cd ~/vrx_ws/src/vrx
git clone <YOUR_REPOSITORY_URL> Scripts_git
cd Scripts_git
```

또는 파일을 직접 복사:
```bash
# 로컬 머신에서 원격 서버로 복사
scp -r /path/to/Scripts_git username@remote:/home/yuneyoungjun/vrx_ws/src/vrx/
```

### 3. Python 종속성 설치

```bash
cd ~/vrx_ws/src/vrx/Scripts_git
pip3 install -r requirements.txt
```

requirements.txt 내용:
- numpy==1.26.4
- opencv-python==4.10.0.84
- pillow==10.4.0
- torch==2.2.2 (이미 설치됨)
- torchvision==0.17.2 (이미 설치됨)
- timm==0.9.16
- onnxruntime==1.18.0 (또는 onnxruntime-gpu)
- transformers==4.44.2
- huggingface-hub==0.24.6
- tokenizers==0.19.1
- safetensors==0.4.5
- requests==2.32.3
- filelock==3.15.4
- urllib3==2.2.2
- regex==2024.9.11
- tqdm==4.66.5

### 4. ONNX 모델 설정

ONNX 모델이 올바른 위치에 있는지 확인:

```bash
# 모델 디렉토리 확인
ls -la ~/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/

# Ray.onnx 파일이 있어야 함
# 없다면 모델 파일을 복사하거나 다운로드
```

모델 파일이 없다면:
```bash
# 로컬에서 복사
scp /path/to/Ray.onnx username@remote:~/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/
```

### 5. 경로 설정 확인

`utils/config.py` 파일에서 경로가 올바른지 확인:

```bash
cd ~/vrx_ws/src/vrx/Scripts_git
python3 -c "
from utils.config import Constants
print(f'NanoOWL 경로: {Constants.Paths.NANOOWL_DIR}')
print(f'ONNX 모델 경로: {Constants.Paths.get_onnx_model_path()}')
print('✓ 경로 설정 확인 완료')
"
```

---

## 시스템 검증

### 1. Python 모듈 Import 테스트

```bash
cd ~/vrx_ws/src/vrx/Scripts_git
python3 -c "
import numpy as np
import cv2
import torch
import onnxruntime
from utils.config import Constants
from utils.detection_system import DetectionSystem, MissionType
from utils.imm_pdaf_tracker import create_tracker
from utils.mission_strategies_new import MissionManager
from utils.waypoint_manager import WaypointManager
print('✓ 모든 모듈 import 성공!')
"
```

### 2. GPU 사용 가능 여부 확인

```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 3. ONNX Runtime 테스트

```bash
python3 -c "
import onnxruntime as ort
print(f'ONNX Runtime version: {ort.__version__}')
print(f'Available providers: {ort.get_available_providers()}')
# CUDAExecutionProvider가 있으면 GPU 사용 가능
"
```

### 4. NanoOWL 테스트

```bash
python3 -c "
import sys
from utils.config import Constants
sys.path.insert(0, str(Constants.Paths.NANOOWL_DIR))
from nanoowl.owl_predictor import OwlPredictor
predictor = OwlPredictor('google/owlvit-base-patch32')
print('✓ NanoOWL 로드 성공!')
"
```

### 5. MiDaS 테스트

```bash
python3 -c "
from utils.depth_estimation import MiDaSHybridDepthEstimator
import numpy as np
estimator = MiDaSHybridDepthEstimator()
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
depth = estimator.estimate(dummy_image)
print(f'✓ MiDaS 깊이 추정 성공! 출력 shape: {depth.shape}')
"
```

---

## 실행 방법

### 1. ROS2 환경 소싱

```bash
source /opt/ros/humble/setup.bash
```

### 2. Main_MCP.py 실행

통합 미션 제어 시스템:

```bash
cd ~/vrx_ws/src/vrx/Scripts_git
python3 Main_MCP.py
```

예상 출력:
```
================================================================================
VRX 통합 미션 제어 시스템 초기화
================================================================================
[INFO] [vrx_mission_controller]: VRXSystemFactory 초기화...
[INFO] [vrx_mission_controller]: ✓ 깊이 추정기 생성 완료
[INFO] [vrx_mission_controller]: ✓ 탐지 시스템 생성 완료
...
[INFO] [vrx_mission_controller]: ✓ 초기화 완료!
```

### 3. 시각화 실행 (별도 터미널)

궤적 및 센서 데이터 시각화:

```bash
cd ~/vrx_ws/src/vrx/Scripts_git
python3 trajectory_viz.py
```

### 4. 웨이포인트 발행 (테스트용)

```bash
# 웨이포인트 발행 (x=100.0, y=50.0)
ros2 topic pub --once /vrx/waypoint geometry_msgs/msg/Point "{x: 100.0, y: 50.0, z: 0.0}"
```

### 5. 시스템 모니터링

ROS2 토픽 모니터링:

```bash
# 추력 명령 확인
ros2 topic echo /wamv/thrusters/left/thrust
ros2 topic echo /wamv/thrusters/right/thrust

# 미션 상태 확인
ros2 topic echo /vrx/mission_status

# 탐지 결과 확인
ros2 topic echo /vrx/detections

# 모든 토픽 목록
ros2 topic list
```

### 6. 실제 보트 테스트

실제 보트에서 테스트할 때:

1. **하드웨어 연결 확인**
   - LiDAR 센서 연결 및 토픽 발행 확인: `/wamv/sensors/lidars/lidar_wamv_sensor/scan`
   - GPS 연결 확인: `/wamv/sensors/gps/gps/fix`
   - IMU 연결 확인: `/wamv/sensors/imu/imu/data`
   - 카메라 연결 확인: `/wamv/sensors/cameras/front_left_camera_sensor/image_raw`
   - 추력기 제어 토픽: `/wamv/thrusters/left/thrust`, `/wamv/thrusters/right/thrust`

2. **센서 데이터 확인**
   ```bash
   # 각 센서 토픽이 발행되는지 확인
   ros2 topic hz /wamv/sensors/lidars/lidar_wamv_sensor/scan
   ros2 topic hz /wamv/sensors/gps/gps/fix
   ros2 topic hz /wamv/sensors/imu/imu/data
   ```

3. **제어 시스템 시작**
   ```bash
   cd ~/vrx_ws/src/vrx/Scripts_git
   python3 Main_MCP.py
   ```

4. **안전 모니터링**
   - 초기에는 낮은 추력으로 테스트 (`thrust_scale` 조정)
   - 비상 정지 메커니즘 준비
   - 수동 제어로 전환 가능한 상태 유지

---

## 문제 해결

### 문제 1: ROS2 토픽이 보이지 않음

```bash
# ROS2 데몬 재시작
ros2 daemon stop
ros2 daemon start

# 환경 변수 확인
echo $ROS_DOMAIN_ID
echo $ROS_LOCALHOST_ONLY
```

### 문제 2: CUDA out of memory

```python
# utils/config.py에서 배치 크기 조정
class Constants:
    DETECTION_BATCH_SIZE = 1  # 더 작게
```

또는 CPU 모드로 전환:
```python
# utils/config.py
class Constants:
    DEVICE = 'cpu'  # 'cuda' 대신
```

### 문제 3: NanoOWL import 오류

```bash
# 경로 확인
python3 -c "
from utils.config import Constants
print(Constants.Paths.NANOOWL_DIR)
import os
print(os.path.exists(Constants.Paths.NANOOWL_DIR))
"

# NanoOWL 재설치
cd ~/vrx_ws/src/vrx/vrx_env/nanoowl
pip3 install -e . --force-reinstall
```

### 문제 4: ONNX 모델 로드 실패

```bash
# 모델 파일 확인
ls -lh ~/vrx_ws/src/vrx/Scripts_git/models/correct_IMU/gpu/Ray.onnx

# 모델 경로 수동 지정
python3 -c "
from utils.config import Constants
print(Constants.Paths.get_onnx_model_path())
"
```

### 문제 5: cv_bridge 오류

```bash
# cv_bridge는 pip가 아닌 apt로 설치해야 함
sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv

# ROS2 환경 다시 소싱
source /opt/ros/humble/setup.bash
```

### 문제 6: Python 버전 충돌

```bash
# Python 버전 확인
python3 --version

# ROS2 Humble은 Python 3.10을 사용
# 가상환경 사용 시 주의
```

### 문제 7: LiDAR 데이터 수신 안 됨

```bash
# LiDAR 토픽 확인
ros2 topic list | grep lidar
ros2 topic info /wamv/sensors/lidars/lidar_wamv_sensor/scan
ros2 topic echo /wamv/sensors/lidars/lidar_wamv_sensor/scan --once

# 토픽 이름이 다른 경우 utils/config.py에서 수정
```

### 문제 8: IMU 헤딩이 이상함

```bash
# IMU 데이터 확인
ros2 topic echo /wamv/sensors/imu/imu/data

# Quaternion → Euler 변환 확인
python3 -c "
from utils.sensor_preprocessing import IMUProcessor
processor = IMUProcessor()
# 테스트 quaternion
quat = [0, 0, 0, 1]  # w, x, y, z
heading = processor.process_imu(quat)
print(f'Heading: {heading}')
"
```

### 문제 9: 추력기 명령이 안 나감

```bash
# 추력기 토픽 수동 발행 테스트
ros2 topic pub /wamv/thrusters/left/thrust std_msgs/msg/Float64 "{data: 100.0}"
ros2 topic pub /wamv/thrusters/right/thrust std_msgs/msg/Float64 "{data: 100.0}"

# 토픽이 발행되는지 확인
ros2 topic hz /wamv/thrusters/left/thrust
```

### 문제 10: 메모리 부족

```bash
# 시스템 리소스 확인
free -h
nvidia-smi  # GPU 메모리

# 불필요한 프로세스 종료
# 또는 모델을 CPU 모드로 실행
```

---

## 성능 최적화

### GPU 사용률 극대화

1. **CUDA 메모리 캐싱**
   ```python
   # utils/config.py
   import torch
   torch.backends.cudnn.benchmark = True
   ```

2. **Mixed Precision Training**
   ```python
   # NanoOWL 추론 시 fp16 사용
   model.half()
   ```

### 실시간 성능 확보

1. **타이머 주기 조정**
   ```python
   # utils/config.py
   class Constants:
       MAIN_LOOP_PERIOD = 0.01  # 100Hz (기본값)
       # 성능이 부족하면 0.02 (50Hz)로 조정
   ```

2. **탐지 주기 조정**
   ```python
   class Constants:
       DETECTION_PERIOD = 0.1  # 10Hz (기본값)
       # 더 빠른 반응이 필요하면 0.05 (20Hz)
   ```

---

## 체크리스트

배포 전 확인사항:

- [ ] Ubuntu 22.04 LTS 설치 완료
- [ ] ROS2 Humble 설치 및 환경 소싱
- [ ] NVIDIA 드라이버 및 CUDA 설치 (GPU 사용 시)
- [ ] Python 3.10+ 설치
- [ ] PyTorch 2.2.2 설치 및 CUDA 확인
- [ ] NanoOWL 클론 및 설치
- [ ] MiDaS 모델 다운로드
- [ ] VRX 워크스페이스 설정
- [ ] requirements.txt 종속성 설치
- [ ] ONNX 모델 파일 존재 확인
- [ ] 모든 Python 모듈 import 테스트
- [ ] ROS2 토픽 통신 확인
- [ ] 센서 데이터 수신 확인 (실제 보트)
- [ ] 추력기 명령 발행 확인 (실제 보트)
- [ ] trajectory_viz.py 시각화 테스트
- [ ] Main_MCP.py 실행 및 로그 확인
- [ ] 안전 메커니즘 준비 (비상 정지 등)

---

## 추가 리소스

### 공식 문서
- [ROS2 Humble 문서](https://docs.ros.org/en/humble/)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [NanoOWL GitHub](https://github.com/NVIDIA-AI-IOT/nanoowl)
- [MiDaS GitHub](https://github.com/isl-org/MiDaS)

### 프로젝트 문서
- `CLAUDE.md`: 프로젝트 개요 및 아키텍처
- `MAIN_MCP_REFACTORING.md`: 리팩토링 가이드
- `COORDINATE_SYSTEMS.md`: 좌표계 설명
- `PERFORMANCE_OPTIMIZATION.md`: 성능 최적화 가이드

### 지원
- GitHub Issues: <프로젝트 저장소 URL>
- 문의: <연락처>

---

**배포 가이드 버전**: 1.0
**최종 업데이트**: 2025-01-29
**테스트 환경**: Ubuntu 22.04, ROS2 Humble, Python 3.10, CUDA 11.8
