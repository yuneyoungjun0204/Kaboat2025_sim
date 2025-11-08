# 이미지 처리 모듈 테스트 스크립트

Main_MCP.py에서 사용하는 이미지 처리 기능들을 개별적으로 테스트하기 위한 스크립트 모음입니다.

## 📋 테스트 스크립트 목록

### 1. test_depth_estimation.py
MiDaS Hybrid 모델을 사용한 깊이 추정 테스트

**기능:**
- 카메라 이미지에서 깊이 맵 생성
- 다양한 컬러맵으로 깊이 시각화
- 원본 이미지와 깊이 맵 블렌딩

**트랙바 파라미터:**
- `Colormap (0-20)`: 깊이 맵 컬러맵 선택 (INFERNO, VIRIDIS, JET 등)
- `Blend Alpha (0-100%)`: 원본 이미지와 깊이 맵 블렌딩 비율
- `Min Depth (0-100%)`: 표시할 최소 깊이 범위
- `Max Depth (0-100%)`: 표시할 최대 깊이 범위

**실행 방법:**
```bash
cd /home/yuneyoungjun/vrx_ws
source install/setup.bash
python3 src/vrx/kaboat_backup/test/test_depth_estimation.py
```

---

### 2. test_object_detection.py
NanoOWL을 사용한 객체 탐지 테스트 (깊이 필터링 없음)

**기능:**
- VLM 기반 부표 탐지 (빨강/초록 원뿔 부표, 파란 부표)
- 바운딩 박스 및 신뢰도 표시
- 박스 크기 필터링

**트랙바 파라미터:**
- `Mission Type (0-1)`:
  - 0 = Pass Between Buoys (빨강/초록 부표 탐지)
  - 1 = Circle Buoy (파란 부표 탐지)
- `Threshold (0-100)`: 탐지 임계값 (실제 값 = 값/100, 기본 3 = 0.03)
- `Min Box Area (0-5000)`: 최소 박스 면적 (픽셀²)
- `Max Box Area (0-2000 x100)`: 최대 박스 면적 (실제 값 = 값x100)

**실행 방법:**
```bash
cd /home/yuneyoungjun/vrx_ws
source install/setup.bash
python3 src/vrx/kaboat_backup/test/test_object_detection.py
```

---

### 3. test_detection_with_depth.py
NanoOWL + MiDaS 통합 테스트 (Main_MCP.py와 동일한 탐지 시스템)

**기능:**
- 객체 탐지 + 깊이 필터링 통합
- 탐지된 객체의 깊이 정보 표시
- 깊이 범위로 객체 필터링

**트랙바 파라미터:**
- `Mission Type (0-1)`: 미션 타입 선택
  - 0 = Pass Between Buoys
  - 1 = Circle Buoy
- `Threshold (0-100)`: 탐지 임계값
- `Min Box Area (0-5000)`: 최소 박스 면적
- `Max Box Area (0-2000 x100)`: 최대 박스 면적
- `Min Depth (0-100m)`: 최소 깊이 (미터)
- `Max Depth (0-100m)`: 최대 깊이 (미터)
- `Show Depth (0-1)`: 깊이 맵 표시 여부

**실행 방법:**
```bash
cd /home/yuneyoungjun/vrx_ws
source install/setup.bash
python3 src/vrx/kaboat_backup/test/test_detection_with_depth.py
```

---

## 🎮 사용 방법

1. **시뮬레이터 실행**: VRX 시뮬레이터가 실행 중이어야 합니다.

2. **카메라 토픽 확인**:
   ```bash
   ros2 topic list | grep camera
   # /wamv/sensors/cameras/front_left_camera/image_raw
   ```

3. **테스트 스크립트 실행**: 위의 실행 방법 참조

4. **트랙바 조정**: OpenCV 창에서 트랙바를 조정하여 실시간으로 파라미터 변경

5. **종료**: `Ctrl+C` 또는 OpenCV 창 닫기

---

## 📊 출력 형식

### test_depth_estimation.py
- 좌측: 원본 이미지와 깊이 맵 블렌딩
- 우측: 깊이 맵 (컬러맵 적용)

### test_object_detection.py
- 탐지된 객체에 바운딩 박스, 레이블, 신뢰도 표시

### test_detection_with_depth.py
- 좌측: 탐지 결과 (바운딩 박스, 레이블, 신뢰도, 깊이)
- 우측: 깊이 맵 (Show Depth=1일 때)

---

## 🔧 트러블슈팅

### CUDA 오류
- GPU 메모리 부족 시: 한 번에 하나의 테스트 스크립트만 실행
- CUDA 사용 불가 시: 코드에서 `device="cuda"` → `device="cpu"` 변경

### 카메라 토픽 없음
```bash
# 시뮬레이터 실행 확인
ros2 topic list

# 카메라 토픽이 없으면 시뮬레이터 재시작
```

### 모델 로딩 실패
```bash
# NanoOWL 경로 확인
ls /home/yuneyoungjun/nanoowl

# MiDaS 모델은 자동 다운로드됨 (처음 실행 시 시간 소요)
```

---

## 📝 주요 파라미터 권장 값

### 깊이 추정
- **Colormap**: 11 (INFERNO) - 깊이 시각화에 적합
- **Blend Alpha**: 50% - 원본과 깊이 맵을 균형있게 표시

### 객체 탐지
- **Threshold**: 3-5 (0.03-0.05) - 너무 낮으면 오탐지 증가
- **Min Box Area**: 500 - 너무 작은 노이즈 제거
- **Max Box Area**: 80000 - 화면 전체를 차지하는 오탐지 제거

### 깊이 필터링
- **Min Depth**: 0m - 최소값
- **Max Depth**: 30-50m - 원거리 객체 필터링 (시뮬레이터 환경에 따라 조정)

---

## 🎯 활용 방법

1. **파라미터 최적화**: 트랙바로 실시간 조정하여 최적 값 찾기
2. **미션별 테스트**: Mission Type을 변경하여 각 미션에 맞는 탐지 성능 확인
3. **깊이 필터링 효과**: 깊이 범위를 조정하여 원거리/근거리 객체 필터링 효과 확인
4. **성능 분석**: 탐지 속도 및 정확도 분석

---

## 📚 관련 파일

- `utils/detection_system.py`: DetectionSystem 클래스 (Main_MCP.py에서 사용)
- `utils/depth_estimation.py`: MiDaSHybridDepthEstimator 클래스
- `Main_MCP.py`: 전체 시스템 통합 코드
