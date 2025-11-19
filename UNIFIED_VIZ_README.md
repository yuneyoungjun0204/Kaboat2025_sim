# VRX 통합 시각화 (Unified Visualization)

## 개요

`trajectory_viz_unified.py`는 기존 `trajectory_viz.py`의 개선 버전으로, 모든 좌표계를 **ENU(East-North-Up)**로 완전히 통일하여 더욱 직관적이고 명확한 시각화를 제공합니다.

## 주요 개선사항

### 1. 좌표계 완전 통일
- **모든 데이터를 ENU 좌표로 통일**: GPS, LiDAR, 웨이포인트, 장애물 영역 등
- **CoordinateTransformer 클래스**: 모든 좌표 변환을 한 곳에서 관리
- **명확한 좌표 규칙**: X=East, Y=North (일관성)

### 2. 장애물 검사 영역 시각화 개선
- **Convex Hull 기반 Polygon 표시**: 영역을 반투명 다각형으로 명확하게 표시
- **다층 시각화**:
  - 반투명 오렌지색 영역 채우기 (alpha=0.35)
  - 진한 외곽선 강조 (linewidth=2.5)
  - 중심점 마커 표시
- **Fallback 처리**: Convex Hull 실패시 점으로 표시

### 3. 더 나은 시각화 레이아웃
- **3-column 레이아웃**:
  - 왼쪽: 메인 플롯 (궤적 + LiDAR + 모든 영역)
  - 중간: LiDAR 극좌표 뷰
  - 오른쪽: 제어 출력
- **배경 및 스타일 개선**: 격자선, 색상, 투명도 최적화
- **더 큰 화면**: 18x9 inch (기존 대비 증가)

### 4. 성능 및 코드 품질
- **명확한 클래스 구조**:
  - `CoordinateTransformer`: 좌표 변환 전담
  - `UnifiedPlotManager`: 플롯 관리 전담
  - `UnifiedVizNode`: ROS2 노드 로직
- **효율적인 업데이트**: 동적 요소만 재생성
- **에러 처리 강화**: try-except 블록으로 안정성 향상

## 사용 방법

### 실행

```bash
# 기본 실행
python3 trajectory_viz_unified.py

# 또는 ROS2 run (setup.py에 entry point 추가 필요)
ros2 run vrx trajectory_viz_unified
```

### 의존성

기존 의존성에 추가로 필요:
```bash
pip install scipy  # Convex Hull 계산용
```

전체 의존성:
```bash
pip install numpy matplotlib scipy
```

### 웨이포인트 설정

1. 왼쪽 메인 플롯에서 원하는 위치를 **마우스 왼쪽 클릭**
2. 녹색 원형 마커가 표시됨
3. 현재 활성 웨이포인트는 큰 녹색 사각형으로 표시
4. ROS2 토픽 `/waypoint`로 자동 발행

## 시각화 요소 설명

### 메인 플롯 (왼쪽)
| 요소 | 색상 | 설명 |
|------|------|------|
| 궤적선 | 파란색 실선 | 로봇 이동 경로 |
| 로봇 마커 | 빨간색 원 | 현재 로봇 위치 |
| 현재 헤딩 | 빨간색 화살표 | IMU 기반 현재 방향 |
| 목표 헤딩 | 연두색 화살표 | 계산된 목표 방향 |
| LiDAR 장애물 | 빨간색 점 | 전역 좌표의 장애물 |
| 장애물 검사 영역 | 오렌지색 영역 | 반투명 다각형 + 외곽선 |
| Goal 체크 영역 | 보라색 영역 | 목표 도달 판정 구역 |
| LOS Target | 마젠타 다이아몬드 | Line-of-Sight 목표점 |
| 웨이포인트 | 녹색 원 | 클릭으로 설정한 지점 |

### LiDAR 극좌표 뷰 (중간)
- 로봇 중심의 극좌표 표시
- 북쪽이 위(0도)
- 빨간색 점으로 장애물 표시

### 제어 출력 (오른쪽)
- **Linear Velocity**: 파란색/빨간색 바
- **Angular Velocity**: 녹색/오렌지색 바
- **Mode**: 현재 제어 모드 (ONNX/DIRECT/STOP 등)

## 좌표계 규칙

### ENU (East-North-Up) 통일
```
┌─────────────────────► East (X축)
│
│
│
▼ North (Y축)
```

### 내부 데이터 형식
```python
position = [East, North]  # 모든 위치 데이터
waypoint = [East, North]  # 웨이포인트
los_target = [East, North]  # LOS target
obstacle_area = [[East1, North1], [East2, North2], ...]  # 영역 점들
```

### ROS2 메시지 변환
```python
# GPS 데이터 (입력)
gps_data = {'utm_x': North, 'utm_y': East}  # GPS 관례
position = [gps_data['utm_y'], gps_data['utm_x']]  # ENU로 변환

# 웨이포인트 발행 (출력)
msg = Point(x=North, y=East, z=0.0)  # 시스템 관례
```

## 기존 trajectory_viz.py와 비교

| 특징 | 기존 | Unified 버전 |
|------|------|--------------|
| 좌표계 통일성 | 부분적 (NED/GPS 혼용) | 완전 통일 (ENU) |
| 장애물 영역 표시 | 작은 점들 | Polygon + 외곽선 + 중심점 |
| 코드 구조 | 분산된 변환 로직 | 중앙집중식 (CoordinateTransformer) |
| 화면 크기 | 12x6 inch | 18x9 inch |
| LiDAR 표시 | 2곳 (분리) | 3곳 (메인 + 극좌표) |
| 성능 | 일반 | 최적화 (draw_idle 사용) |
| 에러 처리 | 기본 | 강화 (Fallback 포함) |

## 문제 해결

### scipy import 오류
```bash
# scipy 설치
pip install scipy

# 또는 conda 환경
conda install scipy
```

### 장애물 영역이 표시되지 않음
- ROS2 토픽 `/obstacle_check_area`가 발행되고 있는지 확인
- 데이터가 3개 이상의 점을 포함하는지 확인
- Convex Hull 계산 실패시 점으로 표시됨 (정상 동작)

### 좌표가 이상하게 표시됨
- GPS 원점 설정 확인
- `SensorDataManager`의 GPS 처리 로직 확인
- 로그에서 "축 범위 설정" 메시지 확인

## 개발자 노트

### 확장 방법

새로운 시각화 요소 추가:
```python
# 1. UnifiedPlotManager에 메서드 추가
def update_new_element(self, data):
    # ENU 좌표로 데이터 처리
    scatter = self.ax_main.scatter(...)
    self.dynamic_elements.append(scatter)

# 2. UnifiedVizNode에서 호출
self.plot_manager.update_new_element(self.new_data)
```

### 좌표 변환 추가
```python
# CoordinateTransformer에 메서드 추가
@staticmethod
def custom_to_enu(data) -> np.ndarray:
    # 변환 로직
    return np.array([east, north])
```

## 라이선스

이 코드는 VRX 프로젝트의 일부입니다.

## 기여

버그 리포트 및 개선 제안은 이슈로 등록해주세요.
