# VRX 시스템 좌표계 정리

## 개요

VRX 시스템에서는 여러 좌표계가 사용됩니다. 이 문서는 각 좌표계의 정의, 변환 방법, 사용 위치를 명확히 정리하여 혼동을 방지합니다.

---

## 1. 세계 좌표계 (World/Global Coordinate System)

### 1.1 UTM 좌표계 (Universal Transverse Mercator)

**정의:**
- GPS 위경도를 평면 좌표로 변환한 좌표계
- **X축**: Easting (동서 방향, 동쪽이 양수)
- **Y축**: Northing (남북 방향, 북쪽이 양수)
- **단위**: 미터 (m)

**기준점:**
- 첫 번째 GPS 데이터 수신 시 해당 위치를 (0, 0)으로 설정
- `SensorDataManager`에서 초기화 시점에 `utm_origin_x`, `utm_origin_y`에 저장
- 이후 모든 GPS 데이터는 이 기준점으로부터의 상대 좌표로 변환

**사용 위치:**
- `agent_position`: 로봇의 현재 위치 (2D 벡터)
- 웨이포인트 좌표 (`waypoint_manager.py`)
- LOS 타겟 좌표
- 장애물 체크 영역 좌표

**변환 예시:**
```python
# GPS → UTM (센서 전처리에서 자동 처리)
utm_x, utm_y = utm.from_latlon(latitude, longitude)

# 기준점 기준 상대 좌표
relative_x = utm_x - utm_origin_x
relative_y = utm_y - utm_origin_y
```

---

## 2. 로봇 좌표계 (Robot/Body Coordinate System)

### 2.1 IMU 헤딩 (Heading)

**정의:**
- 로봇의 진행 방향을 나타내는 각도
- **범위**: 0° ~ 360°
- **방향 규칙**:
  - 0° = 서쪽 (West)
  - 90° = 북쪽 (North)
  - 180° = 동쪽 (East)
  - 270° = 남쪽 (South)
- **회전 방향**: 시계 방향 (Clockwise)

**주의사항:**
- 일반적인 수학 좌표계(반시계방향, 0°=동쪽)와 **다름**
- UTM 좌표계와 조합 시 변환 필요

**사용 위치:**
- `agent_heading`: 로봇의 현재 헤딩 각도
- 제어 알고리즘의 목표 각도 계산
- 시각화에서 헤딩 화살표 표시

**UTM 좌표계로의 변환:**
```python
# IMU 헤딩을 UTM 좌표계 방향 벡터로 변환
heading_rad = np.radians(imu_heading)
dx = np.cos(heading_rad)  # X축 성분 (Easting)
dy = np.sin(heading_rad)  # Y축 성분 (Northing)
```

### 2.2 LiDAR 좌표계

**정의:**
- 로봇 중심을 기준으로 한 극좌표계
- **각도 범위**: -100° ~ +100° (총 201개 측정값)
- **각도 기준**:
  - 0° = 로봇 정면 (전방)
  - -100° = 로봇 왼쪽 (좌측)
  - +100° = 로봇 오른쪽 (우측)
- **거리**: 0 ~ 100m

**배열 인덱스 매핑:**
```python
# 각도 → 인덱스
index = angle + 100  # 예: -100° → 0, 0° → 100, +100° → 200

# 인덱스 → 각도
angle = index - 100  # 예: 0 → -100°, 100 → 0°, 200 → +100°
```

**로봇 중심 좌표계로의 변환:**
```python
# LiDAR 극좌표 → 로봇 중심 직교좌표
angle_rad = np.radians(lidar_angle)
x_robot = distance * np.cos(angle_rad)  # 전방 방향
y_robot = distance * np.sin(angle_rad)  # 좌측 방향
```

**사용 위치:**
- `lidar_distances`: 201개 거리 측정값 배열
- 장애물 감지 및 회피 로직
- `trajectory_viz.py`의 LiDAR 시각화

---

## 3. 이미지 좌표계 (Image Coordinate System)

### 3.1 카메라 이미지 좌표

**정의:**
- 카메라로부터 획득한 이미지의 픽셀 좌표계
- **원점**: 이미지 좌상단 (Top-Left)
- **X축**: 수평 방향 (왼쪽 → 오른쪽, 0 ~ 1280)
- **Y축**: 수직 방향 (위 → 아래, 0 ~ 720)
- **단위**: 픽셀 (pixel)

**이미지 크기:**
- 너비 (Width): 1280 픽셀
- 높이 (Height): 720 픽셀

**사용 위치:**
- 객체 탐지 결과 (bounding box)
- 부표 중심 좌표 계산
- 미션 제어 (부표 사이 지나가기, 부표 회전)

**중심 좌표:**
```python
image_center_x = 1280 / 2  # 640 픽셀
image_center_y = 720 / 2   # 360 픽셀
```

### 3.2 깊이 맵 (Depth Map)

**정의:**
- MiDaS를 통해 추정된 각 픽셀의 깊이 정보
- **좌표**: 이미지 좌표계와 동일
- **값**: 0.0 ~ ∞ (정규화된 깊이, 실제 거리의 역수 개념)
- **단위**: 정규화된 값 (무차원)

**특징:**
- 상대적 깊이 값 (절대 거리 아님)
- 깊이 필터링 시 임계값 조정 필요 (`min_depth_threshold`, `max_depth_threshold`)

**사용 위치:**
- 부표 탐지 시 깊이 범위 필터링
- 부표 간 거리 차이 비교 (`PASS_BETWEEN_MAX_DEPTH_DIFF`)

---

## 4. 좌표계 변환

### 4.1 LiDAR → UTM 좌표계

LiDAR 측정값을 세계 좌표계로 변환하는 과정:

**단계:**
1. LiDAR 극좌표 → 로봇 중심 직교좌표
2. 로봇 중심 좌표 → UTM 좌표 (회전 및 평행이동)

**코드 예시:**
```python
# 1. LiDAR 극좌표 → 로봇 중심 좌표
lidar_angle_rad = np.radians(lidar_angle)
x_robot = distance * np.cos(lidar_angle_rad)  # 전방
y_robot = distance * np.sin(lidar_angle_rad)  # 좌측

# 2. 로봇 중심 좌표 → UTM 좌표
# (a) IMU 헤딩에 따른 회전 변환
heading_rad = np.radians(agent_heading)
cos_h = np.cos(heading_rad)
sin_h = np.sin(heading_rad)

# (b) 90도 회전 적용 (LiDAR 좌표계 → 로봇 좌표계)
rotated_x = y_robot   # Y → X
rotated_y = -x_robot  # X → -Y

# (c) 헤딩 회전 적용
utm_x = agent_x + (rotated_x * cos_h - rotated_y * sin_h)
utm_y = agent_y + (rotated_x * sin_h + rotated_y * cos_h)
```

**사용 위치:**
- `trajectory_viz.py`의 `update_trajectory_with_lidar()`

### 4.2 이미지 → 로봇 좌표계

카메라 이미지 좌표를 로봇 좌표계로 변환하는 과정 (주로 제어용):

**제어 오차 계산:**
```python
# 부표 중심 X 좌표 (픽셀)
buoy_x = detection['center'][0]

# 이미지 중심으로부터 오차 계산
image_center_x = 1280 / 2
error = buoy_x - image_center_x  # 좌(-), 우(+)

# 정규화 (선택적)
normalized_error = error / (1280 / 2)  # -1.0 ~ 1.0
```

**주의:**
- 이미지 좌표는 직접 세계 좌표계로 변환하지 않음
- 상대적 오차만 사용하여 제어 명령 생성

---

## 5. 각도 정규화 (Angle Normalization)

### 5.1 각도 범위 변환

**0° ~ 360° 정규화:**
```python
def normalize_angle_0_360(angle):
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle
```

**-180° ~ 180° 정규화:**
```python
def normalize_angle_pm180(angle):
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle
```

**-π ~ π 정규화 (라디안):**
```python
def normalize_angle_rad(angle_rad):
    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    while angle_rad <= -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad
```

---

## 6. 주요 좌표계 사용 맵

| 모듈 / 변수 | 좌표계 | 단위 | 설명 |
|------------|--------|------|------|
| `agent_position` | UTM | m | 로봇의 세계 좌표 위치 |
| `agent_heading` | IMU | degree (0-360°) | 로봇의 진행 방향 |
| `lidar_distances` | LiDAR 극좌표 | m, degree | 201개 거리 측정값 |
| `detection['center']` | 이미지 | pixel | 탐지된 객체의 중심 좌표 |
| `detection['depth']` | 깊이 맵 | normalized | 객체까지의 상대 깊이 |
| `waypoint['x']`, `['y']` | UTM | m | 웨이포인트 좌표 |
| `los_target` | UTM | m | LOS 알고리즘의 목표 지점 |

---

## 7. 변환 체크리스트

새로운 기능 개발 시 다음을 확인하세요:

1. **입력 데이터의 좌표계 확인**
   - GPS → UTM (기준점 기준)
   - IMU → 0-360° (서쪽 기준)
   - LiDAR → 극좌표 (-100° ~ +100°)
   - 이미지 → 픽셀 좌표

2. **필요한 좌표 변환 수행**
   - 각 좌표계 간 변환 공식 적용
   - 회전 행렬 사용 시 올바른 각도 사용
   - 평행이동 (로봇 위치 더하기)

3. **각도 정규화**
   - 각도 연산 후 항상 정규화
   - 올바른 범위로 변환 (0-360° 또는 -180-180°)

4. **단위 일관성**
   - 거리: 미터 (m)
   - 각도: 도 (degree) 또는 라디안 (radian) 명확히 구분
   - 픽셀: 정수 (int)

---

## 8. 디버깅 팁

### 8.1 좌표 값 확인

```python
# UTM 좌표 출력
print(f"Agent Position: ({agent_position[0]:.2f}, {agent_position[1]:.2f}) m")

# 헤딩 확인
print(f"Agent Heading: {agent_heading:.1f}°")

# LiDAR 전방 거리
print(f"LiDAR Front: {lidar_distances[100]:.2f} m")

# 이미지 중심으로부터 오차
print(f"Image Error: {error:.1f} pixels")
```

### 8.2 시각화 확인

- `trajectory_viz.py`를 실행하여 실시간 좌표계 시각화
- 헤딩 화살표 방향이 로봇 진행 방향과 일치하는지 확인
- LiDAR 포인트가 올바른 위치에 표시되는지 확인

---

## 9. 참고 자료

- [CLAUDE.md](./CLAUDE.md) - 시스템 전체 구조 및 제어 흐름
- [WAYPOINT_CONFIG_GUIDE.md](./WAYPOINT_CONFIG_GUIDE.md) - 웨이포인트 설정 가이드
- [utils/config.py](./utils/config.py) - 모든 시스템 파라미터 중앙 관리

---

## 변경 이력

- 2025-01-XX: 초기 문서 작성
