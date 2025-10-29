# 웨이포인트 및 미션 설정 가이드

## 1. 웨이포인트 기본값 변경

웨이포인트의 기본값은 `/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/config.py` 파일에서 관리됩니다.

### config.py 파일 위치
```
/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/config.py
```

### 웨이포인트 수정 방법

`config.py` 파일을 열고 `PREDEFINED_WAYPOINTS` 리스트를 수정하세요:

```python
# 웨이포인트 기본값 (수정 가능)
# 형식: (x, y, mission_type, radius, params)
PREDEFINED_WAYPOINTS = [
    # 웨이포인트 1: 부표 사이 지나가기
    (40, 80, MissionType.PASS_BETWEEN_BUOYS, 20.0, {}),
    
    # 웨이포인트 2: 부표 한바퀴 회전
    (42, 115, MissionType.CIRCLE_BUOY, 20.0, {
        'rotation_direction': 2,  # 1=시계방향, 2=반시계방향
        'circle_radius': 15.0
    }),
    
    # 웨이포인트 3: 웨이포인트 추종
    (0, 165, MissionType.WAYPOINT_FOLLOW, 20.0, {}),
    
    # 웨이포인트 4: 장애물 회피
    (0, 0, MissionType.OBSTACLE_AVOID, 20.0, {})
]
```

### 파라미터 설명

각 웨이포인트는 다음 형식을 따릅니다:
```python
(x, y, mission_type, radius, params)
```

- **x, y**: 웨이포인트 좌표 (미터)
  - x: UTM 좌표계의 X값 (Easting)
  - y: UTM 좌표계의 Y값 (Northing)

- **mission_type**: 미션 타입
  - `MissionType.OBSTACLE_AVOID`: 장애물 회피
  - `MissionType.PASS_BETWEEN_BUOYS`: 부표 사이 지나가기
  - `MissionType.CIRCLE_BUOY`: 부표 한바퀴 회전
  - `MissionType.WAYPOINT_FOLLOW`: 웨이포인트 추종

- **radius**: 웨이포인트 도달 판정 반경 (미터)
  - 기본값: 20.0m
  - 로봇이 이 반경 내에 들어오면 다음 웨이포인트로 이동

- **params**: 미션별 추가 파라미터 (딕셔너리)
  - Circle 미션의 경우:
    - `rotation_direction`: 회전 방향 (1=시계방향, 2=반시계방향)
    - `circle_radius`: 선회 반경 (미터)
  - 다른 미션의 경우: 빈 딕셔너리 `{}`

### 예시: 웨이포인트 추가/수정

**새 웨이포인트 추가:**
```python
PREDEFINED_WAYPOINTS = [
    # 기존 웨이포인트들...
    
    # 새 웨이포인트: 좌표 (50, 100)에서 장애물 회피
    (50, 100, MissionType.OBSTACLE_AVOID, 15.0, {}),
    
    # 새 웨이포인트: 좌표 (60, 120)에서 시계방향 회전
    (60, 120, MissionType.CIRCLE_BUOY, 20.0, {
        'rotation_direction': 1,
        'circle_radius': 10.0
    })
]
```

**좌표 수정:**
```python
# 웨이포인트 1의 좌표를 (40, 80)에서 (50, 90)으로 변경
(50, 90, MissionType.PASS_BETWEEN_BUOYS, 20.0, {}),
```

**도달 반경 조정:**
```python
# 웨이포인트 2의 도달 반경을 20m에서 25m로 변경
(42, 115, MissionType.CIRCLE_BUOY, 25.0, {...}),
```

## 2. 트랙바로 미션 순서 변경

실행 중에 **Parameters** 창의 **Mission Order** 트랙바로 미션 순서를 실시간으로 변경할 수 있습니다.

### 트랙바 값 의미

- **0 = 기본 순서**: config.py의 PREDEFINED_WAYPOINTS 순서 사용
  1. PASS_BETWEEN_BUOYS (부표 사이 지나가기)
  2. CIRCLE_BUOY (부표 한바퀴 회전)
  3. WAYPOINT_FOLLOW (웨이포인트 추종)
  4. OBSTACLE_AVOID (장애물 회피)

- **1 = 장애물 회피**: 클릭한 모든 웨이포인트를 OBSTACLE_AVOID 미션으로 설정

- **2 = 부표 사이 지나가기**: 클릭한 모든 웨이포인트를 PASS_BETWEEN_BUOYS 미션으로 설정

- **3 = 부표 한바퀴 회전**: 클릭한 모든 웨이포인트를 CIRCLE_BUOY 미션으로 설정

### 사용 예시

1. **기본 순서 사용** (Mission Order = 0)
   - trajectory_viz.py에서 클릭한 첫 번째 점: PASS_BETWEEN_BUOYS
   - 두 번째 점: CIRCLE_BUOY
   - 세 번째 점: WAYPOINT_FOLLOW
   - 네 번째 점 이후: OBSTACLE_AVOID

2. **모든 웨이포인트를 장애물 회피로** (Mission Order = 1)
   - trajectory_viz.py에서 클릭한 모든 점: OBSTACLE_AVOID

3. **모든 웨이포인트를 부표 사이 지나가기로** (Mission Order = 2)
   - trajectory_viz.py에서 클릭한 모든 점: PASS_BETWEEN_BUOYS

4. **모든 웨이포인트를 부표 회전으로** (Mission Order = 3)
   - trajectory_viz.py에서 클릭한 모든 점: CIRCLE_BUOY
   - 회전 방향은 Circle: Rotation Dir 트랙바로 조정 가능

### 트랙바 조합 예시

**시나리오 1: 3개의 장애물 회피 포인트 설정**
1. Parameters 창에서 Mission Order = 1 (장애물 회피)
2. trajectory_viz.py 궤적 창에서 3개 지점 클릭
3. 모든 웨이포인트가 OBSTACLE_AVOID로 설정됨

**시나리오 2: 2개의 부표 회전 포인트 설정 (반시계방향)**
1. Parameters 창에서 Mission Order = 3 (부표 회전)
2. Parameters 창에서 Circle: Rotation Dir = 2 (반시계방향)
3. trajectory_viz.py 궤적 창에서 2개 지점 클릭
4. 모든 웨이포인트가 CIRCLE_BUOY (반시계방향)로 설정됨

**시나리오 3: 혼합 미션 설정**
1. Mission Order = 1, 첫 번째 점 클릭 → OBSTACLE_AVOID
2. Mission Order = 2, 두 번째 점 클릭 → PASS_BETWEEN_BUOYS
3. Mission Order = 3, 세 번째 점 클릭 → CIRCLE_BUOY
4. 각 웨이포인트마다 다른 미션 설정 가능

## 3. 빠른 수정 체크리스트

### config.py에서 기본값 수정
- [ ] config.py 파일 열기
- [ ] PREDEFINED_WAYPOINTS 찾기
- [ ] 좌표 (x, y) 수정
- [ ] 미션 타입 변경 (필요시)
- [ ] 도달 반경 조정 (필요시)
- [ ] Circle 미션 파라미터 조정 (필요시)
- [ ] 파일 저장
- [ ] Main_MCP.py 재시작

### 실행 중 트랙바로 수정
- [ ] Main_MCP.py 실행
- [ ] Parameters 창 확인
- [ ] Mission Order 트랙바 조정 (0/1/2/3)
- [ ] Circle 파라미터 트랙바 조정 (필요시)
- [ ] trajectory_viz.py에서 웨이포인트 클릭
- [ ] 로그에서 추가된 미션 확인

## 4. 자주 묻는 질문

**Q: config.py를 수정했는데 적용이 안 돼요.**
A: Main_MCP.py를 재시작해야 합니다. config.py는 프로그램 시작 시 한 번만 로드됩니다.

**Q: 트랙바로 설정한 미션 순서가 저장되나요?**
A: 트랙바 설정은 실행 중에만 유효합니다. 영구적으로 저장하려면 config.py를 수정하세요.

**Q: 웨이포인트를 몇 개까지 추가할 수 있나요?**
A: 제한이 없습니다. config.py나 trajectory_viz.py에서 원하는 만큼 추가할 수 있습니다.

**Q: Circle 미션의 회전 반경을 바꾸고 싶어요.**
A: config.py의 `circle_radius` 값을 변경하거나, 실행 중에는 변경할 수 없습니다. (향후 트랙바 추가 가능)

**Q: 웨이포인트를 동적으로 삭제할 수 있나요?**
A: 현재는 불가능합니다. 프로그램을 재시작해야 합니다.

## 5. 트러블슈팅

### 웨이포인트가 추가되지 않음
- trajectory_viz.py가 실행 중인지 확인
- ROS2 토픽 `/vrx/waypoint`가 발행되는지 확인
- Main_MCP.py 로그에서 "웨이포인트 추가" 메시지 확인

### 미션 순서가 적용되지 않음
- Parameters 창이 열려 있는지 확인
- Mission Order 트랙바가 올바른 값인지 확인
- 웨이포인트 추가 전에 트랙바를 먼저 조정했는지 확인

### config.py 수정이 반영되지 않음
- Main_MCP.py를 완전히 종료했는지 확인
- config.py 파일이 올바르게 저장되었는지 확인
- Python 문법 오류가 없는지 확인

## 6. 고급 사용법

### 조건부 웨이포인트 설정
config.py에서 Python 코드를 사용하여 동적으로 웨이포인트를 생성할 수 있습니다:

```python
# 직선 경로 생성 (10m 간격으로 5개)
PREDEFINED_WAYPOINTS = [
    (i * 10, 50, MissionType.OBSTACLE_AVOID, 15.0, {})
    for i in range(5)
]

# 원형 경로 생성
import math
PREDEFINED_WAYPOINTS = [
    (
        50 + 30 * math.cos(math.radians(i * 45)),
        50 + 30 * math.sin(math.radians(i * 45)),
        MissionType.WAYPOINT_FOLLOW,
        15.0,
        {}
    )
    for i in range(8)  # 45도 간격으로 8개
]
```

### 미션별 맞춤 설정
각 미션 타입에 맞는 파라미터를 설정할 수 있습니다:

```python
PREDEFINED_WAYPOINTS = [
    # 부표 사이 지나가기 (깊이 차이 허용 범위 설정은 트랙바로)
    (40, 80, MissionType.PASS_BETWEEN_BUOYS, 20.0, {}),
    
    # 부표 회전 (회전 방향 및 반경 지정)
    (60, 100, MissionType.CIRCLE_BUOY, 20.0, {
        'rotation_direction': 1,  # 시계방향
        'circle_radius': 12.0     # 12m 반경
    }),
    
    # 장애물 회피 (특별한 파라미터 없음)
    (80, 120, MissionType.OBSTACLE_AVOID, 25.0, {}),
]
```

## 7. 요약

- **기본값 수정**: `config.py`의 `PREDEFINED_WAYPOINTS` 편집
- **실시간 변경**: `Mission Order` 트랙바 조정 (0/1/2/3)
- **웨이포인트 추가**: trajectory_viz.py에서 클릭
- **설정 확인**: Main_MCP.py 로그 메시지 확인

편집 후 변경사항을 적용하려면 Main_MCP.py를 재시작하세요!
