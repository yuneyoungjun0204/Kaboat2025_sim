# VRX 코드 리팩토링 요약

## 리팩토링 날짜
2025-10-27

## 목표
Main_MCP.py를 중심으로 코드를 간결하고 불필요한 부분이 없도록 개선

## 주요 변경사항

### 1. Main_MCP.py 간소화 및 시각화 복구 (511줄 → 441줄, 14% 감소)

**제거된 기능:**
- ❌ ParameterManager 클래스: 별도 모듈 대신 Main_MCP.py 내부에서 직접 처리
- ❌ 강제 장애물 회피 모드: 디버깅용 특수 모드 제거
- ❌ 깊이 맵 시각화: 성능 최적화를 위해 제거

**복구/유지된 기능:**
- ✅ **트랙바 시각화**: Parameters 창에서 실시간 파라미터 조정 가능
- ✅ **객체 박스 시각화**: VRX Mission Control 창에서 탐지/추적 결과 표시
- ✅ **실시간 파라미터 조정**: 탐지 임계값, 박스 크기, 깊이 범위, Thrust Scale 등
- ✅ **IMM-PDAF 추적 시각화**: 추적 ID, Coast 상태 표시
- ✅ **미션 파라미터 트랙바**: Circle 미션, Pass 미션 파라미터 실시간 조정

**주요 개선점:**
- ParameterManager 로직을 `_update_parameters_from_trackbars()` 메서드로 통합
- 트랙바 파라미터를 Detection, Tracker, Mission 시스템에 실시간 적용
- 깊이 맵 제거로 CPU 부하 감소

### 2. trajectory_viz.py 간소화 (1151줄 → 804줄, 30% 감소)

**제거된 Legacy 코드:**
- ❌ calculate_path_width_points(): 배 폭 경로 계산 (51줄)
- ❌ update_path_width_plot(): 배 폭 경로 시각화 (35줄)
- ❌ calculate_goal_check_areas(): goal_check 영역 계산 (48줄)
- ❌ update_goal_check_area(): Legacy goal_check 시각화 (61줄)
- ❌ clear_goal_check_plots(), clear_path_width_plots(): Legacy 정리 함수들
- ❌ update_goal_check_area_from_ros(): 사용하지 않는 ROS goal_check 처리
- ❌ 불필요한 변수들: boat_width, safety_margin, goal_check_areas, current_mode 등

**유지된 핵심 기능:**
- ✅ 궤적 플롯 (위치, 헤딩)
- ✅ LiDAR 플롯 (원형 좌표계)
- ✅ 제어 출력값 트랙바
- ✅ 웨이포인트 표시
- ✅ LOS target 시각화
- ✅ 장애물 체크 영역 시각화

**주요 개선점:**
- Legacy 코드 대량 제거 (347줄)
- clear_all_plots() 단순화
- 모드 표시 로직 간소화

### 3. main_onnx_v5_final_refactored_copy2.py 처리

**조치:**
- ✅ main_onnx_v5_final_refactored_copy2.py.backup으로 이름 변경
- **이유:** Main_MCP.py와 기능이 중복됨 (ONNX 제어, 장애물 회피, 웨이포인트 추종)

### 4. utils 모듈 정리

**제거된 모듈:**
- ❌ parameter_manager.py → parameter_manager.py.backup으로 이동 (Main_MCP.py에 통합)
- ❌ utils/__init__.py에서 ParameterManager export 제거

**유지된 모듈:**
- ✅ config.py: Constants 클래스로 모든 설정 관리
- ✅ sensor_callbacks.py: 센서 콜백 핸들러
- ✅ onnx_controller.py: ONNX 모델 제어
- ✅ mission_executor.py: 미션 실행 로직
- ✅ visualization_system.py: 트랙바 + 객체 박스 시각화

## 시각화 기능 상세

### 표시되는 창
1. **VRX Mission Control** (640x480)
   - 카메라 이미지 위에 탐지/추적 결과 오버레이
   - 추적된 객체: 굵은 실선 박스 + 큰 점
   - 원본 탐지: 얇은 점선 박스 + 작은 점
   - 미션 정보, 웨이포인트 정보, 탐지 통계 표시

2. **Parameters** (600x500)
   - 탐지 파라미터 트랙바 (Detect Threshold, Min/Max Box Area, Min/Max Depth)
   - 제어 파라미터 트랙바 (Thrust Scale)
   - IMM-PDAF 파라미터 트랙바 (Max Coast Frames, Gate Threshold)
   - Circle 미션 파라미터 트랙바 (Rotation Dir, Base Speed, PID Kp 등)
   - Pass 미션 파라미터 트랙바 (Max Depth Diff)

3. **제거된 창**
   - ❌ Depth Map 창 (성능 최적화를 위해 제거)

### 실시간 조정 가능한 파라미터

#### 탐지 파라미터
- `Detect Threshold`: 객체 탐지 신뢰도 임계값 (0.000-1.000)
- `Min Box Area`: 최소 박스 면적 (0-50000 픽셀)
- `Max Box Area`: 최대 박스 면적 (0-150000 픽셀)
- `Min Depth`: 최소 깊이 (0-100m)
- `Max Depth`: 최대 깊이 (0-200m)

#### 제어 파라미터
- `Thrust Scale`: 스러스터 스케일 (0-3000)

#### 추적 파라미터
- `Max Coast Frames`: 최대 Coast 프레임 수 (0-30)
- `Gate Threshold`: 검증 게이트 임계값 (0-20.0)

#### Circle 미션 파라미터
- `Rotation Dir`: 회전 방향 (1=시계, 2=반시계)
- `Base Speed`: 기본 속도 (0-300)
- `Min Speed`: 최소 속도 (0-200)
- `Max Turn`: 최대 회전력 (0-250)
- `PID Kp`: PID 비례 게인 (0.0-5.0)
- `TX BaseX`: Target X 기준값 (0-2000)
- `TX Slope`: Target X 기울기 (0-10000)
- `TX MinX`: Target X 최소값 (0-2000)
- `TX MaxX`: Target X 최대값 (0-2000)

#### Pass 미션 파라미터
- `Max Depth Diff`: 최대 깊이 차이 (0-20.0m)

## 전체 코드 라인 수 변화

| 파일 | 이전 | 이후 | 감소량 | 감소율 |
|------|------|------|--------|--------|
| Main_MCP.py | 511줄 | 441줄 | -70줄 | -14% |
| trajectory_viz.py | 1151줄 | 804줄 | -347줄 | -30% |
| **합계** | **1662줄** | **1245줄** | **-417줄** | **-25%** |

## 백업 파일

다음 파일들은 백업으로 보관되었습니다 (필요시 복구 가능):
- `main_onnx_v5_final_refactored_copy2.py.backup`
- `utils/parameter_manager.py.backup`

## 사용법

### Main_MCP.py 실행
```bash
# ROS2 워크스페이스 source
source ~/vrx_ws/install/setup.bash

# Main 컨트롤러 실행
python3 Main_MCP.py
```

실행하면 자동으로 다음 창들이 표시됩니다:
- **VRX Mission Control**: 객체 탐지/추적 결과
- **Parameters**: 실시간 파라미터 조정 트랙바

### trajectory_viz.py 실행 (시각화 필요시)
```bash
# 별도 터미널에서 실행
python3 trajectory_viz.py
```

## 트랙바 사용 예시

### 탐지 감도 조정
1. `Detect Threshold` 트랙바를 조정하여 탐지 민감도 변경
2. 값이 낮을수록 더 많은 객체를 탐지 (노이즈 증가)
3. 값이 높을수록 신뢰도 높은 객체만 탐지

### Circle 미션 속도 조정
1. `Circle: Base Speed` 트랙바로 선회 기본 속도 조정
2. `Circle: PID Kp` 트랙바로 제어 반응성 조정
3. 실시간으로 선회 동작 최적화 가능

### Thrust Scale 조정
1. `Thrust Scale` 트랙바로 전체 추력 크기 조정
2. 시뮬레이션과 실제 환경에서 다른 값 필요시 유용

## 주요 이점

1. **코드 간결성**: 417줄 감소 (25% 감소)
2. **성능 최적화**: 깊이 맵 제거로 CPU 부하 감소
3. **유지보수성**: ParameterManager 통합으로 코드 구조 단순화
4. **실시간 조정**: 트랙바로 모든 파라미터 실시간 변경 가능
5. **시각적 피드백**: 객체 탐지/추적 결과를 명확히 표시

## 다음 단계

리팩토링된 코드를 테스트하려면:

1. ROS2 시뮬레이션 환경에서 Main_MCP.py 실행
2. Parameters 창에서 트랙바로 파라미터 실시간 조정
3. VRX Mission Control 창에서 탐지/추적 결과 확인
4. 각 미션 모드 테스트:
   - PASS_BETWEEN_BUOYS: 트랙바에서 Max Depth Diff 조정
   - CIRCLE_BUOY: 트랙바에서 회전 방향, 속도, PID 게인 조정
   - OBSTACLE_AVOID: LiDAR 장애물 회피 확인
   - WAYPOINT_FOLLOW: 웨이포인트 추종 확인
5. trajectory_viz.py로 궤적 시각화 확인 (선택)

## 참고사항

- 백업 파일들은 `.backup` 확장자로 보관되어 있습니다
- 필요시 `mv *.backup <원본이름>`으로 복구 가능합니다
- 트랙바 조정은 실시간으로 시스템에 반영됩니다
- 깊이 맵이 필요한 경우 `_visualize()` 메서드에서 주석 해제 가능

## 트러블슈팅

### 시각화 창이 보이지 않는 경우
- X11 forwarding이 활성화되어 있는지 확인
- `echo $DISPLAY` 명령으로 DISPLAY 환경변수 확인

### 트랙바가 동작하지 않는 경우
- OpenCV가 올바르게 설치되어 있는지 확인
- `cv2.imshow()` 가 호출되고 있는지 확인

### 객체가 탐지되지 않는 경우
- `Detect Threshold` 트랙바를 낮춰보기
- `Min/Max Box Area` 범위를 넓혀보기
- `Min/Max Depth` 범위를 조정해보기
