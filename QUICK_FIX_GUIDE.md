# 빠른 수정 가이드 - 트랙바 변경

## ❗ 트랙바가 안 보이는 이유
코드가 아직 수정되지 않았습니다. 아래 3개 파일을 수정해야 합니다.

---

## 📝 수정 1: utils/visualization_system.py

### ⚙️ 위치 1: 라인 109-112
**삭제:**
```python
        # 미션 모드 선택 트랙바
        # 0: 자동 (웨이포인트 기반), 1: 장애물 회피, 2: 부표 사이 통과, 3: 부표 회전
        cv2.createTrackbar('Mission Mode', 'Parameters',
                          0, 3, self._dummy_callback)
```

**추가:**
```python
        # 부표 사이 통과 미션 파라미터 트랙바
        cv2.createTrackbar('Pass: Max Depth Diff x10', 'Parameters',
                          50, 200, self._dummy_callback)  # 기본값 5.0m (50/10)
```

### ⚙️ 위치 2: 라인 149-150
**삭제:**
```python
        # 미션 모드 선택
        mission_mode = cv2.getTrackbarPos('Mission Mode', 'Parameters')
```

**추가:**
```python
        # 부표 사이 통과 미션 파라미터
        pass_max_depth_diff = cv2.getTrackbarPos('Pass: Max Depth Diff x10', 'Parameters') / 10.0
```

### ⚙️ 위치 3: 라인 170
**변경 전:**
```python
            'mission_mode': mission_mode
```

**변경 후:**
```python
            'pass_max_depth_diff': pass_max_depth_diff
```

---

## 📝 수정 2: Main_MCP.py

### ⚙️ 위치 1: 라인 275-294
**삭제 (전체 mission_mode 로직):**
```python
        # 트랙바로부터 미션 모드 읽어오기
        # 0: 자동 (웨이포인트 기반), 1: 장애물 회피, 2: 부표 사이 통과, 3: 부표 회전
        mission_mode = params.get('mission_mode', 0)

        # 미션 모드에 따라 현재 미션 타입 결정
        if mission_mode == 0:
            # 자동 모드: 웨이포인트 전환 확인
            self._check_waypoint_transition()
            current_mission_type = self.waypoint_manager.get_current_mission_type()
        elif mission_mode == 1:
            # 수동 장애물 회피 모드
            current_mission_type = MissionType.OBSTACLE_AVOID
        elif mission_mode == 2:
            # 수동 부표 사이 통과 모드
            current_mission_type = MissionType.PASS_BETWEEN_BUOYS
        elif mission_mode == 3:
            # 수동 부표 회전 모드
            current_mission_type = MissionType.CIRCLE_BUOY
        else:
            current_mission_type = None
```

**추가 (간단한 자동 모드만):**
```python
        # 자동 모드: 웨이포인트 기반 미션 전환
        self._check_waypoint_transition()
        current_mission_type = self.waypoint_manager.get_current_mission_type()
```

### ⚙️ 위치 2: 라인 354-360
**변경 전:**
```python
        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                logger=self.get_logger()
            )
```

**변경 후:**
```python
        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            # 트랙바 파라미터 가져오기
            trackbar_params = self.visualization.update_parameters_from_trackbars()
            pass_params = {
                'max_depth_diff': trackbar_params.get('pass_max_depth_diff', 5.0)
            }

            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                raw_detections=self.raw_detections,
                mission_params=pass_params,
                logger=self.get_logger()
            )
```

---

## 📝 수정 3: utils/mission_strategies_new.py

### ⚙️ 위치: 라인 36-54 (PassBetweenBuoysMission.execute 메서드)

**기존 execute 메서드 시그니처:**
```python
    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, **kwargs) -> Tuple[float, float]:
```

**변경 후:**
```python
    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, raw_detections: List[Dict] = None, **kwargs) -> Tuple[float, float]:
```

**라인 38-47 부분 전체 교체:**

**변경 전:**
```python
        """빨간색/초록색 고깔 부표 사이로 지나가기"""
        # 빨간색/초록색 부표 찾기
        red_buoy = None
        green_buoy = None

        for det in detected_objects:
            if det['label'] == 'red_cone':
                red_buoy = det
            elif det['label'] == 'green_cone':
                green_buoy = det

        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
```

**변경 후:**
```python
        """
        빨간색/초록색 고깔 부표 사이로 지나가기

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)
        """
        # 빨간색/초록색 부표 찾기 (원본 탐지값 우선 사용)
        red_buoy = None
        green_buoy = None
        data_source = "RAW"

        # 1. 원본 탐지값(raw)에서 먼저 찾기
        if raw_detections:
            for det in raw_detections:
                if det['label'] == 'red_cone':
                    red_buoy = det
                elif det['label'] == 'green_cone':
                    green_buoy = det

        # 2. 원본값에 없으면 추적값(tracked)에서 찾기 (fallback)
        if (not red_buoy or not green_buoy) and detected_objects:
            data_source = "TRACKED"
            if not red_buoy:
                for det in detected_objects:
                    if det['label'] == 'red_cone':
                        red_buoy = det
                        break
            if not green_buoy:
                for det in detected_objects:
                    if det['label'] == 'green_cone':
                        green_buoy = det
                        break
            if logger and (red_buoy or green_buoy):
                logger.info("⚠️ 원본값 없음 -> 추적값 사용 (fallback)")

        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 깊이 차이 필터링 (너무 차이 나면 멀리 있는 것 무시)
            max_depth_diff = kwargs.get('mission_params', {}).get('max_depth_diff', 5.0)
            red_depth = red_buoy.get('depth', 0.0)
            green_depth = green_buoy.get('depth', 0.0)
            depth_diff = abs(red_depth - green_depth)

            if depth_diff > max_depth_diff:
                # 깊이 차이가 너무 크면 멀리 있는 부표 무시
                if red_depth > green_depth:
                    if logger:
                        logger.warn(f"빨간 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)")
                    red_buoy = None
                else:
                    if logger:
                        logger.warn(f"초록 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)")
                    green_buoy = None

        # 필터링 후 두 부표가 모두 있는 경우
        if red_buoy and green_buoy:
```

**라인 73-76 로그 부분:**

**변경 전:**
```python
            if logger:
                logger.info(
                    f"Pass Buoys: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )
```

**변경 후:**
```python
            if logger:
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )
```

---

## ✅ 수정 완료 후

프로그램을 재시작하면:
1. ❌ "Mission Mode" 트랙바 사라짐
2. ✅ "Pass: Max Depth Diff x10" 트랙바 생성됨 (기본값 50 = 5.0m)
3. ✅ 자동 모드로만 동작 (웨이포인트 기반)
4. ✅ 부표 사이 통과 시 깊이 차이 필터 적용
5. ✅ 원본 탐지값 우선 사용

---

## 🎯 테스트

1. 프로그램 실행
2. Parameters 창에서 "Pass: Max Depth Diff x10" 트랙바 확인
3. 부표 탐지 시 로그 확인:
   - `Pass Buoys [RAW]: ...` 또는 `Pass Buoys [TRACKED]: ...`
   - `빨간 부표 무시 (깊이 차이 X.XXm > Y.YYm)`
   - `초록 부표 무시 (깊이 차이 X.XXm > Y.YYm)`
