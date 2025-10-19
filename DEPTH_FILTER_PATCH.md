# 부표 사이 통과 미션 깊이 차이 필터 + mission_mode 트랙바 제거

## 수정 개요
1. **깊이 차이 필터 추가**: 빨강/초록 부표의 깊이 차이가 일정 이상이면 멀리 있는 부표 무시
2. **트랙바로 조정 가능**: `Pass: Max Depth Diff x10` 트랙바 추가 (기본값 5.0m)
3. **mission_mode 트랙바 제거**: 자동 모드로만 동작

---

## 수정 1: utils/visualization_system.py

### 위치 1-1: 트랙바 설정 (라인 106-112)

**변경 전:**
```python
        cv2.createTrackbar('Circle: TX MaxX', 'Parameters',
                          1200, 2000, self._dummy_callback)

        # 미션 모드 선택 트랙바
        # 0: 자동 (웨이포인트 기반), 1: 장애물 회피, 2: 부표 사이 통과, 3: 부표 회전
        cv2.createTrackbar('Mission Mode', 'Parameters',
                          0, 3, self._dummy_callback)
```

**변경 후:**
```python
        cv2.createTrackbar('Circle: TX MaxX', 'Parameters',
                          1200, 2000, self._dummy_callback)

        # 부표 사이 통과 미션 파라미터 트랙바
        cv2.createTrackbar('Pass: Max Depth Diff x10', 'Parameters',
                          50, 200, self._dummy_callback)  # 기본값 5.0m (50/10)
```

### 위치 1-2: update_parameters_from_trackbars() 메서드 (라인 146-152)

**변경 전:**
```python
        circle_tx_min_x = float(cv2.getTrackbarPos('Circle: TX MinX', 'Parameters'))
        circle_tx_max_x = float(cv2.getTrackbarPos('Circle: TX MaxX', 'Parameters'))

        # 미션 모드 선택
        mission_mode = cv2.getTrackbarPos('Mission Mode', 'Parameters')

        return {
```

**변경 후:**
```python
        circle_tx_min_x = float(cv2.getTrackbarPos('Circle: TX MinX', 'Parameters'))
        circle_tx_max_x = float(cv2.getTrackbarPos('Circle: TX MaxX', 'Parameters'))

        # 부표 사이 통과 미션 파라미터
        pass_max_depth_diff = cv2.getTrackbarPos('Pass: Max Depth Diff x10', 'Parameters') / 10.0

        return {
```

### 위치 1-3: return 딕셔너리 (라인 166-171)

**변경 전:**
```python
            'circle_tx_slope': circle_tx_slope,
            'circle_tx_min_x': circle_tx_min_x,
            'circle_tx_max_x': circle_tx_max_x,
            'mission_mode': mission_mode
        }
```

**변경 후:**
```python
            'circle_tx_slope': circle_tx_slope,
            'circle_tx_min_x': circle_tx_min_x,
            'circle_tx_max_x': circle_tx_max_x,
            'pass_max_depth_diff': pass_max_depth_diff
        }
```

---

## 수정 2: Main_MCP.py

### 위치 2-1: main_control_loop() 메서드의 미션 모드 결정 (라인 275-294)

**변경 전:**
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

**변경 후:**
```python
        # 자동 모드: 웨이포인트 기반 미션 전환
        self._check_waypoint_transition()
        current_mission_type = self.waypoint_manager.get_current_mission_type()
```

### 위치 2-2: _execute_current_mission() 메서드 (라인 354-360)

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

## 수정 3: utils/mission_strategies_new.py

### 위치 3-1: PassBetweenBuoysMission.execute() 메서드 (라인 49-54)

**변경 전:**
```python
        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 두 부표의 중점 계산 (이미지 좌표)
            red_x = red_buoy['center'][0]
            green_x = green_buoy['center'][0]
            midpoint_x = (red_x + green_x) / 2
```

**변경 후:**
```python
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
            # 두 부표의 중점 계산 (이미지 좌표)
            red_x = red_buoy['center'][0]
            green_x = green_buoy['center'][0]
            midpoint_x = (red_x + green_x) / 2
```

---

## 테스트 방법

1. 코드 수정 후 시스템 실행
2. Parameters 창에서 다음 확인:
   - ❌ "Mission Mode" 트랙바가 **제거**되었는지 확인
   - ✅ "Pass: Max Depth Diff x10" 트랙바가 **추가**되었는지 확인 (기본값 50 = 5.0m)
3. 부표 사이 통과 미션 실행 시:
   - 트랙바를 조정하면서 깊이 차이 필터링 동작 확인
   - 로그에서 "빨간 부표 무시" 또는 "초록 부표 무시" 메시지 확인
4. 시스템은 **자동 모드로만 동작** (웨이포인트 기반 미션 전환)

---

## 수정 완료 후 기능

### ✅ 깊이 차이 필터링
- 빨강/초록 부표의 깊이 차이가 `max_depth_diff` 초과 시 멀리 있는 부표 무시
- 트랙바로 실시간 조정 가능 (0.0m ~ 20.0m, 기본값 5.0m)
- 예: 깊이 차이 7.5m > 5.0m → 멀리 있는 부표 무시

### ✅ 자동 모드 전용
- mission_mode 트랙바 제거
- 웨이포인트 기반으로만 미션 전환
- 더 단순하고 명확한 동작

### ✅ 로그 메시지
```
빨간 부표 무시 (깊이 차이 7.50m > 5.00m)
Pass Buoys [RAW]: midpoint=640.0, error=0.0, steering=0.000
```

---

## 주의 사항

- **mission_mode 트랙바 제거로 인해 수동 모드 사용 불가**
- 모든 미션은 웨이포인트 순서대로 자동 전환
- 특정 미션만 테스트하려면 waypoint_manager.py에서 웨이포인트 순서 조정 필요
