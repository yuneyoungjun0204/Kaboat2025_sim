# 코드 수정 가이드

## 문제 해결 사항
1. 모드 수동 변경 시 장애물 단계 진입 확인 로그 추가
2. 탐지 시 추정값 대신 원본값 우선 사용

---

## 수정 1: Main_MCP.py

### 1-1. `_init_sensor_data()` 메서드 (라인 135-149)

**변경 전:**
```python
        # 수동 목표 위치 (trajectory_viz에서 클릭한 좌표)
        self.manual_target_x = None
        self.manual_target_y = None
```

**변경 후:**
```python
        # 수동 목표 위치 (trajectory_viz에서 클릭한 좌표)
        self.manual_target_x = None
        self.manual_target_y = None

        # 수동 모드 추적용
        self._last_manual_mode = 0
```

### 1-2. `main_control_loop()` 메서드의 미션 모드 결정 부분 (라인 284-293)

**변경 전:**
```python
        elif mission_mode == 1:
            # 수동 장애물 회피 모드
            current_mission_type = MissionType.OBSTACLE_AVOID
        elif mission_mode == 2:
            # 수동 부표 사이 통과 모드
            current_mission_type = MissionType.PASS_BETWEEN_BUOYS
        elif mission_mode == 3:
            # 수동 부표 회전 모드
            current_mission_type = MissionType.CIRCLE_BUOY
```

**변경 후:**
```python
        elif mission_mode == 1:
            # 수동 장애물 회피 모드
            current_mission_type = MissionType.OBSTACLE_AVOID
            if hasattr(self, '_last_manual_mode') and self._last_manual_mode != 1:
                self.get_logger().info("🔵 [수동 모드 전환] 장애물 회피 미션 진입!")
            self._last_manual_mode = 1
        elif mission_mode == 2:
            # 수동 부표 사이 통과 모드
            current_mission_type = MissionType.PASS_BETWEEN_BUOYS
            if hasattr(self, '_last_manual_mode') and self._last_manual_mode != 2:
                self.get_logger().info("🔴 [수동 모드 전환] 부표 사이 통과 미션 진입!")
            self._last_manual_mode = 2
        elif mission_mode == 3:
            # 수동 부표 회전 모드
            current_mission_type = MissionType.CIRCLE_BUOY
            if hasattr(self, '_last_manual_mode') and self._last_manual_mode != 3:
                self.get_logger().info("🔵 [수동 모드 전환] 부표 회전 미션 진입!")
            self._last_manual_mode = 3
```

### 1-3. `_execute_current_mission()` 메서드 (라인 354-360)

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
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                raw_detections=self.raw_detections,  # 원본 탐지값 추가
                logger=self.get_logger()
            )
```

---

## 수정 2: utils/mission_strategies_new.py

### 2-1. PassBetweenBuoysMission 클래스의 execute() 메서드 (라인 36-47)

**변경 전:**
```python
    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, **kwargs) -> Tuple[float, float]:
        """빨간색/초록색 고깔 부표 사이로 지나가기"""
        # 빨간색/초록색 부표 찾기
        red_buoy = None
        green_buoy = None

        for det in detected_objects:
            if det['label'] == 'red_cone':
                red_buoy = det
            elif det['label'] == 'green_cone':
                green_buoy = det
```

**변경 후:**
```python
    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, raw_detections: List[Dict] = None, **kwargs) -> Tuple[float, float]:
        """
        빨간색/초록색 고깔 부표 사이로 지나가기

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)
        """
        # 빨간색/초록색 부표 찾기 (원본 탐지값 우선 사용)
        red_buoy = None
        green_buoy = None
        data_source = "RAW"  # 기본적으로 원본 탐지값 사용

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
```

### 2-2. PassBetweenBuoysMission 클래스의 로그 출력 부분 (라인 73-76)

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

## 테스트 방법

1. 코드 수정 후 시스템 실행
2. Parameters 창에서 'Mission Mode' 트랙바를 0→1→2→3으로 변경
3. 로그에서 다음 메시지 확인:
   - "🔵 [수동 모드 전환] 장애물 회피 미션 진입!"
   - "🔴 [수동 모드 전환] 부표 사이 통과 미션 진입!"
   - "🔵 [수동 모드 전환] 부표 회전 미션 진입!"
4. 부표 탐지 시 로그에서 `[RAW]` 또는 `[TRACKED]` 표시 확인

## 수정 완료 후

이제 다음이 가능합니다:
- ✅ 수동 모드 변경 시 미션 진입 여부를 명확하게 확인 (특히 장애물 회피)
- ✅ 탐지 시 원본 측정값을 우선 사용하여 더 정확한 제어
- ✅ Fallback 메커니즘으로 끊김 방지 (원본값 없을 때 추적값 사용)
