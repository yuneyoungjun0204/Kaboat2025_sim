# 부표 미션 수정 사항 요약

## 문제 분석

### 현재 문제점
1. **추정값/측정값 Fallback 로직 오류**: 빨간색과 초록색 부표를 개별로 판단하지 않고 둘 중 하나라도 없으면 둘 다 fallback 처리
2. **pass_max_depth_diff 스케일 불일치**: `*0.05`를 곱하여 max_depth/min_depth와 스케일이 다름
3. **필터링 적용 여부 불명확**: 필터링이 실제로 적용되는지 확인 필요
4. **시각화 문제**: 필터링된 부표도 박스가 그려짐

## 수정 사항

### 1. mission_strategies_new.py - PassBetweenBuoysMission.execute() 메서드

#### 수정 전 (lines 45-72):
```python
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
if (not red_buoy or not green_buoy) and detected_objects:  # ← 문제: 둘 다 함께 fallback
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
        logger.info("⚠️ 원본값 없음 -> 추정값 사용 (fallback)")
```

#### 수정 후:
```python
# 빨간색/초록색 부표 찾기 (추정값 우선, 개별 fallback)
red_buoy = None
green_buoy = None
red_source = None
green_source = None

# 1. 빨간색 부표: 추정값(tracked) 우선, 없으면 측정값(raw) 사용
if detected_objects:
    for det in detected_objects:
        if det['label'] == 'red_cone':
            red_buoy = det
            red_source = "TRACKED"
            break

if not red_buoy and raw_detections:
    for det in raw_detections:
        if det['label'] == 'red_cone':
            red_buoy = det
            red_source = "RAW"
            if logger:
                logger.info("⚠️ 빨간색 부표: 추정값 없음 -> 측정값 사용")
            break

# 2. 초록색 부표: 추정값(tracked) 우선, 없으면 측정값(raw) 사용
if detected_objects:
    for det in detected_objects:
        if det['label'] == 'green_cone':
            green_buoy = det
            green_source = "TRACKED"
            break

if not green_buoy and raw_detections:
    for det in raw_detections:
        if det['label'] == 'green_cone':
            green_buoy = det
            green_source = "RAW"
            if logger:
                logger.info("⚠️ 초록색 부표: 추정값 없음 -> 측정값 사용")
            break
```

**변경 내용:**
- 빨간색/초록색 부표를 각각 개별적으로 추정값 → 측정값 fallback 처리
- 각 부표의 데이터 소스를 red_source, green_source로 추적

#### 수정 전 (line 77):
```python
max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)*0.05
```

#### 수정 후:
```python
max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)
```

**변경 내용:**
- `*0.05` 제거하여 max_depth/min_depth와 동일한 스케일 사용

#### 수정 전 (lines 82-91):
```python
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
```

#### 수정 후:
```python
if depth_diff > max_depth_diff:
    # 깊이 차이가 너무 크면 멀리 있는 부표 무시하고 필터링 플래그 추가
    if red_depth > green_depth:
        if logger:
            logger.warn(f"빨간 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)")
        # 필터링 플래그 추가
        red_buoy['filtered_by_depth_diff'] = True
        red_buoy = None
    else:
        if logger:
            logger.warn(f"초록 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)")
        # 필터링 플래그 추가
        green_buoy['filtered_by_depth_diff'] = True
        green_buoy = None
```

**변경 내용:**
- 필터링된 부표에 `filtered_by_depth_diff = True` 플래그 추가
- 이 플래그를 시각화 시스템에서 사용하여 박스를 그리지 않음

#### 수정 전 (lines 117-120):
```python
if logger:
    logger.info(
        f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
    )
```

#### 수정 후:
```python
if logger:
    data_source = f"R:{red_source}/G:{green_source}"
    logger.info(
        f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
    )
```

**변경 내용:**
- 로그 메시지에서 각 부표의 소스를 개별적으로 표시 (예: "R:TRACKED/G:RAW")

### 2. visualization_system.py - visualize_detections() 메서드

#### 수정 전 (lines 216-228):
```python
# 원본 탐지 그리기 (얇은 점선)
if raw_detections:
    for det in raw_detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        cx, cy = det["center"]

        color = self.colors.get(label, (255, 255, 255))

        # 얇은 점선 박스
        self._draw_dashed_rectangle(vis_image, (x1, y1), (x2, y2), color, 1)

        # 작은 원
        cv2.circle(vis_image, (cx, cy), 3, color, 1)
```

#### 수정 후:
```python
# 원본 탐지 그리기 (얇은 점선)
if raw_detections:
    for det in raw_detections:
        # pass_max_depth_diff로 필터링된 객체는 그리지 않음
        if det.get('filtered_by_depth_diff', False):
            continue

        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        cx, cy = det["center"]

        color = self.colors.get(label, (255, 255, 255))

        # 얇은 점선 박스
        self._draw_dashed_rectangle(vis_image, (x1, y1), (x2, y2), color, 1)

        # 작은 원
        cv2.circle(vis_image, (cx, cy), 3, color, 1)
```

**변경 내용:**
- 필터링된 객체(`filtered_by_depth_diff == True`)는 continue로 건너뛰어 박스를 그리지 않음

#### 수정 전 (lines 231-244):
```python
# 추적 결과 그리기 (굵은 실선)
for det in detections:
    x1, y1, x2, y2 = det["bbox"]
    label = det["label"]
    conf = det["confidence"]
    depth = det["depth"]
    cx, cy = det["center"]

    color = self.colors.get(label, (255, 255, 255))

    # 굵은 실선 박스
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

    # 큰 중심점
    cv2.circle(vis_image, (cx, cy), 7, color, -1)
```

#### 수정 후:
```python
# 추적 결과 그리기 (굵은 실선)
for det in detections:
    # pass_max_depth_diff로 필터링된 객체는 그리지 않음
    if det.get('filtered_by_depth_diff', False):
        continue

    x1, y1, x2, y2 = det["bbox"]
    label = det["label"]
    conf = det["confidence"]
    depth = det["depth"]
    cx, cy = det["center"]

    color = self.colors.get(label, (255, 255, 255))

    # 굵은 실선 박스
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

    # 큰 중심점
    cv2.circle(vis_image, (cx, cy), 7, color, -1)
```

**변경 내용:**
- 필터링된 객체(`filtered_by_depth_diff == True`)는 continue로 건너뛰어 박스를 그리지 않음

## 적용 방법

### 자동 적용 (권장)
```bash
cd /home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git
python3 apply_comprehensive_fix.py
```

### 수동 적용
1. `utils/mission_strategies_new.py` 파일을 열어서 위의 수정 사항을 적용
2. `utils/visualization_system.py` 파일을 열어서 위의 수정 사항을 적용

## 검증 방법

1. 로그 확인:
   - "⚠️ 빨간색 부표: 추정값 없음 -> 측정값 사용" 메시지 확인
   - "⚠️ 초록색 부표: 추정값 없음 -> 측정값 사용" 메시지 확인
   - "Pass Buoys [R:TRACKED/G:RAW]: ..." 형식의 로그 확인

2. 필터링 확인:
   - pass_max_depth_diff 트랙바 값 조정
   - 깊이 차이가 임계값을 초과할 때 경고 메시지 확인
   - 필터링된 부표의 박스가 화면에 그려지지 않는지 확인

3. 스케일 확인:
   - pass_max_depth_diff=5.0일 때 5.0m 기준으로 동작하는지 확인 (이전에는 0.25m로 동작)
   - max_depth, min_depth와 동일한 단위로 동작하는지 확인

## 파일 위치

- `/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/mission_strategies_new.py`
- `/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/visualization_system.py`
- `/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/apply_comprehensive_fix.py` (자동 적용 스크립트)
