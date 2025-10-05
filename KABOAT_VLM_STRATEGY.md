# KABOAT 2025 대회 VLM 활용 전략

## 📋 대회 임무 분석

### 5가지 핵심 임무

| 임무 | 핵심 요구사항 | VLM 활용 가능성 |
|------|-------------|----------------|
| **1. 장애물 회피** | 실시간 인식 및 회피 | ⭐⭐☆☆☆ (너무 느림) |
| **2. 위치유지** | 부표 5m 이내 5초 유지 | ⭐⭐☆☆☆ (제어 문제) |
| **3. 도킹** | 형상+색상 인식 후 접근 | ⭐⭐⭐⭐⭐ (가장 적합) |
| **4. 탐색** | 색상 인식 후 선회 | ⭐⭐⭐⭐☆ (적합) |
| **5. 항로추종** | 게이트 순차 통과 | ⭐⭐☆☆☆ (GPS가 나음) |

---

## 🎯 임무별 VLM 활용 분석

### ❌ VLM 부적합 임무

#### 1. 장애물 회피
**이유**:
- ⚠️ **속도 문제**: VLM 500ms-2s vs 필요 <100ms
- ⚠️ **실시간성**: 충돌 회피는 즉각 반응 필요
- ⚠️ **과도한 복잡도**: 단순 장애물 인식에 VLM은 오버킬

**권장**:
```python
# LiDAR + 전통적 알고리즘
obstacles = lidar.scan()
path = a_star(obstacles, goal)
boat.follow_path(path)
```

#### 2. 위치유지
**이유**:
- ⚠️ **제어 문제**: 위치 유지는 PID 제어기가 핵심
- ⚠️ **VLM 역할 제한적**: 부표 인식만 가능, 제어는 불가

**권장**:
```python
# OpenCV로 부표 인식
buoy_pos = detect_yellow_buoy(image)

# PID 제어
pid_controller.set_target(buoy_pos)
```

#### 3. 항로추종
**이유**:
- ⚠️ **GPS가 더 정확**: 게이트 순서는 GPS/경로계획
- ⚠️ **색상 인식만 필요**: OpenCV로 충분

**권장**:
```python
# GPS + OpenCV
gates = detect_red_green_pairs(image)
path = plan_path_through_gates(gates, gps)
```

---

### ✅ VLM 활용 가능 임무

#### 임무 3: 도킹 ⭐⭐⭐⭐⭐ (가장 적합)

**요구사항**:
- 3개 도킹 스테이션
- **형상**: 삼각형/원형/네모
- **색상**: 빨강/초록/파랑/주황/노랑
- 목표 표식 식별 후 도킹

**VLM 활용 방안**:

```python
# VLM으로 복잡한 패턴 인식
def find_target_dock(image, target_shape, target_color):
    prompt = f"Find {target_color} {target_shape} on dock. Which dock (left/center/right)?"

    result = vlm.predict(image, prompt)
    # "Center dock has red triangle"

    return parse_dock_position(result)

# 사용 예시
target = find_target_dock(camera_image, "triangle", "red")
boat.navigate_to_dock(target)
```

**장점**:
- ✅ 복잡한 형상+색상 조합 (15가지)
- ✅ 시간 여유 있음 (도킹은 천천히)
- ✅ 정확도 중요 (속도 덜 중요)

#### 임무 4: 탐색 ⭐⭐⭐⭐☆ (적합)

**요구사항**:
- 부표 색상 인식
- 색상별 선회 방향 결정
  - 빨강/초록 → 시계방향
  - 파랑 → 반시계방향

**VLM 활용 방안**:

```python
def determine_circling_direction(image):
    prompt = "What color is the buoy? Answer: red/green/blue only."

    color = vlm.predict(image, prompt)

    if color in ["red", "green"]:
        return "clockwise"
    elif color == "blue":
        return "counter_clockwise"

# 사용
direction = determine_circling_direction(camera_image)
boat.circle_buoy(direction)
```

---

## 🔥 현실적인 활용 전략

### 전략 1: 하이브리드 (전통적 + VLM 백업) ⭐⭐⭐⭐⭐

**95%는 전통적 방법, 5%만 VLM**

```python
class HybridDockingSystem:
    def __init__(self):
        self.traditional = OpenCVDetector()
        self.vlm = QwenVLM()

    def find_dock(self, image, target):
        # 1차: 전통적 방법 시도
        result = self.traditional.detect(image, target)

        if result.confidence > 0.8:
            return result  # 확신 있으면 바로 사용

        # 2차: 애매하면 VLM 사용
        vlm_result = self.vlm.predict(image,
            f"Find {target['color']} {target['shape']}")

        return vlm_result
```

**장점**:
- ✅ 빠름 (대부분 전통적 방법)
- ✅ 안전 (VLM은 백업)
- ✅ 신뢰성 높음

---

### 전략 2: 도킹만 VLM 사용 ⭐⭐⭐⭐

**임무 3(도킹)에만 집중 투자**

```python
# 도킹 임무
if mission == "docking":
    vlm_result = vlm.find_target_dock(image)
    navigate_to(vlm_result)

# 나머지 임무
else:
    traditional_method()
```

**장점**:
- ✅ VLM 강점 활용 (복잡한 패턴)
- ✅ 속도 문제 최소화 (도킹은 느려도 됨)
- ✅ 리스크 제한적 (한 임무만)

---

### 전략 3: VLM 완전 배제 (가장 안전) ⭐⭐⭐⭐⭐

**모든 임무를 전통적 방법으로**

```python
# 임무 3: 도킹
def find_dock_opencv(image, target_shape, target_color):
    # 색상 검출
    mask = detect_color(image, target_color)

    # 형상 검출
    contours = cv2.findContours(mask)

    for cnt in contours:
        shape = classify_shape(cnt)  # 삼각형/원/네모
        if shape == target_shape:
            return get_position(cnt)

# 임무 4: 탐색
def determine_direction_opencv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if is_red(hsv) or is_green(hsv):
        return "clockwise"
    elif is_blue(hsv):
        return "counter_clockwise"
```

**장점**:
- ✅ **100% 신뢰성** (검증된 방법)
- ✅ **매우 빠름** (<10ms)
- ✅ **디버깅 쉬움**
- ✅ **대회 안전**

**단점**:
- ❌ 수동 튜닝 필요 (HSV threshold 등)
- ❌ 조명 변화에 민감

---

## 📊 방법별 비교

| 방법 | 개발 시간 | 신뢰성 | 속도 | 대회 성공률 |
|------|----------|--------|------|-----------|
| **전통적 (OpenCV)** | 1-2일 | ⭐⭐⭐⭐⭐ | <10ms | 95% |
| **하이브리드** | 3-5일 | ⭐⭐⭐⭐☆ | 10-50ms | 90% |
| **VLM 위주** | 1-2주 | ⭐⭐⭐☆☆ | 500ms-2s | 70% |

---

## 🎯 최종 추천

### 📅 시간별 전략

#### 대회 1개월 전: 전략 3 (전통적 방법) ⭐⭐⭐⭐⭐

```bash
# 모든 임무를 OpenCV로
python traditional_vision_control.py
```

**이유**:
- 안정성 최우선
- 검증된 방법
- 디버깅 시간 충분

#### 대회 3개월 전: 전략 2 (도킹만 VLM) ⭐⭐⭐⭐

```python
# 도킹 임무에 VLM 실험
if mission == "docking":
    vlm_docking()
else:
    traditional_method()
```

**이유**:
- 실험 여유 있음
- VLM 장점 활용
- 리스크 관리 가능

#### 대회 6개월 전 + 연구 목적: 전략 1 (하이브리드) ⭐⭐⭐

```python
# 하이브리드 시스템 개발
hybrid_system = HybridDockingSystem()
```

**이유**:
- 학술적 가치
- 충분한 개발/테스트 시간
- 논문 작성 가능

---

## 💡 구체적 구현 방안

### 임무 3 (도킹) - VLM 활용 예시

```python
#!/usr/bin/env python3
"""
KABOAT 도킹 임무 - VLM 활용
"""

class KABOATDockingVLM:
    def __init__(self):
        # 전통적 방법
        self.opencv_detector = ShapeColorDetector()

        # VLM (백업용)
        self.vlm = Qwen2VLAnalyzer(use_quantization=True)

    def find_target_dock(self, image, target_shape, target_color):
        """목표 도킹 스테이션 찾기"""

        # 1차: OpenCV 시도
        opencv_result = self.opencv_detector.detect(
            image, target_shape, target_color
        )

        if opencv_result.confidence > 0.85:
            return opencv_result

        # 2차: VLM 사용 (애매한 경우)
        prompt = f"Which dock has {target_color} {target_shape}? Answer: left/center/right only."

        vlm_answer = self.vlm.analyze_image(image, prompt)

        return parse_dock_position(vlm_answer)

# ROS2 노드
class DockingNode(Node):
    def __init__(self):
        self.docking_vlm = KABOATDockingVLM()
        self.target = {"shape": "triangle", "color": "red"}

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg)

        dock_pos = self.docking_vlm.find_target_dock(
            image,
            self.target["shape"],
            self.target["color"]
        )

        self.navigate_to_dock(dock_pos)
```

---

## ⚠️ VLM 사용 시 주의사항

### 1. 속도 문제

```python
# 타임아웃 설정 필수
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("VLM too slow")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(2)  # 2초 타임아웃

try:
    result = vlm.predict(image)
except TimeoutError:
    result = fallback_traditional_method()
```

### 2. 신뢰성 검증

```python
# VLM 결과 검증
vlm_result = vlm.predict(image)

# 전통적 방법으로 교차 검증
opencv_result = opencv.detect(image)

if vlm_result != opencv_result:
    logger.warning("VLM mismatch!")
    return opencv_result  # 안전하게
```

### 3. 폴백 전략

```python
# 항상 백업 방법 준비
try:
    return vlm_method()
except Exception:
    return traditional_method()  # 폴백
```

---

## 🏆 성공 전략

### Do ✅

1. **전통적 방법 먼저 완성** (OpenCV)
2. **VLM은 보조/백업으로만**
3. **철저한 테스트** (시뮬레이터)
4. **폴백 전략 필수**

### Don't ❌

1. **VLM만 의존하지 말 것**
2. **실시간 제어에 VLM 사용 금지**
3. **검증 없이 대회 투입 금지**

---

## 📚 최종 결론

### Q: KABOAT에서 VLM을 어떻게 활용할까?

**A: 전통적 방법이 90%, VLM은 10% 백업**

### 추천 우선순위

1. 🥇 **전통적 방법 (OpenCV + GPS)** - 모든 임무
2. 🥈 **VLM 백업** - 도킹 임무만 (애매한 경우)
3. 🥉 **VLM 연구** - 시간 여유 있을 때

### 실행 파일

```bash
# 전통적 방법 (추천)
python traditional_vision_control.py

# 하이브리드 (도전)
python hybrid_kaboat_system.py

# VLM만 (연구용)
python vlm_qwen_node.py
```

**현실적 조언**: VLM은 "있으면 좋지만 없어도 되는" 기능입니다. **대회 성공의 핵심은 전통적 방법의 완성도**입니다.
