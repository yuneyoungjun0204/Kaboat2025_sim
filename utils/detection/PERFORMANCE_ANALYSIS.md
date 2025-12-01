# IMM-PDAF Tracker 성능 분석 및 최적화 가이드

## 성능 병목 지점

### 1. 행렬 역행렬 계산 (최우선 최적화)
**위치**: `np.linalg.inv(S)` - 5곳에서 호출
- Line 224: Kalman gain 계산
- Line 309: Likelihood 계산 (IMM update)
- Line 355: Validation gate
- Line 383: Association probabilities
- Line 439: PDAF update

**최적화 방법**:
```python
# 기존 (느림)
S_inv = np.linalg.inv(S)
mahala_dist_sq = innovation.T @ S_inv @ innovation

# 최적화 (빠름) - Cholesky 분해 사용
L, lower = cho_factor(S)
S_inv = cho_solve((L, lower), np.eye(S.shape[0]))
# 또는 역행렬 캐싱
```

**예상 개선**: 30-40% 속도 향상

### 2. 행렬식 재계산 제거
**위치**: `np.linalg.det(S)` - 3곳에서 호출
- Line 307, 376, 381

**최적화 방법**:
```python
# 한 번만 계산하고 재사용
det_S = np.linalg.det(S)
# 이후 모든 곳에서 det_S 재사용
```

**예상 개선**: 10-15% 속도 향상

### 3. Validation Gate 최적화
**위치**: Line 556-560

**현재 문제**:
- 각 측정값마다 `get_innovation()` 호출
- 각 측정값마다 `validation_gate()` 호출 (역행렬 계산 포함)

**최적화 방법**:
```python
# 역행렬을 한 번만 계산
y, S, S_inv = kf.get_innovation_cached(z, H, R)
# 이후 모든 측정값에 대해 S_inv 재사용
for z in z_list:
    y_i = z - H @ kf.x
    if validation_gate_optimized(y_i, S_inv):
        validated_measurements.append(z)
```

**예상 개선**: 측정값 많을 때 20-30% 속도 향상

### 4. IMM 필터 최적화
**위치**: 전체 IMM 구조

**현재 문제**:
- 4개 모델을 모두 업데이트 (항상)
- 각 모델마다 동일한 계산 반복

**최적화 방법**:
- 모델 확률이 낮은 모델은 간소화된 업데이트
- 병렬 처리 (멀티스레딩)

**예상 개선**: 10-20% 속도 향상

## 전체 최적화 효과

| 최적화 항목 | 개별 개선 | 누적 효과 |
|-----------|---------|---------|
| 역행렬 캐싱 | 30-40% | 30-40% |
| 행렬식 재사용 | 10-15% | 40-50% |
| Validation Gate | 20-30% | 50-60% |
| IMM 최적화 | 10-20% | **60-70%** |

**최종 예상 성능**:
- 기존: 8-15ms/프레임 → 최적화 후: **3-6ms/프레임**

## 실용적인 최적화 전략

### 즉시 적용 가능 (Low-hanging fruit)

1. **역행렬 캐싱 추가** (5분 작업)
   - `get_innovation()` 메서드에 캐시 추가
   - Validation gate에서 캐시된 역행렬 사용

2. **행렬식 재사용** (2분 작업)
   - `det_S`를 한 번만 계산하고 전달

3. **측정값 사전 필터링** (10분 작업)
   - 거리 기반 간단한 필터링으로 validation gate 호출 감소

### 중기 최적화

4. **Cholesky 분해 사용** (30분 작업)
   - `np.linalg.inv()` 대신 `scipy.linalg.cho_solve()` 사용

5. **벡터화 개선** (1시간 작업)
   - 반복문을 벡터 연산으로 변환

### 장기 최적화

6. **GPU 가속** (CuPy 사용)
   - 행렬 연산을 GPU로 이동
   - 예상: 5-10배 속도 향상

7. **Cython/Numba 컴파일**
   - 핵심 루프를 컴파일된 코드로 변환
   - 예상: 2-3배 속도 향상

## 성능 모니터링

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.times = {}
    
    def time_function(self, func_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000  # ms
                if func_name not in self.times:
                    self.times[func_name] = []
                self.times[func_name].append(elapsed)
                return result
            return wrapper
        return decorator
    
    def get_stats(self):
        for name, times in self.times.items():
            print(f"{name}: {np.mean(times):.2f}ms (avg), {np.max(times):.2f}ms (max)")
```

## 권장 사항

1. **즉시 적용**: 역행렬 캐싱 + 행렬식 재사용 (약 40-50% 개선)
2. **단기**: Validation Gate 최적화 (추가 20-30% 개선)
3. **중기**: Cholesky 분해 사용 (추가 10-15% 개선)
4. **장기**: GPU 가속 고려 (5-10배 개선 가능)

**총 예상 개선**: 60-70% 속도 향상 (8-15ms → 3-6ms)

