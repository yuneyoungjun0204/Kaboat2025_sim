#!/bin/bash

echo "부표 미션 수정 시작..."

# 파일 백업
cp utils/mission_strategies_new.py utils/mission_strategies_new.py.backup_before_final_fix
cp utils/visualization_system.py utils/visualization_system.py.backup_before_final_fix

echo "1. mission_strategies_new.py 수정 중..."

# 이미 *0.05 제거는 완료됨

# fallback 로직 변경 (45-72 라인)
python3 << 'EOF'
import re

# 파일 읽기
with open('utils/mission_strategies_new.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. fallback 로직 교체
old_fallback = r'''        # 빨간색/초록색 부표 찾기 \(원본 탐지값 우선 사용\)
        red_buoy = None
        green_buoy = None
        data_source = "RAW"

        # 1\. 원본 탐지값\(raw\)에서 먼저 찾기
        if raw_detections:
            for det in raw_detections:
                if det\['label'\] == 'red_cone':
                    red_buoy = det
                elif det\['label'\] == 'green_cone':
                    green_buoy = det

        # 2\. 원본값에 없으면 추적값\(tracked\)에서 찾기 \(fallback\)
        if \(not red_buoy or not green_buoy\) and detected_objects:
            data_source = "TRACKED"
            if not red_buoy:
                for det in detected_objects:
                    if det\['label'\] == 'red_cone':
                        red_buoy = det
                        break
            if not green_buoy:
                for det in detected_objects:
                    if det\['label'\] == 'green_cone':
                        green_buoy = det
                        break
            if logger and \(red_buoy or green_buoy\):
                logger\.info\("⚠️ 원본값 없음 -> 추적값 사용 \(fallback\)"\)'''

new_fallback = '''        # 빨간색/초록색 부표 찾기 (추정값 우선, 개별 fallback)
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
                    break'''

content = re.sub(old_fallback, new_fallback, content)

# 2. 필터링 플래그 추가
old_filter = r'''                # 깊이 차이가 너무 크면 멀리 있는 부표 무시
                if red_depth > green_depth:
                    if logger:
                        logger\.warn\(f"빨간 부표 무시 \(깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m\)"\)
                    red_buoy = None
                else:
                    if logger:
                        logger\.warn\(f"초록 부표 무시 \(깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m\)"\)
                    green_buoy = None'''

new_filter = '''                # 깊이 차이가 너무 크면 멀리 있는 부표 무시하고 필터링 플래그 추가
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
                    green_buoy = None'''

content = re.sub(old_filter, new_filter, content)

# 3. 로그 메시지 업데이트
old_log = r'''            if logger:
                logger\.info\(
                    f"Pass Buoys \[{data_source}\]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                \)'''

new_log = '''            if logger:
                data_source = f"R:{red_source}/G:{green_source}"
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )'''

content = re.sub(old_log, new_log, content)

# 파일 쓰기
with open('utils/mission_strategies_new.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ mission_strategies_new.py 수정 완료")
EOF

echo "2. visualization_system.py 수정 중..."

python3 << 'EOF'
import re

# 파일 읽기
with open('utils/visualization_system.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 원본 탐지 그리기 부분 수정
old_raw = r'''        # 원본 탐지 그리기 \(얇은 점선\)
        if raw_detections:
            for det in raw_detections:
                x1, y1, x2, y2 = det\["bbox"\]'''

new_raw = '''        # 원본 탐지 그리기 (얇은 점선)
        if raw_detections:
            for det in raw_detections:
                # pass_max_depth_diff로 필터링된 객체는 그리지 않음
                if det.get('filtered_by_depth_diff', False):
                    continue

                x1, y1, x2, y2 = det["bbox"]'''

content = re.sub(old_raw, new_raw, content)

# 추적 결과 그리기 부분 수정
old_tracked = r'''        # 추적 결과 그리기 \(굵은 실선\)
        for det in detections:
            x1, y1, x2, y2 = det\["bbox"\]'''

new_tracked = '''        # 추적 결과 그리기 (굵은 실선)
        for det in detections:
            # pass_max_depth_diff로 필터링된 객체는 그리지 않음
            if det.get('filtered_by_depth_diff', False):
                continue

            x1, y1, x2, y2 = det["bbox"]'''

content = re.sub(old_tracked, new_tracked, content)

# 파일 쓰기
with open('utils/visualization_system.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ visualization_system.py 수정 완료")
EOF

echo ""
echo "=" * 80
echo "모든 수정 완료!"
echo "=" * 80
echo "수정된 파일:"
echo "  - utils/mission_strategies_new.py"
echo "  - utils/visualization_system.py"
echo ""
echo "백업 파일:"
echo "  - utils/mission_strategies_new.py.backup_before_final_fix"
echo "  - utils/visualization_system.py.backup_before_final_fix"
