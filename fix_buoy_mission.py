#!/usr/bin/env python3
"""
부표 미션 수정 스크립트
1. 추정값/측정값 fallback을 RED/GREEN 개별로 처리
2. pass_max_depth_diff 스케일 통일 (*0.05 제거)
3. 필터링된 객체에 플래그 추가
4. visualization_system.py에서 필터링된 객체 박스 안 그리기
"""

import re

def fix_mission_strategies():
    """mission_strategies_new.py 수정"""
    filepath = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/mission_strategies_new.py'

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 수정 1: 개별 fallback 로직으로 변경
    old_fallback = '''        # 빨간색/초록색 부표 찾기 (원본 탐지값 우선 사용)
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
                logger.info("⚠️ 원본값 없음 -> 추적값 사용 (fallback)")'''

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

    content = content.replace(old_fallback, new_fallback)

    # 수정 2: *0.05 제거 및 필터링 플래그 추가
    old_filter = '''            # 깊이 차이 필터링 (너무 차이 나면 멀리 있는 것 무시)
            max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)*0.05
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
                    green_buoy = None'''

    new_filter = '''            # 깊이 차이 필터링 (너무 차이 나면 멀리 있는 것 무시)
            max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)
            red_depth = red_buoy.get('depth', 0.0)
            green_depth = green_buoy.get('depth', 0.0)
            depth_diff = abs(red_depth - green_depth)

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
                    green_buoy = None'''

    content = content.replace(old_filter, new_filter)

    # 수정 3: 로그 메시지 업데이트
    old_log = '''            if logger:
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )'''

    new_log = '''            if logger:
                data_source = f"R:{red_source}/G:{green_source}"
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )'''

    content = content.replace(old_log, new_log)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ mission_strategies_new.py 수정 완료")
    print("  - 추정값/측정값 개별 fallback 적용")
    print("  - pass_max_depth_diff 스케일 통일 (*0.05 제거)")
    print("  - 필터링 플래그 추가")

def fix_visualization():
    """visualization_system.py 수정 - 필터링된 객체 박스 안 그리기"""
    filepath = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/visualization_system.py'

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 원본 탐지 그리기 부분 수정
    old_raw = '''        # 원본 탐지 그리기 (얇은 점선)
        if raw_detections:
            for det in raw_detections:
                x1, y1, x2, y2 = det["bbox"]
                label = det["label"]
                cx, cy = det["center"]

                color = self.colors.get(label, (255, 255, 255))

                # 얇은 점선 박스
                self._draw_dashed_rectangle(vis_image, (x1, y1), (x2, y2), color, 1)

                # 작은 원
                cv2.circle(vis_image, (cx, cy), 3, color, 1)'''

    new_raw = '''        # 원본 탐지 그리기 (얇은 점선)
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
                cv2.circle(vis_image, (cx, cy), 3, color, 1)'''

    content = content.replace(old_raw, new_raw)

    # 추적 결과 그리기 부분 수정
    old_tracked = '''        # 추적 결과 그리기 (굵은 실선)
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
            cv2.circle(vis_image, (cx, cy), 7, color, -1)'''

    new_tracked = '''        # 추적 결과 그리기 (굵은 실선)
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
            cv2.circle(vis_image, (cx, cy), 7, color, -1)'''

    content = content.replace(old_tracked, new_tracked)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ visualization_system.py 수정 완료")
    print("  - pass_max_depth_diff로 필터링된 객체 박스 안 그리기")

if __name__ == '__main__':
    print("=" * 60)
    print("부표 미션 수정 스크립트 실행")
    print("=" * 60)

    fix_mission_strategies()
    print()
    fix_visualization()

    print()
    print("=" * 60)
    print("모든 수정 완료!")
    print("=" * 60)
