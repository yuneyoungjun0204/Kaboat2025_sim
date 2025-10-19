#!/usr/bin/env python3
"""모든 파일 수정 스크립트"""
import shutil
from datetime import datetime

def fix_mission_strategies():
    """mission_strategies_new.py 수정"""
    mission_file = "utils/mission_strategies_new.py"

    # 백업
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = mission_file + f".backup_{timestamp}"
    shutil.copy2(mission_file, backup_file)
    print(f"✅ 백업: {backup_file}")

    # 파일 읽기
    with open(mission_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 수정 1: 주석 변경
    content = content.replace(
        '        # 빨간색/초록색 부표 찾기 (원본 탐지값 우선 사용)',
        '        # 빨간색/초록색 부표 찾기 (추정값 우선, 개별 처리)'
    )

    # 수정 2: 변수 및 로직 변경
    old_logic = '''        red_buoy = None
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

    new_logic = '''        red_buoy = None
        green_buoy = None
        red_source = None
        green_source = None

        # 1. 추정값(tracked)에서 먼저 찾기 - 각각 개별로
        if detected_objects:
            for det in detected_objects:
                if det['label'] == 'red_cone' and not red_buoy:
                    red_buoy = det
                    red_source = "TRACKED"
                elif det['label'] == 'green_cone' and not green_buoy:
                    green_buoy = det
                    green_source = "TRACKED"

        # 2. 추정값에 없는 것만 측정값(raw)에서 찾기 - 각각 개별로
        if raw_detections:
            for det in raw_detections:
                if det['label'] == 'red_cone' and not red_buoy:
                    red_buoy = det
                    red_source = "RAW"
                    if logger:
                        logger.info("⚠️ 빨간색 추정값 없음 -> 측정값 사용")
                elif det['label'] == 'green_cone' and not green_buoy:
                    green_buoy = det
                    green_source = "RAW"
                    if logger:
                        logger.info("⚠️ 초록색 추정값 없음 -> 측정값 사용")'''

    content = content.replace(old_logic, new_logic)

    # 수정 3: pass_max_depth_diff 변수명 및 스케일
    content = content.replace(
        "max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)*0.05",
        "pass_max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)"
    )

    # 수정 4: 깊이 로깅 추가
    old_depth_check = '''            red_depth = red_buoy.get('depth', 0.0)
            green_depth = green_buoy.get('depth', 0.0)
            depth_diff = abs(red_depth - green_depth)

            if depth_diff > max_depth_diff:'''

    new_depth_check = '''            red_depth = red_buoy.get('depth', 0.0)
            green_depth = green_buoy.get('depth', 0.0)
            depth_diff = abs(red_depth - green_depth)

            if logger:
                logger.info(f"깊이 비교: 빨강={red_depth:.2f}m ({red_source}), 초록={green_depth:.2f}m ({green_source}), 차이={depth_diff:.2f}m, 임계값={pass_max_depth_diff:.2f}m")

            if depth_diff > pass_max_depth_diff:'''

    content = content.replace(old_depth_check, new_depth_check)

    # 수정 5: 필터링 시 변수명 수정 및 source None 처리
    old_filter = '''                    if logger:
                        logger.warn(f"빨간 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)")
                    red_buoy = None
                else:
                    if logger:
                        logger.warn(f"초록 부표 무시 (깊이 차이 {depth_diff:.2f}m > {max_depth_diff:.2f}m)")
                    green_buoy = None'''

    new_filter = '''                    if logger:
                        logger.warn(f"빨간 부표 무시 (깊이 차이 {depth_diff:.2f}m > {pass_max_depth_diff:.2f}m)")
                    red_buoy = None
                    red_source = None
                else:
                    if logger:
                        logger.warn(f"초록 부표 무시 (깊이 차이 {depth_diff:.2f}m > {pass_max_depth_diff:.2f}m)")
                    green_buoy = None
                    green_source = None'''

    content = content.replace(old_filter, new_filter)

    # 수정 6: 로그 출력 변경
    content = content.replace(
        'f"Pass Buoys [{data_source}]:',
        'f"Pass Buoys [R:{red_source},G:{green_source}]:'
    )

    # 파일 쓰기
    if content != original:
        with open(mission_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {mission_file} 수정 완료!")
        return True
    else:
        print(f"⚠️  {mission_file} 변경사항 없음")
        return False

def fix_visualization_system():
    """visualization_system.py 수정"""
    viz_file = "utils/visualization_system.py"

    # 백업
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = viz_file + f".backup_{timestamp}"
    shutil.copy2(viz_file, backup_file)
    print(f"✅ 백업: {backup_file}")

    # 파일 읽기
    with open(viz_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 수정 1: 함수 시그니처에 mission_params 추가
    content = content.replace(
        'bridge=None, viz_image_pub=None):',
        'bridge=None, viz_image_pub=None, mission_params: Optional[Dict] = None):'
    )

    # 수정 2: docstring 업데이트
    content = content.replace(
        '''            viz_image_pub: 시각화 이미지 퍼블리셔 (선택)
        """''',
        '''            viz_image_pub: 시각화 이미지 퍼블리셔 (선택)
            mission_params: 미션 파라미터 (선택)
        """'''
    )

    # 수정 3: 필터링 로직 추가
    old_vis_start = '''        if image is None:
            return

        vis_image = image.copy()

        # 원본 탐지 그리기 (얇은 점선)'''

    new_vis_start = '''        if image is None:
            return

        vis_image = image.copy()

        # 부표 사이 통과 미션일 때 깊이 필터링 확인
        filtered_labels = set()
        if mission_name == "PASS_BETWEEN_BUOYS" and mission_params:
            pass_max_depth_diff = mission_params.get('pass_max_depth_diff', 5.0)

            # 빨간색과 초록색 부표 찾기
            red_buoy = None
            green_buoy = None
            for det in detections:
                if det['label'] == 'red_cone':
                    red_buoy = det
                elif det['label'] == 'green_cone':
                    green_buoy = det

            # 깊이 차이 확인
            if red_buoy and green_buoy:
                red_depth = red_buoy.get('depth', 0.0)
                green_depth = green_buoy.get('depth', 0.0)
                depth_diff = abs(red_depth - green_depth)

                if depth_diff > pass_max_depth_diff:
                    # 멀리 있는 부표 필터링 표시
                    if red_depth > green_depth:
                        filtered_labels.add('red_cone')
                    else:
                        filtered_labels.add('green_cone')

        # 원본 탐지 그리기 (얇은 점선)'''

    content = content.replace(old_vis_start, new_vis_start)

    # 수정 4: 추적 결과 그리기 부분에 필터링 처리 추가
    old_tracking = '''        # 추적 결과 그리기 (굵은 실선)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            depth = det["depth"]
            cx, cy = det["center"]

            color = self.colors.get(label, (255, 255, 255))

            # 굵은 실선 박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)'''

    new_tracking = '''        # 추적 결과 그리기 (굵은 실선)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            depth = det["depth"]
            cx, cy = det["center"]

            color = self.colors.get(label, (255, 255, 255))

            # 필터링된 객체는 X 표시만 그리고 건너뛰기
            if label in filtered_labels:
                # X 표시 그리기 (회색)
                cv2.line(vis_image, (x1, y1), (x2, y2), (128, 128, 128), 3)
                cv2.line(vis_image, (x2, y1), (x1, y2), (128, 128, 128), 3)
                # 텍스트 표시 (FILTERED)
                cv2.putText(vis_image, f"{label} FILTERED", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                continue  # 박스 그리지 않고 건너뛰기

            # 굵은 실선 박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)'''

    content = content.replace(old_tracking, new_tracking)

    # 수정 5: 범례에 필터링 설명 추가
    old_legend = '''        cv2.putText(vis_image, "Dashed box + small dot = Raw detection", (15, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 화면 표시'''

    new_legend = '''        cv2.putText(vis_image, "Dashed box + small dot = Raw detection", (15, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        legend_y += 20
        cv2.putText(vis_image, "X mark (gray) = Depth filtered", (15, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # 화면 표시'''

    content = content.replace(old_legend, new_legend)

    # 파일 쓰기
    if content != original:
        with open(viz_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {viz_file} 수정 완료!")
        return True
    else:
        print(f"⚠️  {viz_file} 변경사항 없음")
        return False

def fix_main_mcp():
    """Main_MCP.py 수정"""
    main_file = "Main_MCP.py"

    # 백업
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = main_file + f".backup_{timestamp}"
    shutil.copy2(main_file, backup_file)
    print(f"✅ 백업: {backup_file}")

    # 파일 읽기
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 수정: visualize_detections 호출 전에 mission_params 추가
    old_viz_call = '''        # 탐지 결과 시각화 (원본 탐지 + 추적 결과)
        self.visualization.visualize_detections(
            self.current_image,
            self.detected_objects,
            current_mission_type.name,
            self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints(),
            raw_detections=self.raw_detections,
            bridge=self.bridge,
            viz_image_pub=self.ros_comm.publishers.get('viz_image')
        )'''

    new_viz_call = '''        # 현재 미션 파라미터 가져오기 (시각화용)
        current_mission_params = None
        if current_mission_type == MissionType.PASS_BETWEEN_BUOYS:
            trackbar_params = self.visualization.update_parameters_from_trackbars()
            current_mission_params = {
                'pass_max_depth_diff': trackbar_params.get('pass_max_depth_diff', 5.0)
            }

        # 탐지 결과 시각화 (원본 탐지 + 추적 결과)
        self.visualization.visualize_detections(
            self.current_image,
            self.detected_objects,
            current_mission_type.name,
            self.waypoint_manager.get_waypoint_index(),
            self.waypoint_manager.get_total_waypoints(),
            raw_detections=self.raw_detections,
            bridge=self.bridge,
            viz_image_pub=self.ros_comm.publishers.get('viz_image'),
            mission_params=current_mission_params
        )'''

    content = content.replace(old_viz_call, new_viz_call)

    # 파일 쓰기
    if content != original:
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {main_file} 수정 완료!")
        return True
    else:
        print(f"⚠️  {main_file} 변경사항 없음")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("부표 미션 수정 스크립트")
    print("=" * 60)

    print("\n[1/3] mission_strategies_new.py 수정 중...")
    success1 = fix_mission_strategies()

    print("\n[2/3] visualization_system.py 수정 중...")
    success2 = fix_visualization_system()

    print("\n[3/3] Main_MCP.py 수정 중...")
    success3 = fix_main_mcp()

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

    if success1 or success2 or success3:
        print("\n변경사항:")
        if success1:
            print("  ✅ mission_strategies_new.py: TRACKED 우선, pass_max_depth_diff 수정")
        if success2:
            print("  ✅ visualization_system.py: 필터링된 객체 X 표시")
        if success3:
            print("  ✅ Main_MCP.py: mission_params 전달 추가")
    else:
        print("\n⚠️  모든 파일이 이미 수정되어 있거나 패턴을 찾을 수 없습니다.")
