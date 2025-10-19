#!/usr/bin/env python3
"""
부표 사이 통과 미션 깊이 차이 필터 추가 및 mission_mode 트랙바 제거
"""

def apply_fixes():
    print("=" * 80)
    print("코드 수정 시작...")
    print("=" * 80)

    # ==============================================
    # 1. visualization_system.py 수정
    # ==============================================
    viz_file = "utils/visualization_system.py"
    print(f"\n📝 {viz_file} 읽는 중...")

    with open(viz_file, 'r', encoding='utf-8') as f:
        viz_content = f.read()

    # 수정 1-1: mission_mode 트랙바 제거하고 pass_max_depth_diff 트랙바 추가
    old_trackbar = """        cv2.createTrackbar('Circle: TX MaxX', 'Parameters',
                          1200, 2000, self._dummy_callback)

        # 미션 모드 선택 트랙바
        # 0: 자동 (웨이포인트 기반), 1: 장애물 회피, 2: 부표 사이 통과, 3: 부표 회전
        cv2.createTrackbar('Mission Mode', 'Parameters',
                          0, 3, self._dummy_callback)"""

    new_trackbar = """        cv2.createTrackbar('Circle: TX MaxX', 'Parameters',
                          1200, 2000, self._dummy_callback)

        # 부표 사이 통과 미션 파라미터 트랙바
        cv2.createTrackbar('Pass: Max Depth Diff x10', 'Parameters',
                          50, 200, self._dummy_callback)  # 기본값 5.0m (50/10)"""

    viz_content = viz_content.replace(old_trackbar, new_trackbar)

    # 수정 1-2: mission_mode 파라미터 제거하고 pass_max_depth_diff 파라미터 추가
    old_params = """        circle_tx_min_x = float(cv2.getTrackbarPos('Circle: TX MinX', 'Parameters'))
        circle_tx_max_x = float(cv2.getTrackbarPos('Circle: TX MaxX', 'Parameters'))

        # 미션 모드 선택
        mission_mode = cv2.getTrackbarPos('Mission Mode', 'Parameters')

        return {"""

    new_params = """        circle_tx_min_x = float(cv2.getTrackbarPos('Circle: TX MinX', 'Parameters'))
        circle_tx_max_x = float(cv2.getTrackbarPos('Circle: TX MaxX', 'Parameters'))

        # 부표 사이 통과 미션 파라미터
        pass_max_depth_diff = cv2.getTrackbarPos('Pass: Max Depth Diff x10', 'Parameters') / 10.0

        return {"""

    viz_content = viz_content.replace(old_params, new_params)

    # 수정 1-3: return dict에서 mission_mode 제거하고 pass_max_depth_diff 추가
    old_return = """            'circle_tx_slope': circle_tx_slope,
            'circle_tx_min_x': circle_tx_min_x,
            'circle_tx_max_x': circle_tx_max_x,
            'mission_mode': mission_mode
        }"""

    new_return = """            'circle_tx_slope': circle_tx_slope,
            'circle_tx_min_x': circle_tx_min_x,
            'circle_tx_max_x': circle_tx_max_x,
            'pass_max_depth_diff': pass_max_depth_diff
        }"""

    viz_content = viz_content.replace(old_return, new_return)

    with open(viz_file, 'w', encoding='utf-8') as f:
        f.write(viz_content)
    print(f"✅ {viz_file} 수정 완료!")

    # ==============================================
    # 2. Main_MCP.py 수정
    # ==============================================
    main_file = "Main_MCP.py"
    print(f"\n📝 {main_file} 읽는 중...")

    with open(main_file, 'r', encoding='utf-8') as f:
        main_content = f.read()

    # 수정 2-1: mission_mode 관련 로직 제거 (자동 모드로만 동작)
    old_mission_mode = """        # 트랙바로부터 미션 모드 읽어오기
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
            current_mission_type = None"""

    new_mission_mode = """        # 자동 모드: 웨이포인트 기반 미션 전환
        self._check_waypoint_transition()
        current_mission_type = self.waypoint_manager.get_current_mission_type()"""

    main_content = main_content.replace(old_mission_mode, new_mission_mode)

    # 수정 2-2: PassBetweenBuoys 실행 시 pass_max_depth_diff 파라미터 전달
    old_pass_exec = """        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                logger=self.get_logger()
            )"""

    new_pass_exec = """        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
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
            )"""

    main_content = main_content.replace(old_pass_exec, new_pass_exec)

    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(main_content)
    print(f"✅ {main_file} 수정 완료!")

    # ==============================================
    # 3. mission_strategies_new.py 수정
    # ==============================================
    strategy_file = "utils/mission_strategies_new.py"
    print(f"\n📝 {strategy_file} 읽는 중...")

    with open(strategy_file, 'r', encoding='utf-8') as f:
        strategy_content = f.read()

    # 수정 3-1: PassBetweenBuoysMission.execute()에 깊이 차이 필터 로직 추가
    old_buoy_logic = """        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 두 부표의 중점 계산 (이미지 좌표)
            red_x = red_buoy['center'][0]
            green_x = green_buoy['center'][0]
            midpoint_x = (red_x + green_x) / 2"""

    new_buoy_logic = """        # 두 부표가 모두 탐지된 경우
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
            midpoint_x = (red_x + green_x) / 2"""

    strategy_content = strategy_content.replace(old_buoy_logic, new_buoy_logic)

    with open(strategy_file, 'w', encoding='utf-8') as f:
        f.write(strategy_content)
    print(f"✅ {strategy_file} 수정 완료!")

    print("\n" + "=" * 80)
    print("🎉 모든 수정 완료!")
    print("=" * 80)
    print("\n수정 내용:")
    print("1. ✅ mission_mode 트랙바 제거 (자동 모드로만 동작)")
    print("2. ✅ Pass: Max Depth Diff x10 트랙바 추가 (기본값 5.0m)")
    print("3. ✅ 부표 사이 통과 미션에 깊이 차이 필터 로직 추가")
    print("   - 빨강/초록 부표의 깊이 차이가 기준치 초과 시 멀리 있는 것 무시")
    print("\n테스트:")
    print("- Parameters 창에서 'Pass: Max Depth Diff x10' 트랙바 조정")
    print("- 부표 사이 통과 미션 실행 시 로그에서 필터링 메시지 확인")

if __name__ == '__main__':
    try:
        apply_fixes()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
