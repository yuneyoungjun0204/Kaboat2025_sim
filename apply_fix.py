#!/usr/bin/env python3
"""
Main_MCP.py와 mission_strategies_new.py 수정 스크립트
1. 모드 수동 변경 시 장애물 단계 진입 로그 추가
2. 탐지 시 원본값(raw detections) 우선 사용
"""

import os
import re

def apply_fixes():
    # 경로 설정
    main_file = "Main_MCP.py"
    strategy_file = "utils/mission_strategies_new.py"

    # ==============================================
    # 1. Main_MCP.py 수정
    # ==============================================
    with open(main_file, 'r', encoding='utf-8') as f:
        main_content = f.read()

    # 수정 1-1: 수동 모드 추적 변수 초기화
    main_content = main_content.replace(
        """        # 수동 목표 위치 (trajectory_viz에서 클릭한 좌표)
        self.manual_target_x = None
        self.manual_target_y = None""",
        """        # 수동 목표 위치 (trajectory_viz에서 클릭한 좌표)
        self.manual_target_x = None
        self.manual_target_y = None

        # 수동 모드 추적용
        self._last_manual_mode = 0"""
    )

    # 수정 1-2: 미션 모드 전환 로그 추가
    old_mode_switch = """        elif mission_mode == 1:
            # 수동 장애물 회피 모드
            current_mission_type = MissionType.OBSTACLE_AVOID
        elif mission_mode == 2:
            # 수동 부표 사이 통과 모드
            current_mission_type = MissionType.PASS_BETWEEN_BUOYS
        elif mission_mode == 3:
            # 수동 부표 회전 모드
            current_mission_type = MissionType.CIRCLE_BUOY"""

    new_mode_switch = """        elif mission_mode == 1:
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
            self._last_manual_mode = 3"""

    main_content = main_content.replace(old_mode_switch, new_mode_switch)

    # 수정 1-3: PassBetweenBuoys 미션에 raw_detections 전달
    old_pass_between = """        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                logger=self.get_logger()
            )"""

    new_pass_between = """        if mission_type == MissionType.PASS_BETWEEN_BUOYS:
            return self.mission_manager.execute_mission(
                mission_type,
                detected_objects=self.detected_objects,
                current_image=self.current_image,
                raw_detections=self.raw_detections,  # 원본 탐지값 추가
                logger=self.get_logger()
            )"""

    main_content = main_content.replace(old_pass_between, new_pass_between)

    # Main_MCP.py 저장
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(main_content)
    print(f"✅ {main_file} 수정 완료!")

    # ==============================================
    # 2. mission_strategies_new.py 수정
    # ==============================================
    with open(strategy_file, 'r', encoding='utf-8') as f:
        strategy_content = f.read()

    # 수정 2-1: PassBetweenBuoysMission.execute() 시그니처 및 로직 변경
    old_pass_signature = """    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, **kwargs) -> Tuple[float, float]:
        \"\"\"빨간색/초록색 고깔 부표 사이로 지나가기\"\"\"
        # 빨간색/초록색 부표 찾기
        red_buoy = None
        green_buoy = None

        for det in detected_objects:
            if det['label'] == 'red_cone':
                red_buoy = det
            elif det['label'] == 'green_cone':
                green_buoy = det"""

    new_pass_signature = """    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                logger=None, raw_detections: List[Dict] = None, **kwargs) -> Tuple[float, float]:
        \"\"\"
        빨간색/초록색 고깔 부표 사이로 지나가기

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)
        \"\"\"
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
                logger.info("⚠️ 원본값 없음 -> 추적값 사용 (fallback)")"""

    strategy_content = strategy_content.replace(old_pass_signature, new_pass_signature)

    # 수정 2-2: 로그에 데이터 소스 표시
    old_log = """            if logger:
                logger.info(
                    f"Pass Buoys: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )"""

    new_log = """            if logger:
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )"""

    strategy_content = strategy_content.replace(old_log, new_log)

    # mission_strategies_new.py 저장
    with open(strategy_file, 'w', encoding='utf-8') as f:
        f.write(strategy_content)
    print(f"✅ {strategy_file} 수정 완료!")

    print("\n🎉 모든 수정 완료!\n")
    print("수정 내용:")
    print("1. 모드 수동 변경 시 미션 진입 로그 추가 (장애물 회피 포함)")
    print("2. 부표 사이 통과 미션에서 원본 탐지값(raw detections) 우선 사용")
    print("3. 데이터 소스 표시 (RAW vs TRACKED)")

if __name__ == '__main__':
    apply_fixes()
