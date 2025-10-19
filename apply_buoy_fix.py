#!/usr/bin/env python3
"""
부표 사이 통과 미션 수정 스크립트
- 추정값 우선 사용, 측정값은 fallback으로 개별 처리
- pass_max_depth_diff 파라미터 수정
"""

def apply_fix():
    file_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/mission_strategies_new.py'

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 수정할 부분 찾기
    old_code = '''        # 빨간색/초록색 부표 찾기 (원본 탐지값 우선 사용)
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
                logger.info("⚠️ 원본값 없음 -> 추적값 사용 (fallback)")

        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 깊이 차이 필터링 (너무 차이 나면 멀리 있는 것 무시)
            max_depth_diff = kwargs.get('mission_params', {}).get('max_depth_diff', 5.0)*0.05
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

            # 이미지 중심
            image_center_x = current_image.shape[1] / 2

            # 오차 계산
            error = midpoint_x - image_center_x

            # 비례 제어
            steering_gain = 0.003
            forward_speed = 0.5

            steering = error * steering_gain
            steering = np.clip(steering, -0.3, 0.3)

            # 스러스터 명령 계산
            left_thrust = (forward_speed + steering) * self.thrust_scale
            right_thrust = (forward_speed - steering) * self.thrust_scale

            if logger:
                logger.info(
                    f"Pass Buoys [{data_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )'''

    new_code = '''        # 빨간색/초록색 부표 찾기 (추정값 우선, 개별 처리)
        red_buoy = None
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
                        logger.info("⚠️ 초록색 추정값 없음 -> 측정값 사용")

        # 두 부표가 모두 탐지된 경우
        if red_buoy and green_buoy:
            # 깊이 차이 필터링 (너무 차이 나면 멀리 있는 것 무시)
            pass_max_depth_diff = kwargs.get('mission_params', {}).get('pass_max_depth_diff', 5.0)
            red_depth = red_buoy.get('depth', 0.0)
            green_depth = green_buoy.get('depth', 0.0)
            depth_diff = abs(red_depth - green_depth)

            if logger:
                logger.info(f"깊이 비교: 빨강={red_depth:.2f}m, 초록={green_depth:.2f}m, 차이={depth_diff:.2f}m, 임계값={pass_max_depth_diff:.2f}m")

            if depth_diff > pass_max_depth_diff:
                # 깊이 차이가 너무 크면 멀리 있는 부표 무시
                if red_depth > green_depth:
                    if logger:
                        logger.warn(f"빨간 부표 무시 (깊이 차이 {depth_diff:.2f}m > {pass_max_depth_diff:.2f}m)")
                    red_buoy = None
                    red_source = None
                else:
                    if logger:
                        logger.warn(f"초록 부표 무시 (깊이 차이 {depth_diff:.2f}m > {pass_max_depth_diff:.2f}m)")
                    green_buoy = None
                    green_source = None

        # 필터링 후 두 부표가 모두 있는 경우
        if red_buoy and green_buoy:
            # 두 부표의 중점 계산 (이미지 좌표)
            red_x = red_buoy['center'][0]
            green_x = green_buoy['center'][0]
            midpoint_x = (red_x + green_x) / 2

            # 이미지 중심
            image_center_x = current_image.shape[1] / 2

            # 오차 계산
            error = midpoint_x - image_center_x

            # 비례 제어
            steering_gain = 0.003
            forward_speed = 0.5

            steering = error * steering_gain
            steering = np.clip(steering, -0.3, 0.3)

            # 스러스터 명령 계산
            left_thrust = (forward_speed + steering) * self.thrust_scale
            right_thrust = (forward_speed - steering) * self.thrust_scale

            if logger:
                logger.info(
                    f"Pass Buoys [R:{red_source},G:{green_source}]: midpoint={midpoint_x:.1f}, error={error:.1f}, steering={steering:.3f}"
                )'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 수정 완료!")
        print("\n변경 사항:")
        print("1. 추정값(TRACKED) 우선 → 측정값(RAW) fallback으로 변경")
        print("2. 빨간색/초록색 부표를 개별로 처리")
        print("3. pass_max_depth_diff 파라미터 이름 수정 및 스케일 통일 (*0.05 제거)")
        print("4. 깊이 비교 로깅 추가")
        print("5. 각 부표의 데이터 소스 표시 (R:TRACKED, G:RAW 등)")
    else:
        print("❌ 수정할 코드를 찾을 수 없습니다.")
        print("파일이 이미 수정되었거나 코드가 변경되었을 수 있습니다.")
        return False

    return True

if __name__ == "__main__":
    apply_fix()
