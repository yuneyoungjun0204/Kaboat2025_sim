#!/usr/bin/env python3
"""
시각화 테스트 스크립트 (시뮬레이터 없이 테스트)
"""

import cv2
import numpy as np
import sys
sys.path.append('/home/yuneyoungjun/vrx_ws/src/vrx/kaboat_backup')

from utils.visualization_system import VisualizationSystem
from utils.detection_system import MissionType

def create_test_image():
    """테스트 이미지 생성"""
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 가짜 부표 그리기
    cv2.circle(image, (200, 240), 30, (0, 0, 255), -1)  # 빨간 원
    cv2.circle(image, (440, 240), 30, (0, 255, 0), -1)  # 초록 원

    return image

def create_test_depth_map():
    """테스트 깊이 맵 생성"""
    # 중심에서 멀어질수록 깊이가 증가하는 패턴
    y, x = np.ogrid[:480, :640]
    cy, cx = 240, 320
    depth_map = np.sqrt((x - cx)**2 + (y - cy)**2) / 400.0
    depth_map = np.clip(depth_map, 0, 1)
    return depth_map

def create_test_detections():
    """테스트 탐지 결과 생성"""
    return [
        {
            'label': 'red_cone',
            'confidence': 0.95,
            'bbox': [170, 210, 230, 270],
            'center': (200, 240),
            'depth': 5.2,
            'track_id': 1
        },
        {
            'label': 'green_cone',
            'confidence': 0.92,
            'bbox': [410, 210, 470, 270],
            'center': (440, 240),
            'depth': 5.8,
            'track_id': 2
        }
    ]

def main():
    """테스트 메인"""
    print("=" * 60)
    print("시각화 테스트 시작")
    print("=" * 60)

    # Visualization System 초기화
    viz = VisualizationSystem()

    print("\n창이 표시되었습니다!")
    print("- 'VRX Mission Display' 창: 탐지 결과")
    print("- 'Depth Map' 창: 깊이 맵")
    print("\n종료하려면 아무 키나 누르세요...")

    # 테스트 데이터 생성
    test_image = create_test_image()
    test_depth_map = create_test_depth_map()
    test_detections = create_test_detections()

    frame_count = 0

    # 루프
    while True:
        frame_count += 1

        # 시각화 업데이트
        viz.visualize_detections(
            image=test_image,
            detections=test_detections,
            mission_name="PASS_BETWEEN_BUOYS",
            waypoint_index=1,
            total_waypoints=5,
            depth_map=test_depth_map
        )

        # 프레임 카운트 표시
        if frame_count % 30 == 0:
            print(f"프레임 {frame_count} 처리 중...")

        # 키 입력 확인
        key = cv2.waitKey(30)
        if key != -1:
            print(f"\n키 입력 감지: {key}")
            break

    # 정리
    viz.cleanup()
    print("\n✅ 테스트 완료!")

if __name__ == '__main__':
    main()
