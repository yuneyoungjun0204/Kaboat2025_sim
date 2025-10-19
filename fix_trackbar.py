#!/usr/bin/env python3
import re

# visualization_system.py 수정
with open('utils/visualization_system.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Mission Mode 트랙바 제거하고 Pass: Max Depth Diff 트랙바 추가
content = re.sub(
    r"        # 미션 모드 선택 트랙바\n        # 0: 자동 \(웨이포인트 기반\), 1: 장애물 회피, 2: 부표 사이 통과, 3: 부표 회전\n        cv2\.createTrackbar\('Mission Mode', 'Parameters',\n                          0, 3, self\._dummy_callback\)",
    "        # 부표 사이 통과 미션 파라미터 트랙바\n        cv2.createTrackbar('Pass: Max Depth Diff x10', 'Parameters',\n                          50, 200, self._dummy_callback)  # 기본값 5.0m (50/10)",
    content
)

# 2. mission_mode 파라미터를 pass_max_depth_diff로 변경
content = re.sub(
    r"        # 미션 모드 선택\n        mission_mode = cv2\.getTrackbarPos\('Mission Mode', 'Parameters'\)",
    "        # 부표 사이 통과 미션 파라미터\n        pass_max_depth_diff = cv2.getTrackbarPos('Pass: Max Depth Diff x10', 'Parameters') / 10.0",
    content
)

# 3. return dict에서 변경
content = content.replace(
    "'mission_mode': mission_mode",
    "'pass_max_depth_diff': pass_max_depth_diff"
)

with open('utils/visualization_system.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ visualization_system.py 수정 완료!")
