#!/usr/bin/env python3
"""
Fix depth scale inconsistency in visualization_system.py
"""

import re

file_path = '/home/yuneyoungjun/vrx_ws/src/vrx/Scripts_git/utils/visualization_system.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Change trackbar name from "x10" to "(m)" and default value from 10 to 5
content = re.sub(
    r"cv2\.createTrackbar\('Pass: Max Depth Diff x10', 'Parameters',\s*\n\s*10, 200, self\._dummy_callback\)  # 기본값 5\.0m \(50/10\)",
    "cv2.createTrackbar('Pass: Max Depth Diff (m)', 'Parameters',\n                          5, 200, self._dummy_callback)  # 기본값 5.0m",
    content
)

# Fix 2: Remove /20.0 scaling from min_depth_threshold
content = re.sub(
    r"self\.min_depth_threshold = float\(cv2\.getTrackbarPos\('Min Depth \(m\)', 'Parameters'\)\) / 20\.0",
    "self.min_depth_threshold = float(cv2.getTrackbarPos('Min Depth (m)', 'Parameters'))",
    content
)

# Fix 3: Remove /10.0 scaling from pass_max_depth_diff and update trackbar name
content = re.sub(
    r"pass_max_depth_diff = cv2\.getTrackbarPos\('Pass: Max Depth Diff x10', 'Parameters'\) / 10\.0",
    "pass_max_depth_diff = float(cv2.getTrackbarPos('Pass: Max Depth Diff (m)', 'Parameters'))",
    content
)

# Write the file back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed depth scale inconsistency in visualization_system.py")
print("  - Changed 'Pass: Max Depth Diff x10' to 'Pass: Max Depth Diff (m)'")
print("  - Removed /20.0 scaling from min_depth_threshold")
print("  - Removed /10.0 scaling from pass_max_depth_diff")
print("  - All depth parameters now use direct meter values (same scale)")
