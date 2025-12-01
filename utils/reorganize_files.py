#!/usr/bin/env python3
"""
파일 재구성 스크립트
utils 폴더의 파일들을 기능별 폴더로 이동
"""
import shutil
import os

# 현재 디렉토리
base_dir = os.path.dirname(os.path.abspath(__file__))

# 파일 이동 매핑
file_moves = {
    # mission/
    'mission_strategies_new.py': 'mission/',
    'mission_control.py': 'mission/',
    'mission_executor.py': 'mission/',
    'waypoint_manager.py': 'mission/',
    'parameter_manager.py': 'mission/',
    
    # sensors/
    'sensor_callbacks.py': 'sensors/',
    'sensor_preprocessing.py': 'sensors/',
    'depth_estimation.py': 'sensors/',
    'depth_estimation_optimized.py': 'sensors/',
    'depth_estimation_ultra.py': 'sensors/',
    'depth_filter.py': 'sensors/',
    
    # detection/
    'detection_system.py': 'detection/',
    'detection_system_optimized.py': 'detection/',
    'imm_pdaf_tracker.py': 'detection/',
    
    # control/
    'avoid_control.py': 'control/',
    'thruster_allocation.py': 'control/',
    'onnx_controller.py': 'control/',
    
    # visualization/
    'visualization_system.py': 'visualization/',
    'viz_components.py': 'visualization/',
    'image_preprocessor.py': 'visualization/',
    
    # communication/
    'ros_communication.py': 'communication/',
    'px4_adapter.py': 'communication/',
    
    # core/
    'config.py': 'core/',
    'helpers.py': 'core/',
    'system_factory.py': 'core/',
    'super_optimizer.py': 'core/',
    'jetson_optimizer.py': 'core/',
}

def move_files():
    """파일 이동 실행"""
    moved = []
    errors = []
    
    for filename, target_dir in file_moves.items():
        src = os.path.join(base_dir, filename)
        dst_dir = os.path.join(base_dir, target_dir)
        dst = os.path.join(dst_dir, filename)
        
        if os.path.exists(src):
            try:
                # 대상 디렉토리가 없으면 생성
                os.makedirs(dst_dir, exist_ok=True)
                # 파일 이동
                shutil.move(src, dst)
                moved.append(f"{filename} -> {target_dir}")
                print(f"✓ {filename} -> {target_dir}")
            except Exception as e:
                errors.append(f"{filename}: {e}")
                print(f"✗ {filename}: {e}")
        else:
            print(f"⚠ {filename} not found, skipping")
    
    print(f"\n이동 완료: {len(moved)}개 파일")
    if errors:
        print(f"오류: {len(errors)}개")
        for error in errors:
            print(f"  - {error}")

if __name__ == '__main__':
    move_files()

