#!/bin/bash
# VRX 시스템 단축어 설정 스크립트

echo "🚀 VRX 시스템 단축어 설정 중..."

# 새로운 단축어들 추가
alias vrx_main='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 main.py'
alias vrx_run='cd ~/vrx_ws/src/vrx/scripts && python3 run_vrx_system.py'
alias vrx_plot='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 visual_main.py'
alias vrx_visual='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 visual_console.py'
alias vrx_robot='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 robot_visualizer.py'
alias vrx_simple_robot='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 simple_robot_viz.py'
alias vrx_trajectory='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 trajectory_viz.py'
alias vrx_simple='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 simple_plot_v2.py'
alias vrx_console='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 console_plot.py'
alias vrx_save='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 simple_plot_v3.py'
alias vrx_nav='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 vrx_plot_system.py'
alias vrx_test='cd ~/vrx_ws/src/vrx/scripts && python3 test_matplotlib_visualization.py'
alias vrx_test2='cd ~/vrx_ws/src/vrx/scripts && python3 test_simple_plot.py'
alias vrx_onnx='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 main_onnx.py'
alias vrx_onnx_v2='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 main_onnx_v2.py'
alias vrx_onnx_v3='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 main_onnx_v3.py'

# 기존 단축어들 (참고용)
# alias vrx2='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 blob_depth_detector_hybrid_multi_target_tracking.py'
# alias vrxcontrol='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 buoy_navigation_controller_hybrid.py'
# alias approach='cd ~/vrx_ws/src/vrx/scripts && source vrx_env/bin/activate && python3 object_approach_controller.py'

echo "✅ 단축어 설정 완료!"
echo ""
echo "사용 가능한 명령어:"
echo "  vrx_main  - 통합된 VRX 시스템 실행 (main.py)"
echo "  vrx_run   - VRX 시스템 실행 (래퍼 스크립트)"
echo "  vrx_plot  - Matplotlib 시각화 실행 (배 위치 및 장애물 정보)"
echo "  vrx_visual- VRX 콘솔 기반 시각화 (ASCII 아트 맵)"
echo "  vrx_robot - VRX 로봇 실시간 시각화 (LiDAR, GPS, IMU)"
echo "  vrx_simple_robot - 간단한 VRX 로봇 시각화 (더 안정적)"
echo "  vrx_trajectory - VRX 로봇 궤적 시각화 (헤딩 포함)"
echo "  vrx_simple- 간단한 VRX 데이터 시각화 (LiDAR, GPS, IMU)"
echo "  vrx_console- 콘솔 기반 VRX 데이터 모니터링 (GUI 없음)"
echo "  vrx_save   - VRX 데이터 플롯 파일 저장 (5초마다 PNG 저장)"
echo "  vrx_nav    - VRX 네비게이션 플롯 시스템 (클릭으로 waypoint 설정)"
echo "  vrx_test  - Matplotlib 시각화 테스트 (더미 데이터)"
echo "  vrx_test2 - 간단한 VRX 데이터 시각화 테스트 (더미 데이터)"
echo "  vrx_onnx  - VRX ONNX 강화학습 제어 시스템 (Ray-48130414.onnx)"
echo "  vrx_onnx_v2 - VRX ONNX 제어 시스템 v2 (TurtleBot 스타일)"
echo "  vrx_onnx_v3 - VRX ONNX 제어 시스템 v3 (클릭 웨이포인트)"
echo ""
echo "기존 명령어들:"
echo "  vrx2      - 다중 표적 추적 시스템 (이전 버전)"
echo "  vrxcontrol- 부표 네비게이션 컨트롤러 (이전 버전)"
echo "  approach  - 객체 접근 컨트롤러 (이전 버전)"
echo ""
echo "💡 .bashrc 또는 .zshrc에 추가하려면:"
echo "  echo 'source ~/vrx_ws/src/vrx/scripts/setup_aliases.sh' >> ~/.bashrc"
