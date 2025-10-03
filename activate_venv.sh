#!/bin/bash
# VRX 가상환경 활성화 스크립트

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 프로젝트 디렉토리 찾기
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}🚤 VRX Kaboat2025 시뮬레이션 환경${NC}"
echo -e "${BLUE}=================================================${NC}"

# 가상환경 존재 확인
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  가상환경이 없습니다. 생성 중...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ 가상환경 생성 완료!${NC}"
else
    source venv/bin/activate
    echo -e "${GREEN}✅ 가상환경 활성화 완료${NC}"
fi

# 설치된 패키지 정보
echo ""
echo -e "${BLUE}📦 주요 패키지:${NC}"
pip list | grep -E "numpy|opencv|onnx|matplotlib|PyYAML|scipy" | sed 's/^/  /'

# ROS2 환경 확인
echo ""
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}⚠️  ROS2 환경이 설정되지 않았습니다${NC}"
    echo -e "${YELLOW}   다음 명령어로 ROS2를 활성화하세요:${NC}"
    echo -e "   ${GREEN}source /opt/ros/humble/setup.bash${NC}"
    echo -e "   ${GREEN}source ~/vrx_ws/install/setup.bash${NC}"
else
    echo -e "${GREEN}✅ ROS2 환경: $ROS_DISTRO${NC}"
fi

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}📝 사용 방법:${NC}"
echo -e "  ${BLUE}1.${NC} Config 테스트: ${GREEN}python utils/config_manager.py${NC}"
echo -e "  ${BLUE}2.${NC} 메인 실행:    ${GREEN}python main_mission.py${NC}"
echo -e "  ${BLUE}3.${NC} 시각화:       ${GREEN}python trajectory_viz.py${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# 프롬프트 변경 (가상환경 활성화 표시)
export PS1="(vrx-venv) \[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "
