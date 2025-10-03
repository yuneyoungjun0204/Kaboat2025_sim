#!/bin/bash
# VRX 가상환경 초기 설정 스크립트 (최초 1회 실행)

set -e  # 오류 시 중단

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}🔧 VRX 가상환경 초기 설정${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Python 버전 확인
echo -e "${BLUE}1. Python 버전 확인...${NC}"
PYTHON_VERSION=$(python3 --version)
echo -e "   ${GREEN}✓ $PYTHON_VERSION${NC}"
echo ""

# 가상환경 생성
echo -e "${BLUE}2. 가상환경 생성...${NC}"
if [ -d "venv" ]; then
    echo -e "   ${YELLOW}⚠️  기존 가상환경이 있습니다. 삭제하고 재생성하시겠습니까? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo -e "   ${GREEN}✓ 기존 가상환경 삭제됨${NC}"
    else
        echo -e "   ${YELLOW}→ 기존 가상환경 유지${NC}"
        exit 0
    fi
fi

python3 -m venv venv
echo -e "   ${GREEN}✓ 가상환경 생성 완료${NC}"
echo ""

# 가상환경 활성화
echo -e "${BLUE}3. 가상환경 활성화...${NC}"
source venv/bin/activate
echo -e "   ${GREEN}✓ 활성화 완료${NC}"
echo ""

# pip 업그레이드
echo -e "${BLUE}4. pip 업그레이드...${NC}"
pip install --upgrade pip -q
PIP_VERSION=$(pip --version | awk '{print $2}')
echo -e "   ${GREEN}✓ pip $PIP_VERSION${NC}"
echo ""

# 패키지 설치
echo -e "${BLUE}5. 필수 패키지 설치...${NC}"
echo -e "   ${YELLOW}→ requirements.txt 기반 설치 중...${NC}"
pip install -r requirements.txt
echo -e "   ${GREEN}✓ 패키지 설치 완료${NC}"
echo ""

# 설치 확인
echo -e "${BLUE}6. 설치된 패키지 확인...${NC}"
echo -e "   ${GREEN}주요 패키지:${NC}"
pip list | grep -E "numpy|opencv|onnx|matplotlib|PyYAML|scipy" | sed 's/^/     /'
echo ""

# Config 테스트
echo -e "${BLUE}7. Config 시스템 테스트...${NC}"
if python3 utils/config_manager.py > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ ConfigManager 정상 동작${NC}"
else
    echo -e "   ${RED}✗ ConfigManager 오류 발생${NC}"
    exit 1
fi
echo ""

# 완료 메시지
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ 가상환경 설정 완료!${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "${YELLOW}다음 명령어로 가상환경을 활성화하세요:${NC}"
echo -e "  ${GREEN}source activate_venv.sh${NC}"
echo ""
echo -e "${YELLOW}또는:${NC}"
echo -e "  ${GREEN}source venv/bin/activate${NC}"
echo ""
