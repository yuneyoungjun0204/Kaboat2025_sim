#!/bin/bash
# Jetson 최적화 명령어에 sudo 권한 설정
# 이 스크립트를 실행하면 nvpmodel과 jetson_clocks를 비밀번호 없이 실행할 수 있습니다

echo "======================================================================="
echo "🔧 Jetson 최적화 sudo 권한 설정"
echo "======================================================================="
echo ""
echo "이 스크립트는 다음 명령어를 비밀번호 없이 실행할 수 있게 합니다:"
echo "  - sudo nvpmodel -m 0"
echo "  - sudo jetson_clocks"
echo ""
echo "이렇게 하면 프로그램 실행 시 자동으로 Jetson 성능이 최적화됩니다."
echo ""

# 현재 사용자 확인
CURRENT_USER=$(whoami)
echo "현재 사용자: $CURRENT_USER"
echo ""

# sudoers.d 파일 생성
SUDOERS_FILE="/etc/sudoers.d/jetson-optimizer"

echo "sudo 권한 설정 중..."
echo ""

# sudoers 파일 내용
SUDOERS_CONTENT="# Jetson 최적화 명령어 (비밀번호 없이 실행)
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/jetson_clocks
"

# sudoers 파일 생성 (sudo 필요)
echo "$SUDOERS_CONTENT" | sudo tee $SUDOERS_FILE > /dev/null

# 파일 권한 설정
sudo chmod 0440 $SUDOERS_FILE

# 설정 확인
if sudo visudo -c -f $SUDOERS_FILE &>/dev/null; then
    echo "✅ sudo 권한 설정 완료!"
    echo ""
    echo "이제 다음 명령어를 비밀번호 없이 실행할 수 있습니다:"
    echo "  - sudo nvpmodel -m 0"
    echo "  - sudo jetson_clocks"
    echo ""
    echo "Main_MCP.py를 실행하면 자동으로 Jetson이 최적화됩니다! 🚀"
else
    echo "❌ 설정 실패! sudoers 파일 문법 오류"
    sudo rm -f $SUDOERS_FILE
    exit 1
fi

echo ""
echo "======================================================================="
echo "🎯 다음 단계"
echo "======================================================================="
echo ""
echo "1. Jetson 성능 최대화 (바로 실행):"
echo "   sudo nvpmodel -m 0"
echo "   sudo jetson_clocks"
echo ""
echo "2. TensorRT 엔진 생성 (최초 1회, 5-10분 소요):"
echo "   python3 convert_to_tensorrt.py --precision fp16"
echo ""
echo "   또는 INT8 (더 빠름, 10-15분 소요):"
echo "   python3 convert_to_tensorrt.py --precision int8"
echo ""
echo "3. 프로그램 실행:"
echo "   python3 Main_MCP.py"
echo ""
echo "======================================================================="
