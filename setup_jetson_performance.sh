#!/bin/bash
###############################################################################
# Jetson Orin Nano 성능 최적화 스크립트
# - 최대 전력 모드 설정
# - GPU/CPU 클럭 최대화
# - 메모리 대역폭 최적화
###############################################################################

echo "================================================"
echo "🚀 Jetson Orin Nano 성능 최적화 시작"
echo "================================================"

# Root 권한 확인
if [ "$EUID" -ne 0 ]; then
    echo "❌ 이 스크립트는 root 권한이 필요합니다."
    echo "다음과 같이 실행하세요: sudo bash $0"
    exit 1
fi

echo ""
echo "📊 현재 전력 모드 확인..."
nvpmodel -q 2>/dev/null || echo "nvpmodel이 설치되어 있지 않습니다."

echo ""
echo "⚡ 최대 성능 모드 설정 (MAXN)..."
# Jetson Orin Nano: Mode 0 = MAXN (최대 성능)
nvpmodel -m 0 2>/dev/null && echo "✅ MAXN 모드 활성화" || echo "⚠️ nvpmodel 명령 실패"

echo ""
echo "🔥 Jetson Clocks 활성화 (최대 클럭)..."
jetson_clocks 2>/dev/null && echo "✅ Jetson clocks 활성화" || echo "⚠️ jetson_clocks 명령 실패"

echo ""
echo "🎯 GPU 최대 성능 모드 설정..."
# GPU 전원 관리 비활성화 (최대 성능)
if [ -f /sys/devices/17000000.ga10b/power/control ]; then
    echo "on" > /sys/devices/17000000.ga10b/power/control 2>/dev/null
    echo "✅ GPU 전원 관리 on"
fi

echo ""
echo "💾 메모리 대역폭 최적화..."
# 캐시 최적화
echo 1 > /proc/sys/vm/drop_caches 2>/dev/null && echo "✅ 캐시 정리 완료" || true

echo ""
echo "🔧 CUDA 최적화 설정..."
# CUDA_LAUNCH_BLOCKING을 0으로 설정 (비동기 실행)
export CUDA_LAUNCH_BLOCKING=0
# CUDA 캐시 경로 설정
export CUDA_CACHE_PATH=/tmp/cuda_cache
mkdir -p $CUDA_CACHE_PATH 2>/dev/null
echo "✅ CUDA 최적화 변수 설정"

echo ""
echo "📊 최종 상태 확인..."
echo "=== 전력 모드 ==="
nvpmodel -q 2>/dev/null || echo "nvpmodel 정보 없음"

echo ""
echo "=== GPU 상태 ==="
tegrastats --interval 1000 --logfile /dev/null 2>&1 &
TEGRA_PID=$!
sleep 2
kill $TEGRA_PID 2>/dev/null || true

echo ""
echo "================================================"
echo "✅ Jetson 성능 최적화 완료!"
echo "================================================"
echo ""
echo "💡 Python 환경 변수 설정을 위해 다음 명령을 추가하세요:"
echo "   export CUDA_LAUNCH_BLOCKING=0"
echo "   export CUDA_CACHE_PATH=/tmp/cuda_cache"
echo ""
echo "또는 ~/.bashrc에 추가:"
echo "   echo 'export CUDA_LAUNCH_BLOCKING=0' >> ~/.bashrc"
echo "   echo 'export CUDA_CACHE_PATH=/tmp/cuda_cache' >> ~/.bashrc"
echo ""
