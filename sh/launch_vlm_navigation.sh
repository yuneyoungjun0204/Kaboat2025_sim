#!/bin/bash
# VLM 기반 선박 자동 주차 내비게이션 실행 스크립트

echo "========================================"
echo "VLM Navigation Node 실행 옵션"
echo "========================================"
echo "1. Qwen2.5-VL-7B (최고 정확도, GPU 4GB+)"
echo "2. Phi-3-Vision (균형잡힌 성능, GPU 2-3GB)"
echo "========================================"
read -p "선택 (1 또는 2): " choice

case $choice in
    1)
        echo "Qwen2.5-VL-7B 노드 실행 중..."
        python3 vlm_qwen_node.py
        ;;
    2)
        echo "Phi-3-Vision 노드 실행 중..."
        python3 vlm_phi3_node.py
        ;;
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac
