#!/usr/bin/env python3
"""
KABOAT 도킹 임무 전용 하이브리드 시스템
전통적 방법(OpenCV) + VLM 백업(Qwen2.5-VL)

임무 3: 도킹
- 3개 도킹 스테이션 (좌/중앙/우)
- 형상: 삼각형, 원형, 네모
- 색상: 빨강, 초록, 파랑, 주황, 노랑
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ShapeColorDetector:
    """전통적인 OpenCV 기반 형상+색상 검출"""

    def __init__(self):
        # 색상 HSV 범위 정의
        self.color_ranges = {
            "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                   (np.array([160, 100, 100]), np.array([180, 255, 255]))],
            "green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            "blue": [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
            "orange": [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
            "yellow": [(np.array([25, 100, 100]), np.array([35, 255, 255]))]
        }

        self.min_area = 500  # 최소 도형 크기

    def detect_shape(self, contour):
        """컨투어로부터 형상 분류 (삼각형/원형/네모)"""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 3:
            return "triangle"
        elif len(approx) == 4:
            return "square"
        else:
            # 원형도 검사
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity > 0.7:
                return "circle"
        return "unknown"

    def detect_color(self, image, color_name):
        """특정 색상 검출"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 색상 마스크 생성
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges.get(color_name, []):
            mask |= cv2.inRange(hsv, lower, upper)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_docks(self, image, target_shape, target_color):
        """도킹 스테이션 찾기"""
        # 색상 검출
        mask = self.detect_color(image, target_color)

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # 형상 분류
            shape = self.detect_shape(cnt)
            if shape == "unknown":
                continue

            # 중심점 계산
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 신뢰도 계산 (면적 기반)
            confidence = min(area / 5000.0, 1.0)

            results.append({
                "position": (cx, cy),
                "shape": shape,
                "color": target_color,
                "area": area,
                "confidence": confidence,
                "contour": cnt
            })

        return results

    def classify_dock_position(self, cx, image_width):
        """도크 위치 분류 (left/center/right)"""
        if cx < image_width * 0.33:
            return "left"
        elif cx < image_width * 0.67:
            return "center"
        else:
            return "right"


class Qwen2VLAnalyzer:
    """Qwen2.5-VL 기반 백업 분석기"""

    def __init__(self, use_quantization=True):
        print("Qwen2.5-VL 로딩 중...")

        model_name = "Qwen/Qwen2-VL-7B-Instruct"

        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Qwen2.5-VL 준비 완료!")

    def analyze_image(self, image, prompt):
        """이미지 분석"""
        # PIL로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 처리
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )

        output_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response.strip()


class KABOATDockingHybrid:
    """하이브리드 도킹 시스템"""

    def __init__(self, use_vlm_backup=True):
        print("=" * 70)
        print("KABOAT 도킹 하이브리드 시스템 초기화")
        print("=" * 70)

        # 전통적 검출기 (항상 사용)
        self.opencv_detector = ShapeColorDetector()
        print("✓ OpenCV 검출기 준비")

        # VLM 백업 (선택적)
        self.use_vlm = use_vlm_backup
        self.vlm = None
        if use_vlm_backup:
            try:
                self.vlm = Qwen2VLAnalyzer(use_quantization=True)
                print("✓ VLM 백업 준비")
            except Exception as e:
                print(f"⚠ VLM 로딩 실패 (OpenCV만 사용): {e}")
                self.use_vlm = False

        print("=" * 70)

    def find_target_dock(self, image, target_shape, target_color):
        """목표 도킹 스테이션 찾기 (하이브리드)"""
        h, w = image.shape[:2]

        print(f"\n[검색] {target_color} {target_shape} 찾는 중...")

        # 1단계: OpenCV 시도
        opencv_results = self.opencv_detector.find_docks(image, target_shape, target_color)

        # 목표 형상과 일치하는 것만 필터링
        matched = [r for r in opencv_results if r["shape"] == target_shape]

        if matched:
            # 가장 큰 것 선택
            best = max(matched, key=lambda x: x["area"])

            if best["confidence"] > 0.85:
                # 신뢰도 높음 → OpenCV 결과 사용
                dock_pos = self.opencv_detector.classify_dock_position(best["position"][0], w)
                print(f"✓ [OpenCV] {dock_pos} 도크 발견 (신뢰도: {best['confidence']:.2f})")
                return {
                    "method": "opencv",
                    "position": dock_pos,
                    "pixel_location": best["position"],
                    "confidence": best["confidence"],
                    "contour": best["contour"]
                }
            else:
                print(f"⚠ [OpenCV] 신뢰도 낮음 ({best['confidence']:.2f})")
        else:
            print("⚠ [OpenCV] 매칭 없음")

        # 2단계: VLM 백업 사용
        if self.use_vlm and self.vlm:
            print("→ [VLM] 백업 분석 시작...")

            prompt = f"""Look at this image of 3 docking stations.
Find the dock that has a {target_color} {target_shape}.
Answer with ONLY one word: left, center, or right."""

            try:
                vlm_answer = self.vlm.analyze_image(image, prompt)

                # 파싱
                vlm_answer_lower = vlm_answer.lower()
                if "left" in vlm_answer_lower:
                    dock_pos = "left"
                elif "center" in vlm_answer_lower or "middle" in vlm_answer_lower:
                    dock_pos = "center"
                elif "right" in vlm_answer_lower:
                    dock_pos = "right"
                else:
                    print(f"✗ [VLM] 파싱 실패: '{vlm_answer}'")
                    return None

                print(f"✓ [VLM] {dock_pos} 도크 (응답: '{vlm_answer}')")
                return {
                    "method": "vlm",
                    "position": dock_pos,
                    "vlm_response": vlm_answer,
                    "confidence": 0.7
                }

            except Exception as e:
                print(f"✗ [VLM] 오류: {e}")
                return None

        print("✗ 도크를 찾을 수 없음")
        return None

    def visualize_result(self, image, result):
        """결과 시각화"""
        debug_img = image.copy()
        h, w = image.shape[:2]

        if result is None:
            cv2.putText(debug_img, "NO DOCK FOUND", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return debug_img

        # 방법 표시
        method = result.get("method", "unknown")
        position = result.get("position", "unknown")
        confidence = result.get("confidence", 0.0)

        cv2.putText(debug_img, f"Method: {method.upper()}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Position: {position.upper()}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Confidence: {confidence:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # OpenCV 결과면 컨투어 표시
        if method == "opencv" and "contour" in result:
            cv2.drawContours(debug_img, [result["contour"]], -1, (0, 255, 0), 3)
            px, py = result["pixel_location"]
            cv2.circle(debug_img, (px, py), 10, (0, 255, 0), -1)

        # 구역 구분선
        cv2.line(debug_img, (w//3, 0), (w//3, h), (255, 255, 0), 2)
        cv2.line(debug_img, (2*w//3, 0), (2*w//3, h), (255, 255, 0), 2)
        cv2.putText(debug_img, "LEFT", (w//6, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(debug_img, "CENTER", (w//2-40, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(debug_img, "RIGHT", (5*w//6-30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return debug_img


def test_docking_system():
    """테스트 실행"""
    import os

    # 하이브리드 시스템 생성
    # use_vlm_backup=False로 설정하면 OpenCV만 사용
    system = KABOATDockingHybrid(use_vlm_backup=True)

    # 테스트 시나리오
    test_cases = [
        {
            "image": "test_img/Screenshot from 2025-10-03 10-15-26.png",
            "target_shape": "triangle",
            "target_color": "red"
        },
        {
            "image": "test_img/Screenshot from 2025-10-03 10-19-43.png",
            "target_shape": "circle",
            "target_color": "blue"
        }
    ]

    print("\n" + "=" * 70)
    print("도킹 시스템 테스트")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        img_path = test["image"]

        if not os.path.exists(img_path):
            print(f"\n[{i}] 이미지 없음: {img_path}")
            continue

        print(f"\n{'='*70}")
        print(f"[{i}] {os.path.basename(img_path)}")
        print(f"목표: {test['target_color']} {test['target_shape']}")
        print('='*70)

        # 이미지 로드
        image = cv2.imread(img_path)

        # 도크 찾기
        result = system.find_target_dock(
            image,
            test["target_shape"],
            test["target_color"]
        )

        # 시각화
        debug_img = system.visualize_result(image, result)

        # 저장
        output_path = f"output_docking_{i}.png"
        cv2.imwrite(output_path, debug_img)
        print(f"\n→ 결과 저장: {output_path}")


if __name__ == "__main__":
    test_docking_system()

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)
    print("\n사용법:")
    print("  # VLM 백업 포함")
    print("  system = KABOATDockingHybrid(use_vlm_backup=True)")
    print("")
    print("  # OpenCV만 (빠름)")
    print("  system = KABOATDockingHybrid(use_vlm_backup=False)")
    print("")
    print("  result = system.find_target_dock(image, 'triangle', 'red')")
