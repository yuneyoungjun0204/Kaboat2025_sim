"""
OWLv2 모델을 사용한 도킹 임무 테스트 스크립트
Zero-shot object detection으로 표식 탐지 및 도킹 유도

OWLv2 특징:
- 파인튜닝 없이 텍스트로 객체 지정 가능
- "red triangle", "blue circle" 등으로 즉시 탐지
- bbox 좌표로 정밀한 위치 제어
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import transformers
Owlv2Processor = transformers.Owlv2Processor
Owlv2ForObjectDetection = transformers.Owlv2ForObjectDetection
import json
from datetime import datetime


class OWLv2DockingDetector:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", confidence_threshold=0.1):
        """OWLv2 모델 초기화

        Args:
            model_id: 모델 ID (base 또는 large)
            confidence_threshold: 탐지 신뢰도 임계값
        """
        print(f"모델 로딩 중: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold

        print("모델 로딩 완료!")

        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.2f}GB (예약: {reserved:.2f}GB)")

    def detect_marker(self, image_path, target_queries):
        """특정 표식 탐지

        Args:
            image_path: 이미지 경로
            target_queries: 찾을 객체 리스트 (예: ["red triangle", "blue circle"])

        Returns:
            detections: 탐지 결과 리스트
        """
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 입력 준비
        inputs = self.processor(
            text=target_queries,
            images=image,
            return_tensors="pt"
        )

        # numpy 배열을 PyTorch 텐서로 변환
        inputs = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in inputs.items()
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 탐지 수행
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 결과 후처리
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # [height, width]
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # 탐지 결과 정리
        detections = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            box = box.cpu().tolist()
            score = score.cpu().item()
            label = target_queries[label_idx.cpu().item()]

            # 바운딩 박스 중심점 계산
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            detections.append({
                "label": label,
                "confidence": score,
                "bbox": box,  # [x1, y1, x2, y2]
                "center": [center_x, center_y],
                "width": box[2] - box[0],
                "height": box[3] - box[1]
            })

        return detections, image

    def calculate_navigation_command(self, detection, image_size, tolerance=50):
        """탐지된 표식 위치 기반 항법 명령 생성

        Args:
            detection: 탐지 결과
            image_size: 이미지 크기 (width, height)
            tolerance: 중심 허용 오차 (픽셀)

        Returns:
            command: 항법 명령 문자열
        """
        if not detection:
            return "NO TARGET FOUND - SEARCH"

        img_width, img_height = image_size
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        marker_x, marker_y = detection["center"]
        marker_width = detection["width"]

        # 거리 판단 (마커 크기로 추정)
        # 마커가 이미지 너비의 20% 이상이면 충분히 가까움
        distance_ratio = marker_width / img_width

        # 좌우 오차
        horizontal_error = marker_x - img_center_x
        # 상하 오차
        vertical_error = marker_y - img_center_y

        # 명령 우선순위: 좌우 정렬 > 거리 조절
        if abs(horizontal_error) > tolerance:
            if horizontal_error > 0:
                return f"MOVE LEFT (마커가 {int(horizontal_error)}px 오른쪽에 있음)"
            else:
                return f"MOVE RIGHT (마커가 {int(abs(horizontal_error))}px 왼쪽에 있음)"

        # 좌우 정렬 완료 후 거리 판단
        if distance_ratio < 0.15:  # 너무 멀면
            return f"MOVE FORWARD (거리: {distance_ratio*100:.1f}%)"
        elif distance_ratio > 0.35:  # 너무 가까우면
            return f"MOVE BACK (거리: {distance_ratio*100:.1f}%)"
        else:
            # 상하 정렬 확인
            if abs(vertical_error) > tolerance:
                if vertical_error > 0:
                    return f"ADJUST UP (마커가 {int(vertical_error)}px 아래에 있음)"
                else:
                    return f"ADJUST DOWN (마커가 {int(abs(vertical_error))}px 위에 있음)"

            return f"READY TO DOCK - 정렬 완료! (거리: {distance_ratio*100:.1f}%)"

    def visualize_detection(self, image, detections, command, save_path=None):
        """탐지 결과 시각화

        Args:
            image: PIL Image
            detections: 탐지 결과 리스트
            command: 항법 명령
            save_path: 저장 경로 (선택)
        """
        draw = ImageDraw.Draw(image)

        # 이미지 중심 표시
        img_width, img_height = image.size
        center_x, center_y = img_width / 2, img_height / 2

        # 중심 십자선
        cross_size = 30
        draw.line([(center_x - cross_size, center_y), (center_x + cross_size, center_y)],
                  fill="yellow", width=3)
        draw.line([(center_x, center_y - cross_size), (center_x, center_y + cross_size)],
                  fill="yellow", width=3)

        # 색상 매핑 (표식별 다른 색상)
        color_map = {
            "red": "red",
            "green": "lime",
            "blue": "cyan",
            "orange": "orange",
            "yellow": "yellow",
            "buoy": "magenta",  # 부표는 보라색
            "cross": "white",   # 십자가는 흰색
        }

        # 탐지된 객체 표시 (최대 10개만)
        for idx, det in enumerate(detections[:10]):
            bbox = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            # 표식 색상 추출
            box_color = "red"  # 기본값
            for color_name, color_value in color_map.items():
                if color_name in label.lower():
                    box_color = color_value
                    break

            # 최고 신뢰도는 굵게
            width = 4 if idx == 0 else 2

            # 바운딩 박스
            draw.rectangle(bbox, outline=box_color, width=width)

            # 중심점 (최고 신뢰도만)
            if idx == 0:
                cx, cy = det["center"]
                draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=box_color, outline="white")

            # 레이블 (배경 추가)
            text = f"{label}: {conf:.2f}"
            text_bbox = draw.textbbox((bbox[0], bbox[1]-22), text)
            draw.rectangle(text_bbox, fill="black")
            draw.text((bbox[0], bbox[1]-20), text, fill=box_color)

        # 항법 명령 표시 (배경 추가)
        cmd_bbox = draw.textbbox((8, 8), command)
        draw.rectangle(cmd_bbox, fill="black")
        draw.text((10, 10), command, fill="lime")

        if save_path:
            image.save(save_path)

        return image

    def batch_test(self, input_dir, output_dir, target_queries=None):
        """test_img 폴더의 모든 이미지로 도킹 테스트

        Args:
            input_dir: 입력 이미지 디렉토리
            output_dir: 출력 디렉토리
            target_queries: 찾을 표식 리스트 (None이면 모든 조합 탐지)
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 지원하는 이미지 확장자
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

        # 이미지 파일 목록
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        if not image_files:
            print(f"경고: {input_dir}에 이미지 파일이 없습니다.")
            return

        # KABOAT 2025 실제 표식 정의
        if target_queries is None:
            target_queries = []

            # 1. 도킹 임무: 형상(삼각형, 원형, 네모) × 색상(빨강, 초록, 파랑, 주황, 노랑)
            docking_shapes = ["triangle", "circle", "square"]
            docking_colors = ["red", "green", "blue", "orange", "yellow"]
            target_queries.extend([f"{color} {shape}" for color in docking_colors for shape in docking_shapes])

            # 2. 장애물 회피: 주황색 부표/원뿔
            target_queries.extend([
                "orange buoy", "orange cone", "orange marker"
            ])

            # 3. 위치유지: 노란색 부표
            target_queries.extend([
                "yellow buoy", "yellow marker", "yellow waypoint"
            ])

            # 4. 탐색: 빨강/초록/파랑 부표
            target_queries.extend([
                "red buoy", "green buoy", "blue buoy"
            ])

            # 5. 항로추종: 빨강/초록 게이트 부표
            target_queries.extend([
                "red navigation buoy", "green navigation buoy",
                "red gate marker", "green gate marker"
            ])

        print(f"\n총 {len(image_files)}개의 이미지를 분석합니다.")
        print(f"탐지할 표식 ({len(target_queries)}개): {target_queries}\n")

        results = []

        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, image_file)
            print(f"[{idx}/{len(image_files)}] 분석 중: {image_file}")

            try:
                # 표식 탐지
                detections, image = self.detect_marker(image_path, target_queries)

                # 신뢰도 순으로 정렬
                detections_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)

                # 최고 신뢰도 탐지 선택
                best_detection = detections_sorted[0] if detections_sorted else None

                # 항법 명령 생성
                command = self.calculate_navigation_command(
                    best_detection,
                    image.size
                )

                result = {
                    "image_file": image_file,
                    "image_path": image_path,
                    "detections": detections_sorted,
                    "best_detection": best_detection,
                    "navigation_command": command,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

                # 결과 출력
                print(f"  총 탐지 수: {len(detections_sorted)}")
                if detections_sorted:
                    print(f"  탐지된 표식 (상위 5개):")
                    for i, det in enumerate(detections_sorted[:5], 1):
                        print(f"    {i}. {det['label']}: {det['confidence']:.3f}")
                    print(f"  명령: {command}")
                else:
                    print(f"  명령: {command}")

                # 시각화 저장
                vis_path = os.path.join(vis_dir, f"vis_{image_file}")
                self.visualize_detection(image, detections_sorted, command, vis_path)
                print(f"  시각화 저장: {vis_path}")
                print("-" * 80)

            except Exception as e:
                print(f"오류 발생 ({image_file}): {str(e)}")
                results.append({
                    "image_file": image_file,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # 결과 저장
        output_file = os.path.join(
            output_dir,
            f"owlv2_all_markers_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n분석 완료! 결과 저장: {output_file}")
        print(f"시각화 이미지: {vis_dir}")

        return results


def main():
    """메인 실행 함수"""
    # 경로 설정
    test_img_dir = "/home/yune/Kaboat2025_sim/test_img"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"

    # 디렉토리 확인
    if not os.path.exists(test_img_dir):
        print(f"오류: {test_img_dir} 디렉토리가 존재하지 않습니다.")
        return

    print("=" * 80)
    print("OWLv2 KABOAT 표식 탐지 테스트")
    print("=" * 80)
    print("\n[KABOAT 임무 표식]")
    print("색상: 빨강, 초록, 파랑, 주황, 노랑")
    print("형상: 삼각형, 원형, 사각형, 십자가")
    print("부표: 빨강, 초록 부표 (항로추종용)")
    print("총 35개 표식 조합 탐지\n")

    # 탐지기 초기화 (신뢰도 낮춤 - 더 많은 후보 탐지)
    detector = OWLv2DockingDetector(confidence_threshold=0.2)

    # 배치 테스트 실행 (target_queries=None이면 모든 조합 탐지)
    results = detector.batch_test(test_img_dir, test_vla_dir, target_queries=None)

    print("\n" + "=" * 80)
    print(f"전체 분석 완료: {len(results)}개 이미지 처리됨")
    print("=" * 80)


if __name__ == "__main__":
    main()
