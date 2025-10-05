"""
NanoOWL 기반 KABOAT 도킹 시스템
NVIDIA NanoOWL을 사용한 실시간 표식 탐지

NanoOWL 특징:
- OWL-ViT 최적화 버전
- 텍스트 인코딩 캐싱
- 효율적인 이미지 전처리
- TensorRT 엔진 지원 (선택사항)
"""

import os
import torch
from PIL import Image, ImageDraw
from nanoowl.nanoowl.owl_predictor import OwlPredictor
from nanoowl.nanoowl.owl_drawing import draw_owl_output
import json
from datetime import datetime
import time


class NanoOwlDockingDetector:
    def __init__(self, model_name="google/owlvit-base-patch32", device="cuda", threshold=0.01):
        """NanoOWL 기반 탐지기 초기화

        Args:
            model_name: OWL-ViT 모델 이름
            device: 디바이스 (cuda/cpu)
            threshold: 탐지 임계값
        """
        print(f"=" * 80)
        print(f"NanoOWL 초기화")
        print(f"=" * 80)
        print(f"모델: {model_name}")
        print(f"디바이스: {device}")
        print(f"임계값: {threshold}")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold

        # NanoOWL Predictor 초기화 (TensorRT 엔진 없이)
        print(f"\nNanoOWL Predictor 로딩 중...")
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None  # TensorRT 없이 PyTorch 사용
        )

        print("✓ NanoOWL Predictor 로딩 완료!")

        # GPU 메모리 사용량
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.2f}GB (예약: {reserved:.2f}GB)")

        # 텍스트 인코딩 캐시
        self.text_encodings_cache = {}

        print("=" * 80)

    def encode_text_cached(self, text_queries):
        """텍스트 인코딩 (캐싱)"""
        cache_key = str(text_queries)

        if cache_key not in self.text_encodings_cache:
            print(f"텍스트 인코딩 중... ({len(text_queries)}개 쿼리)")
            text_encodings = self.predictor.encode_text(text_queries)
            self.text_encodings_cache[cache_key] = text_encodings
            print("✓ 텍스트 인코딩 완료 (캐시됨)")

        return self.text_encodings_cache[cache_key]

    def detect_markers(self, image_path, text_queries):
        """표식 탐지

        Args:
            image_path: 이미지 경로
            text_queries: 탐지할 표식 리스트

        Returns:
            detections: 탐지 결과
            image: PIL Image
            inference_time: 추론 시간
        """
        start_time = time.time()

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 이미지를 NanoOWL 입력 형식으로 변환
        image_tensor = self.predictor.image_preprocessor.preprocess_pil_image(image)

        # 텍스트 인코딩 (캐시됨)
        text_encodings = self.encode_text_cached(text_queries)

        # ROI 설정 (전체 이미지)
        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device
        )

        # 이미지 인코딩 (ROI 사용하여 자동 리사이즈)
        image_encodings = self.predictor.encode_rois(image_tensor, rois, pad_square=True)

        # 탐지 수행
        output = self.predictor.decode(
            image_output=image_encodings,
            text_output=text_encodings,
            threshold=self.threshold
        )

        inference_time = time.time() - start_time

        # 결과 필터링 (임계값 적용)
        detections = []

        if output.labels is not None and len(output.labels) > 0:
            for i in range(len(output.labels)):
                score = output.scores[i].item()

                if score >= self.threshold:
                    bbox = output.boxes[i].detach().cpu().numpy()
                    label_idx = output.labels[i].item()
                    label = text_queries[label_idx]

                    # encode_rois는 이미 픽셀 좌표를 반환하므로 그대로 사용
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = float(bbox[2])
                    y2 = float(bbox[3])

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    detections.append({
                        "label": label,
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2],
                        "center": [center_x, center_y],
                        "width": width,
                        "height": height
                    })

        return detections, image, inference_time

    def calculate_navigation_command(self, detection, image_size, tolerance=50):
        """항법 명령 생성"""
        if not detection:
            return "NO TARGET FOUND - SEARCH"

        img_width, img_height = image_size
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        marker_x, marker_y = detection["center"]
        marker_width = detection["width"]

        distance_ratio = marker_width / img_width
        horizontal_error = marker_x - img_center_x
        vertical_error = marker_y - img_center_y

        if abs(horizontal_error) > tolerance:
            if horizontal_error > 0:
                return f"MOVE LEFT ({int(horizontal_error)}px)"
            else:
                return f"MOVE RIGHT ({int(abs(horizontal_error))}px)"

        if distance_ratio < 0.15:
            return f"MOVE FORWARD (거리: {distance_ratio*100:.1f}%)"
        elif distance_ratio > 0.35:
            return f"MOVE BACK (거리: {distance_ratio*100:.1f}%)"
        else:
            if abs(vertical_error) > tolerance:
                if vertical_error > 0:
                    return f"ADJUST UP ({int(vertical_error)}px)"
                else:
                    return f"ADJUST DOWN ({int(abs(vertical_error))}px)"

            return f"✓ READY TO DOCK (거리: {distance_ratio*100:.1f}%)"

    def visualize_detection(self, image, detections, command, inference_time, save_path=None):
        """탐지 결과 시각화"""
        draw = ImageDraw.Draw(image)

        img_width, img_height = image.size
        center_x, center_y = img_width / 2, img_height / 2

        # 중심 십자선
        cross_size = 30
        draw.line([(center_x - cross_size, center_y), (center_x + cross_size, center_y)],
                  fill="yellow", width=3)
        draw.line([(center_x, center_y - cross_size), (center_x, center_y + cross_size)],
                  fill="yellow", width=3)

        # 색상 매핑
        color_map = {
            "red": "red",
            "green": "lime",
            "blue": "cyan",
            "orange": "orange",
            "yellow": "yellow",
            "buoy": "magenta",
            "cross": "white",
        }

        # 탐지된 객체 표시 (최대 10개)
        for idx, det in enumerate(detections[:10]):
            bbox = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            box_color = "red"
            for color_name, color_value in color_map.items():
                if color_name in label.lower():
                    box_color = color_value
                    break

            width = 4 if idx == 0 else 2
            draw.rectangle(bbox, outline=box_color, width=width)

            if idx == 0:
                cx, cy = det["center"]
                draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=box_color, outline="white")

            text = f"{label}: {conf:.2f}"
            text_bbox = draw.textbbox((bbox[0], bbox[1]-22), text)
            draw.rectangle(text_bbox, fill="black")
            draw.text((bbox[0], bbox[1]-20), text, fill=box_color)

        # 명령 및 성능 정보
        info_text = f"{command} | {inference_time*1000:.0f}ms | NanoOWL"
        info_bbox = draw.textbbox((8, 8), info_text)
        draw.rectangle(info_bbox, fill="black")
        draw.text((10, 10), info_text, fill="lime")

        if save_path:
            image.save(save_path)

        return image

    def batch_test(self, input_dir, output_dir, target_queries=None):
        """배치 테스트"""
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations_nanoowl")
        os.makedirs(vis_dir, exist_ok=True)

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        if not image_files:
            print(f"경고: {input_dir}에 이미지 파일이 없습니다.")
            return

        # KABOAT 표식 정의
        if target_queries is None:
            colors = ["red", "green", "blue", "orange", "yellow"]
            shapes = ["triangle", "circle", "square"]
            target_queries = [f"{color} {shape}" for color in colors for shape in shapes]
            target_queries.extend([f"{color} cross" for color in colors])
            target_queries.extend(["cross shape", "cross marker", "plus sign"])
            target_queries.extend([
                "red buoy", "green buoy", "blue buoy",
                "orange buoy", "yellow buoy",
                "red marker buoy", "green marker buoy"
            ])

        print(f"\n{'='*80}")
        print(f"배치 테스트 시작")
        print(f"{'='*80}")
        print(f"이미지 수: {len(image_files)}")
        print(f"표식 수: {len(target_queries)}")
        print(f"{'='*80}\n")

        results = []
        total_time = 0

        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, image_file)
            print(f"[{idx}/{len(image_files)}] {image_file}")

            try:
                detections, image, inference_time = self.detect_markers(
                    image_path, target_queries
                )
                total_time += inference_time

                detections_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)
                best_detection = detections_sorted[0] if detections_sorted else None

                command = self.calculate_navigation_command(best_detection, image.size)

                result = {
                    "image_file": image_file,
                    "image_path": image_path,
                    "detections": detections_sorted,
                    "best_detection": best_detection,
                    "navigation_command": command,
                    "inference_time_ms": inference_time * 1000,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

                print(f"  탐지: {len(detections_sorted)}개")
                print(f"  시간: {inference_time*1000:.0f}ms")
                if detections_sorted:
                    print(f"  TOP3:")
                    for i, det in enumerate(detections_sorted[:3], 1):
                        print(f"    {i}. {det['label']}: {det['confidence']:.3f}")
                    print(f"  → {command}")
                else:
                    print(f"  → {command}")

                vis_path = os.path.join(vis_dir, f"nano_{image_file}")
                self.visualize_detection(image, detections_sorted, command, inference_time, vis_path)
                print(f"  저장: {vis_path}")
                print("-" * 80)

            except Exception as e:
                print(f"오류: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    "image_file": image_file,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # 성능 통계
        avg_time = total_time / len(image_files) if image_files else 0
        fps = 1 / avg_time if avg_time > 0 else 0

        print(f"\n{'='*80}")
        print(f"NanoOWL 성능 통계")
        print(f"{'='*80}")
        print(f"평균 추론 시간: {avg_time*1000:.0f}ms")
        print(f"예상 FPS: {fps:.1f}")
        print(f"{'='*80}\n")

        # 결과 저장
        output_file = os.path.join(
            output_dir,
            f"nanoowl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "performance": {
                    "avg_inference_time_ms": avg_time * 1000,
                    "fps": fps,
                    "total_images": len(image_files),
                    "model": "NanoOWL (PyTorch mode)"
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"결과 저장: {output_file}")
        print(f"시각화: {vis_dir}\n")

        return results


def main():
    """메인 함수"""
    test_img_dir = "/home/yune/Kaboat2025_sim/test_img"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"

    if not os.path.exists(test_img_dir):
        print(f"오류: {test_img_dir} 디렉토리가 존재하지 않습니다.")
        return

    print("\n" + "=" * 80)
    print("NVIDIA NanoOWL KABOAT 도킹 시스템")
    print("=" * 80)
    print("\n[특징]")
    print("✓ NVIDIA NanoOWL 최적화 프레임워크")
    print("✓ 효율적인 텍스트 인코딩 캐싱")
    print("✓ PyTorch 모드 (TensorRT 선택사항)")
    print("✓ 실시간 표식 탐지\n")

    # NanoOWL 탐지기 초기화
    detector = NanoOwlDockingDetector(
        model_name="google/owlvit-base-patch32",
        device="cuda",
        threshold=0.009
    )

    # 배치 테스트
    results = detector.batch_test(
        test_img_dir,
        test_vla_dir,
        target_queries=None
    )

    print("=" * 80)
    print(f"✓ 전체 처리 완료: {len(results)}개 이미지")
    print("=" * 80)


if __name__ == "__main__":
    main()
