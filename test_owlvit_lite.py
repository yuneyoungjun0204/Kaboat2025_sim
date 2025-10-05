"""
경량화된 OWL-ViT 기반 도킹 시스템
NanoOWL의 최적화 아이디어를 적용한 TensorRT 없는 경량 버전

최적화 기법:
- 작은 모델 사용 (base-patch32)
- Half precision (FP16) 사용
- 이미지 해상도 최적화
- 텍스트 인코딩 캐싱
- 불필요한 연산 제거
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import transformers
OwlViTProcessor = transformers.OwlViTProcessor
OwlViTForObjectDetection = transformers.OwlViTForObjectDetection
import json
from datetime import datetime
import time


class LightweightOwlDetector:
    def __init__(self, model_id="google/owlvit-base-patch32", confidence_threshold=0.1, use_fp16=True):
        """경량화된 OWL-ViT 모델 초기화

        Args:
            model_id: 모델 ID (patch32가 patch16보다 빠름)
            confidence_threshold: 탐지 신뢰도 임계값
            use_fp16: FP16 사용 여부 (2배 빠름, 메모리 절반)
        """
        print(f"경량 모델 로딩 중: {model_id}")
        print(f"FP16 사용: {use_fp16}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        print(f"사용 디바이스: {self.device}")

        # Processor 로드
        self.processor = OwlViTProcessor.from_pretrained(model_id)

        # 모델 로드 (FP16 최적화)
        if self.use_fp16:
            self.model = OwlViTForObjectDetection.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)

        self.model.eval()  # 추론 모드
        self.confidence_threshold = confidence_threshold

        # 텍스트 인코딩 캐시
        self.text_cache = {}

        print("모델 로딩 완료!")

        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.2f}GB (예약: {reserved:.2f}GB)")

    def encode_text_cached(self, text_queries):
        """텍스트 인코딩 (캐싱으로 속도 향상)

        Args:
            text_queries: 텍스트 쿼리 리스트

        Returns:
            캐시된 또는 새로 인코딩된 텍스트
        """
        cache_key = str(text_queries)

        if cache_key not in self.text_cache:
            # 첫 인코딩만 수행하고 캐시
            inputs = self.processor(text=text_queries, return_tensors="pt")

            # numpy 배열을 PyTorch 텐서로 변환
            inputs = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in inputs.items()
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

            with torch.no_grad():
                text_outputs = self.model.owlvit.text_model(**inputs)
                self.text_cache[cache_key] = text_outputs.pooler_output

        return self.text_cache[cache_key]

    def detect_marker(self, image_path, target_queries, max_image_size=768):
        """특정 표식 탐지 (최적화 버전)

        Args:
            image_path: 이미지 경로
            target_queries: 찾을 객체 리스트
            max_image_size: 최대 이미지 크기 (작을수록 빠름)

        Returns:
            detections: 탐지 결과 리스트
            image: PIL Image
            inference_time: 추론 시간
        """
        start_time = time.time()

        # 이미지 로드 및 리사이징 (속도 최적화)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # 이미지 크기 최적화
        if max(image.size) > max_image_size:
            ratio = max_image_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image_resized = image.resize(new_size, Image.Resampling.BILINEAR)
        else:
            image_resized = image

        # 입력 준비
        inputs = self.processor(images=image_resized, text=target_queries, return_tensors="pt")

        # numpy 배열을 PyTorch 텐서로 변환
        inputs = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in inputs.items()
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_fp16:
            # FP16 변환
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # 탐지 수행
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 결과 후처리 (원본 이미지 크기 기준)
        target_sizes = torch.tensor([original_size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        inference_time = time.time() - start_time

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
                "bbox": box,
                "center": [center_x, center_y],
                "width": box[2] - box[0],
                "height": box[3] - box[1]
            })

        return detections, image, inference_time

    def calculate_navigation_command(self, detection, image_size, tolerance=50):
        """탐지된 표식 위치 기반 항법 명령 생성"""
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
                return f"MOVE LEFT (마커가 {int(horizontal_error)}px 오른쪽에 있음)"
            else:
                return f"MOVE RIGHT (마커가 {int(abs(horizontal_error))}px 왼쪽에 있음)"

        if distance_ratio < 0.15:
            return f"MOVE FORWARD (거리: {distance_ratio*100:.1f}%)"
        elif distance_ratio > 0.35:
            return f"MOVE BACK (거리: {distance_ratio*100:.1f}%)"
        else:
            if abs(vertical_error) > tolerance:
                if vertical_error > 0:
                    return f"ADJUST UP (마커가 {int(vertical_error)}px 아래에 있음)"
                else:
                    return f"ADJUST DOWN (마커가 {int(abs(vertical_error))}px 위에 있음)"

            return f"READY TO DOCK - 정렬 완료! (거리: {distance_ratio*100:.1f}%)"

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
        info_text = f"{command} | {inference_time*1000:.0f}ms"
        info_bbox = draw.textbbox((8, 8), info_text)
        draw.rectangle(info_bbox, fill="black")
        draw.text((10, 10), info_text, fill="lime")

        if save_path:
            image.save(save_path)

        return image

    def batch_test(self, input_dir, output_dir, target_queries=None, max_image_size=768):
        """배치 테스트 (최적화 버전)"""
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations_lite")
        os.makedirs(vis_dir, exist_ok=True)

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        if not image_files:
            print(f"경고: {input_dir}에 이미지 파일이 없습니다.")
            return

        # KABOAT 표식 조합
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

        print(f"\n총 {len(image_files)}개의 이미지를 분석합니다.")
        print(f"탐지할 표식 ({len(target_queries)}개)")
        print(f"최대 이미지 크기: {max_image_size}px\n")

        results = []
        total_time = 0

        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, image_file)
            print(f"[{idx}/{len(image_files)}] 분석 중: {image_file}")

            try:
                detections, image, inference_time = self.detect_marker(
                    image_path, target_queries, max_image_size
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

                print(f"  총 탐지 수: {len(detections_sorted)}")
                print(f"  추론 시간: {inference_time*1000:.0f}ms")
                if detections_sorted:
                    print(f"  탐지된 표식 (상위 3개):")
                    for i, det in enumerate(detections_sorted[:3], 1):
                        print(f"    {i}. {det['label']}: {det['confidence']:.3f}")
                    print(f"  명령: {command}")
                else:
                    print(f"  명령: {command}")

                vis_path = os.path.join(vis_dir, f"vis_{image_file}")
                self.visualize_detection(image, detections_sorted, command, inference_time, vis_path)
                print(f"  시각화 저장: {vis_path}")
                print("-" * 80)

            except Exception as e:
                print(f"오류 발생 ({image_file}): {str(e)}")
                results.append({
                    "image_file": image_file,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # 성능 통계
        avg_time = total_time / len(image_files) if image_files else 0
        fps = 1 / avg_time if avg_time > 0 else 0

        print(f"\n{'='*80}")
        print(f"성능 통계:")
        print(f"  평균 추론 시간: {avg_time*1000:.0f}ms")
        print(f"  예상 FPS: {fps:.1f}")
        print(f"{'='*80}\n")

        # 결과 저장
        output_file = os.path.join(
            output_dir,
            f"owlvit_lite_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "performance": {
                    "avg_inference_time_ms": avg_time * 1000,
                    "fps": fps,
                    "total_images": len(image_files)
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"분석 완료! 결과 저장: {output_file}")
        print(f"시각화 이미지: {vis_dir}")

        return results


def main():
    """메인 실행 함수"""
    test_img_dir = "/home/yune/Kaboat2025_sim/test_img"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"

    if not os.path.exists(test_img_dir):
        print(f"오류: {test_img_dir} 디렉토리가 존재하지 않습니다.")
        return

    print("=" * 80)
    print("경량화 OWL-ViT KABOAT 표식 탐지 시스템")
    print("=" * 80)
    print("\n[최적화 기법]")
    print("- 작은 모델 사용: OWL-ViT base-patch32")
    print("- Half precision (FP16) 추론")
    print("- 이미지 해상도 최적화")
    print("- 텍스트 인코딩 캐싱\n")

    # 탐지기 초기화
    detector = LightweightOwlDetector(
        model_id="google/owlvit-base-patch32",
        confidence_threshold=0.01,  # OWL-ViT는 더 낮은 임계값 필요
        use_fp16=True
    )

    # 배치 테스트 실행
    results = detector.batch_test(
        test_img_dir,
        test_vla_dir,
        target_queries=None,
        max_image_size=768  # 이미지 크기 제한 (작을수록 빠름)
    )

    print("\n" + "=" * 80)
    print(f"전체 분석 완료: {len(results)}개 이미지 처리됨")
    print("=" * 80)


if __name__ == "__main__":
    main()
