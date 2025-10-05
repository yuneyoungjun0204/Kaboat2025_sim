"""
NanoOWL 비디오 실시간 탐지
파란 +모양, 빨간 세모, 초록 원 탐지 전용
"""

import os
import torch
import cv2
import numpy as np
import time
from PIL import Image
from nanoowl.nanoowl.owl_predictor import OwlPredictor


class NanoOWLVideoDetector:
    def __init__(self, model_name="google/owlvit-base-patch32", confidence_threshold=0.5, nms_iou_threshold=0.5, 
                 min_box_area=100, max_box_area=50000):
        """NanoOWL 비디오 탐지기 초기화"""
        print(f"=" * 80)
        print(f"NanoOWL 비디오 실시간 탐지 초기화")
        print(f"=" * 80)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_box_area = min_box_area  # 최소 박스 면적
        self.max_box_area = max_box_area  # 최대 박스 면적

        # NanoOWL Predictor 로드
        print(f"모델 로딩: {model_name}")
        self.predictor = OwlPredictor(
            model_name,
            device=self.device,
            image_encoder_engine=None  # TensorRT 없이 PyTorch 사용
        )

        self.target_queries = [
            # 파란 십자가/플러스 (인덱스 0-3)
            "a bright blue plus-sign shaped marker for navigation",
            "a simple blue cross symbol on a flat surface",
            "a blue marker shaped like the letter X",
            "a blue cross, not a circle or triangle", # 부정 예시 추가

            # 빨간 삼각형 (인덱스 4-7)
            "a solid red triangle pointing upwards",
            "a triangular sign painted in vibrant red color",
            "a red marker with three sharp corners",
            "a red triangle, not a cross or circle", # 부정 예시 추가

            # 초록 원 (인덱스 8-11)
            "a green circle, not a triangle or cross",
            "a solid green dot or circular marker",
            "a green marker with no corners or straight lines",
            "a green circle, not a triangle or cross", # 부정 예시 추가
        ]

        # 라벨 간소화 매핑 (표시용)
        self.label_mapping = {
            0: "blue cross", 1: "blue cross", 2: "blue cross", 3: "blue cross",
            4: "red triangle", 5: "red triangle", 6: "red triangle", 7: "red triangle",
            8: "green circle", 9: "green circle", 10: "green circle", 11: "green circle",
        }

        # 텍스트 인코딩 미리 수행 (캐싱)
        print(f"텍스트 인코딩 중... ({len(self.target_queries)}개 구체적 프롬프트)")
        self.text_encodings = self.predictor.encode_text(self.target_queries)

        # # 라벨 간소화 매핑 (표시용)
        # self.label_mapping = {
        #     0: "blue cross", 1: "blue cross", 2: "blue cross",
        #     3: "red triangle", 4: "red triangle", 5: "red triangle",
        #     6: "green circle", 7: "green circle", 8: "green circle"
        # }

        print(f"✓ 탐지 대상 (구체적 프롬프트):")
        print(f"  - 파란 +: 3가지 변형")
        print(f"  - 빨간 세모: 3가지 변형")
        print(f"  - 초록 원: 3가지 변형")
        print(f"✓ 박스 크기 제한: {min_box_area} ~ {max_box_area} 픽셀")
        print(f"✓ 초기화 완료 (디바이스: {self.device})")
        print("=" * 80)

    def _calculate_iou(self, box1, box2):
        """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        # 교집합 면적
        intersection_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

        # 각 박스의 면적
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 합집합 면적
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def _filter_by_size(self, detections):
        """박스 크기로 필터링"""
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            if self.min_box_area <= area <= self.max_box_area:
                filtered.append(det)
        return filtered

    def _select_best_per_color(self, detections):
        """색깔별로 가장 높은 신뢰도의 객체만 선택"""
        if not detections:
            return detections

        # 색깔별로 그룹화
        color_groups = {
            "blue": [],
            "red": [],
            "green": []
        }

        for det in detections:
            label = det['label'].lower()
            if "blue" in label:
                color_groups["blue"].append(det)
            elif "red" in label:
                color_groups["red"].append(det)
            elif "green" in label:
                color_groups["green"].append(det)

        # 각 색깔별로 가장 높은 신뢰도 선택
        best_detections = []
        for color, group in color_groups.items():
            if group:
                # 신뢰도 기준으로 정렬하여 가장 높은 것 선택
                best = max(group, key=lambda x: x['confidence'])
                best_detections.append(best)

        return best_detections

    def detect_frame(self, frame_rgb):
        """프레임에서 객체 탐지 (RGB 입력)"""
        # NumPy -> PIL (copy로 writable 경고 방지)
        image = Image.fromarray(frame_rgb.copy())

        # 이미지 전처리
        image_tensor = self.predictor.image_preprocessor.preprocess_pil_image(image)

        # ROI 설정 (전체 이미지)
        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device
        )

        # 이미지 인코딩 (ROI 사용)
        image_encodings = self.predictor.encode_rois(image_tensor, rois, pad_square=False)

        # 탐지 수행 (decode 사용)
        output = self.predictor.decode(
            image_output=image_encodings,
            text_output=self.text_encodings,
            threshold=self.threshold
        )

        # 결과 파싱
        detections = []
        if output.labels is not None and len(output.labels) > 0:
            for i in range(len(output.labels)):
                score = output.scores[i].item()

                if score >= self.threshold:
                    bbox = output.boxes[i].detach().cpu().numpy()
                    label_idx = output.labels[i].item()

                    if label_idx < len(self.target_queries):
                        # 간소화된 라벨 사용 (표시용)
                        label = self.label_mapping.get(label_idx, self.target_queries[label_idx])
                        detections.append({
                            "label": label,
                            "confidence": score,
                            "bbox": [int(b) for b in bbox]
                        })

        # 1. 박스 크기 필터링
        detections = self._filter_by_size(detections)
        
        # 2. 색깔별 최고 신뢰도 선택
        detections = self._select_best_per_color(detections)

        return detections

    def process_video(self, video_path, output_path, display=True):
        """비디오 실시간 탐지"""
        print(f"\n비디오 처리 시작: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"오류: 비디오 열기 실패")
            return None

        # 비디오 속성
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"해상도: {width}x{height}, FPS: {fps}, 총 프레임: {total_frames}")

        # 출력 비디오
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 디스플레이 창 초기화
        if display:
            cv2.namedWindow('NanoOWL Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('NanoOWL Detection', 1280, 720)

        frame_count = 0
        total_time = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # NanoOWL 탐지
                detections = self.detect_frame(frame_rgb)

                inference_time = time.time() - start_time
                total_time += inference_time

                # 시각화
                frame = self._draw(frame, detections, inference_time, frame_count, total_frames)
                out.write(frame)

                if display:
                    cv2.imshow('NanoOWL Detection', frame)
                    # 적절한 딜레이
                    key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
                    if key == ord('q'):
                        print("\n사용자가 중단했습니다.")
                        break

                # 진행 상황
                if frame_count % max(1, total_frames // 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = frame_count / total_time
                    print(f"진행: {progress:.0f}% | FPS: {avg_fps:.1f} | 탐지: {len(detections)}개")

        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n완료! 평균 FPS: {avg_fps:.1f}")
        print(f"저장: {output_path}\n")

        return {"frames": frame_count, "avg_fps": avg_fps}

    def _draw(self, frame, detections, inference_time, frame_num, total_frames):
        """탐지 결과 그리기 (개선된 시각화)"""
        h, w = frame.shape[:2]

        # 색상 매핑 (BGR 형식)
        colors = {
            "blue": (255, 100, 0),    # 밝은 파란색
            "red": (0, 50, 255),      # 밝은 빨간색
            "green": (0, 255, 100)    # 밝은 초록색
        }

        # 탐지 박스 그리기
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            # 색상 선택
            color = (255, 255, 255)  # 기본 흰색
            for c_name, c_val in colors.items():
                if c_name in label.lower():
                    color = c_val
                    break

            # 바운딩 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 코너 강조 (L자 모양)
            corner_len = 15
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

            # 중심점 표시
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 8, color, 2)

            # 라벨 텍스트 (배경 포함)
            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # 텍스트 크기 측정
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # 라벨 위치 결정 (박스 위 또는 아래)
            if y1 - text_h - 10 > 0:
                # 박스 위에 표시
                text_x = x1
                text_y = y1 - 10
                bg_y1 = y1 - text_h - 15
                bg_y2 = y1 - 5
            else:
                # 박스 아래에 표시
                text_x = x1
                text_y = y2 + text_h + 10
                bg_y1 = y2 + 5
                bg_y2 = y2 + text_h + 15

            # 텍스트 배경 (반투명)
            overlay = frame.copy()
            cv2.rectangle(overlay, (text_x, bg_y1), (text_x + text_w + 10, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # 텍스트 그리기
            cv2.putText(frame, text, (text_x + 5, text_y), font, font_scale, color, thickness)

        # 상단 정보 표시 (반투명 배경)
        info = f"NanoOWL | {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f} | Frame: {frame_num}/{total_frames} | Objects: {len(detections)}"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame


def main():
    """메인 함수"""
    test_video_dir = "/home/yune/Kaboat2025_sim/test_video"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"

    print("\n" + "=" * 80)
    print("NanoOWL 비디오 실시간 탐지")
    print("탐지 대상: 파란 +, 빨간 세모, 초록 원")
    print("=" * 80 + "\n")

    # 탐지기 초기화
    detector = NanoOWLVideoDetector(
        model_name="google/owlvit-base-patch32",
        confidence_threshold=0.01,
        nms_iou_threshold=0.001,  # NMS: IoU 0.4 이상이면 중복으로 제거
        min_box_area=1000,         # 최소 박스 면적 (10x10 픽셀)
        max_box_area=50000        # 최대 박스 면적 (223x223 픽셀)
    )

    # 비디오 파일 찾기
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = [
        f for f in os.listdir(test_video_dir)
        if os.path.splitext(f.lower())[1] in video_exts
    ] if os.path.exists(test_video_dir) else []

    if not video_files:
        print(f"비디오 파일이 없습니다: {test_video_dir}")
        return

    # 각 비디오 처리
    for video_file in video_files:
        video_path = os.path.join(test_video_dir, video_file)
        output_name = f"nanoowl_{os.path.splitext(video_file)[0]}.mp4"
        output_path = os.path.join(test_vla_dir, output_name)

        # display=True: 실시간 화면 표시 (q키로 종료)
        # display=False: 화면 없이 비디오만 저장
        detector.process_video(video_path, output_path, display=True)

    print("=" * 80)
    print(f"✓ 전체 완료: {len(video_files)}개 비디오")
    print("=" * 80)


if __name__ == "__main__":
    main()
