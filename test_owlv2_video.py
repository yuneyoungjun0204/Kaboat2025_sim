"""
OWLv2 비디오 실시간 탐지
파란 +모양, 빨간 세모, 초록 원 탐지 전용
"""

import os
import torch
import cv2
import numpy as np
import time
from PIL import Image
import transformers

Owlv2Processor = transformers.Owlv2Processor
Owlv2ForObjectDetection = transformers.Owlv2ForObjectDetection


class OWLv2VideoDetector:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", confidence_threshold=0.1,
                 use_fp16=True, resize_input=True, frame_skip=1, input_size=(480, 320)):
        """OWLv2 비디오 탐지기 초기화

        Args:
            model_id: 모델 ID
            confidence_threshold: 신뢰도 임계값
            use_fp16: FP16 반정밀도 사용 (속도 2배 향상)
            resize_input: 입력 이미지 크기 줄이기 (속도 향상)
            frame_skip: N 프레임마다 탐지 (1=모든 프레임, 2=격프레임)
            input_size: 입력 이미지 크기 (width, height) - 작을수록 빠름
        """
        print(f"=" * 80)
        print(f"OWLv2 비디오 실시간 탐지 초기화 (최적화 모드)")
        print(f"=" * 80)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = confidence_threshold
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.resize_input = resize_input
        self.frame_skip = frame_skip
        self.resize_size = input_size if resize_input else None

        # 모델 로드
        print(f"모델 로딩: {model_id}")
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)

        # FP16 최적화
        if self.use_fp16:
            self.model.half()
            print("✓ FP16 반정밀도 활성화 (속도 2배 향상)")

        # 평가 모드 (dropout/batchnorm 비활성화)
        self.model.eval()

        # CUDA 최적화
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print("✓ CUDA 최적화 활성화 (cuDNN benchmark, TF32)")

        # PyTorch 2.0+ compile (OWLv2와 호환 문제로 비활성화)
        # try:
        #     if hasattr(torch, 'compile') and self.device == "cuda":
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         print("✓ torch.compile() 활성화 (추가 속도 향상)")
        # except Exception as e:
        #     print(f"⚠ torch.compile() 실패 (무시): {e}")

        # 탐지 대상: 파란 +, 빨간 세모, 초록 원
        self.target_queries = [
            "blue cross",
            "red triangle",
            "green circle"
        ]

        print(f"✓ 탐지 대상: {self.target_queries}")
        if self.resize_input:
            print(f"✓ 입력 리사이즈: {self.resize_size} (속도 향상)")
        if self.frame_skip > 1:
            print(f"✓ 프레임 스킵: {self.frame_skip} (탐지는 {self.frame_skip}프레임마다)")
        print(f"✓ 초기화 완료 (디바이스: {self.device})")
        print("=" * 80)

    def detect_frame(self, frame_rgb, original_size=None):
        """프레임에서 객체 탐지 (RGB 입력)"""
        # NumPy -> PIL
        image = Image.fromarray(frame_rgb)
        original_size = original_size or image.size

        # 입력 리사이즈 (속도 향상)
        if self.resize_input and self.resize_size:
            image = image.resize(self.resize_size, Image.BILINEAR)

        # 입력 준비
        inputs = self.processor(
            text=self.target_queries,
            images=image,
            return_tensors="pt"
        )

        # NumPy -> Torch 텐서
        inputs = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in inputs.items()
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # FP16 변환
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # 탐지 (autocast로 추가 최적화)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_fp16):
            outputs = self.model(**inputs)

        # 후처리 (원본 크기로 스케일링)
        target_sizes = torch.tensor([original_size[::-1]]).to(self.device)  # [H, W]
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )[0]

        # 결과 파싱
        detections = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            box = box.cpu().tolist()
            score = score.cpu().item()
            label = self.target_queries[label_idx.cpu().item()]

            detections.append({
                "label": label,
                "confidence": score,
                "bbox": [int(b) for b in box]  # [x1, y1, x2, y2]
            })

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
            cv2.namedWindow('OWLv2 Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OWLv2 Detection', 1280, 720)

        frame_count = 0
        total_time = 0
        last_detections = []  # 마지막 탐지 결과 캐시 (프레임 스킵용)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                # 프레임 스킵 로직: N 프레임마다만 탐지
                if frame_count % self.frame_skip == 1:
                    # BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # OWLv2 탐지 (원본 크기 정보 전달)
                    detections = self.detect_frame(frame_rgb, original_size=(width, height))
                    last_detections = detections  # 결과 캐싱

                    inference_time = time.time() - start_time
                    total_time += inference_time
                else:
                    # 스킵된 프레임은 마지막 탐지 결과 재사용
                    detections = last_detections
                    inference_time = 0  # 탐지 안 함

                # 시각화
                frame = self._draw(frame, detections, inference_time, frame_count, total_frames)
                out.write(frame)

                if display:
                    cv2.imshow('OWLv2 Detection', frame)
                    # 적절한 딜레이 (30fps 기준 약 33ms)
                    key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
                    if key == ord('q'):
                        print("\n사용자가 중단했습니다.")
                        break

                # 진행 상황
                if frame_count % max(1, total_frames // 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    actual_fps = frame_count / total_time if total_time > 0 else 0
                    print(f"진행: {progress:.0f}% | 실제 FPS: {actual_fps:.1f} | 탐지: {len(detections)}개")

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
        """탐지 결과 그리기"""
        h, w = frame.shape[:2]

        # 색상 매핑
        colors = {
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "green": (0, 255, 0)
        }

        # 탐지 박스
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            # 색상 선택
            color = (255, 255, 255)
            for c_name, c_val in colors.items():
                if c_name in label.lower():
                    color = c_val
                    break

            # 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # 라벨
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 정보 표시
        if inference_time > 0:
            info = f"OWLv2 | {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f} | Frame: {frame_num}/{total_frames}"
        else:
            info = f"OWLv2 | CACHED | Frame: {frame_num}/{total_frames}"
        cv2.rectangle(frame, (5, 5), (w-5, 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame


def main():
    """메인 함수"""
    test_video_dir = "/home/yune/Kaboat2025_sim/test_video"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"

    print("\n" + "=" * 80)
    print("OWLv2 비디오 실시간 탐지 (극한 최적화 모드)")
    print("탐지 대상: 파란 +, 빨간 세모, 초록 원")
    print("=" * 80 + "\n")

    # 탐지기 초기화 (극한 최적화)
    detector = OWLv2VideoDetector(
        model_id="google/owlv2-base-patch16-ensemble",
        confidence_threshold=0.3,
        use_fp16=True,        # FP16 반정밀도 (속도 2배)
        resize_input=True,    # 입력 리사이즈 (속도 향상)
        frame_skip=10,         # 5프레임마다 탐지 (속도 5배)
        input_size=(320, 240) # 매우 작은 입력 (속도 4배)
    )

    print("\n💡 속도 우선 설정:")
    print("  - 입력 크기: 320×240 (원본의 1/4)")
    print("  - 프레임 스킵: 5 (5프레임마다 탐지)")
    print("  - 예상 속도 향상: 10-20배\n")

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
        output_name = f"owlv2_{os.path.splitext(video_file)[0]}.mp4"
        output_path = os.path.join(test_vla_dir, output_name)

        # display=True: 실시간 화면 표시 (q키로 종료)
        # display=False: 화면 없이 비디오만 저장
        detector.process_video(video_path, output_path, display=True)

    print("=" * 80)
    print(f"✓ 전체 완료: {len(video_files)}개 비디오")
    print("=" * 80)


if __name__ == "__main__":
    main()
