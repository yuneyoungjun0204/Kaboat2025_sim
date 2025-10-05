"""
YOLOE 비디오 실시간 탐지
파란 +모양, 빨간 세모, 초록 원 탐지 전용
"""

import os
import torch
import cv2
import time
from ultralytics import YOLO


class YOLOEVideoDetector:
    def __init__(self, model_name="yoloe-v8l-seg.pt", confidence_threshold=0.25):
        """YOLOE 비디오 탐지기 초기화"""
        print(f"=" * 80)
        print(f"YOLOE 비디오 실시간 탐지 초기화")
        print(f"=" * 80)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = confidence_threshold

        # 모델 로드
        print(f"모델 로딩: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # 탐지 대상: 구체적인 설명 사용 (개선)
        self.target_classes = [
            # 파란 십자가/플러스
            "blue cross marker", "blue plus sign", "blue cross shape",
            "blue x marker", "blue plus shape",

            # 빨간 삼각형
            "red triangle marker", "red triangular shape", "red triangle sign",
            "red triangular marker",

            # 초록 원
            "green circle marker", "green circular shape", "green round marker",
            "green circular marker", "green round shape"
        ]

        # 텍스트 프롬프트 설정
        if hasattr(self.model, 'set_classes'):
            self.model.set_classes(self.target_classes)
            print(f"✓ 탐지 대상: {len(self.target_classes)}개 구체적 클래스")

        print(f"✓ 초기화 완료 (디바이스: {self.device})")
        print("=" * 80)

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
            cv2.namedWindow('YOLOE Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLOE Detection', 1280, 720)

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

                # YOLOE 탐지
                results = self.model.predict(
                    source=frame_rgb,
                    conf=self.threshold,
                    device=self.device,
                    verbose=False,
                    imgsz=640
                )

                inference_time = time.time() - start_time
                total_time += inference_time

                # 탐지 결과 파싱
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())

                        if conf >= self.threshold and cls_id < len(self.target_classes):
                            label = self.target_classes[cls_id]
                            x1, y1, x2, y2 = bbox
                            detections.append({
                                "label": label,
                                "confidence": conf,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                            })

                # 시각화
                frame = self._draw(frame, detections, inference_time, frame_count, total_frames)
                out.write(frame)

                if display:
                    cv2.imshow('YOLOE Detection', frame)
                    # 적절한 딜레이 (30fps 기준 약 33ms)
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
        info = f"YOLOE | {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f} | Frame: {frame_num}/{total_frames}"
        cv2.rectangle(frame, (5, 5), (w-5, 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame


def main():
    """메인 함수"""
    test_video_dir = "/home/yune/Kaboat2025_sim/test_video"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"

    print("\n" + "=" * 80)
    print("YOLOE 비디오 실시간 탐지")
    print("탐지 대상: 파란 +, 빨간 세모, 초록 원")
    print("=" * 80 + "\n")

    # 탐지기 초기화
    detector = YOLOEVideoDetector(
        model_name="yoloe-v8l-seg.pt",
        confidence_threshold=0.15
    )

    print("💡 개선: 구체적인 프롬프트 사용 (정확도 향상)\n")

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
        output_name = f"yoloe_{os.path.splitext(video_file)[0]}.mp4"
        output_path = os.path.join(test_vla_dir, output_name)

        # display=True: 실시간 화면 표시 (q키로 종료)
        # display=False: 화면 없이 비디오만 저장
        detector.process_video(video_path, output_path, display=True)

    print("=" * 80)
    print(f"✓ 전체 완료: {len(video_files)}개 비디오")
    print("=" * 80)


if __name__ == "__main__":
    main()
