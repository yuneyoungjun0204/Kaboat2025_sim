"""
YOLOE 기반 KABOAT 도킹 시스템
Real-Time Seeing Anything - 실시간 오픈월드 객체 탐지

YOLOE 특징 (ICCV 2025):
- YOLO-Worldv2 대비 +3.5 AP, 1.4배 빠른 추론
- YOLOv8-L 대비 학습 시간 1/4
- 3가지 프롬프트 방식:
  1️⃣ 텍스트 프롬프트: 클래스 이름으로 탐지
  2️⃣ 비주얼 프롬프트: 예시 이미지로 유사 객체 탐지
  3️⃣ 프롬프트 없음: 모든 객체 자동 탐지
- Zero-shot 성능 SOTA

KABOAT 2025 탐지 대상:
- 도킹: 삼각형/원형/네모 × 빨강/초록/파랑/주황/노랑
- 장애물: 주황색 부표/원뿔
- 위치유지: 노란색 부표
- 탐색: 빨강/초록/파랑 부표
- 항로추종: 빨강/초록 게이트 부표
"""

import os
import torch
from PIL import Image, ImageDraw
import json
from datetime import datetime
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLOE_AVAILABLE = True
except ImportError:
    YOLOE_AVAILABLE = False
    print("경고: ultralytics 패키지가 설치되지 않았습니다.")
    print("설치 방법: pip install ultralytics")


class YOLOEDockingDetector:
    def __init__(self, model_name="yolov8l.pt", confidence_threshold=0.1, device="cuda", use_text_prompt=True):
        """YOLOE 기반 탐지기 초기화

        Args:
            model_name: YOLOE 모델 이름 (yolov8l-world.pt 권장)
            confidence_threshold: 탐지 신뢰도 임계값
            device: 디바이스 (cuda/cpu)
            use_text_prompt: 텍스트 프롬프트 사용 여부
        """
        if not YOLOE_AVAILABLE:
            raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics")

        print(f"=" * 80)
        print(f"YOLOE KABOAT 시스템 초기화")
        print(f"=" * 80)
        print(f"모델: {model_name}")
        print(f"디바이스: {device}")
        print(f"임계값: {confidence_threshold}")
        print(f"텍스트 프롬프트: {use_text_prompt}")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = confidence_threshold
        self.use_text_prompt = use_text_prompt

        # YOLOE 모델 로드
        print(f"\nYOLOE 모델 로딩 중...")
        self.model = YOLO(model_name)

        # 디바이스 설정
        if self.device == "cuda":
            self.model.to("cuda")

        # KABOAT 2025 클래스 정의 (텍스트 프롬프트)
        self.kaboat_classes = self._define_kaboat_classes()

        if self.use_text_prompt and hasattr(self.model, 'set_classes'):
            print(f"\n📝 텍스트 프롬프트 모드: {len(self.kaboat_classes)}개 클래스 설정")
            self.model.set_classes(self.kaboat_classes)
        else:
            print(f"\n🔍 자동 탐지 모드: 프롬프트 없이 모든 객체 탐지")

        print("✓ YOLOE 모델 로딩 완료!")

        # GPU 메모리 사용량
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.2f}GB (예약: {reserved:.2f}GB)")

        print("=" * 80)

    def _define_kaboat_classes(self):
        """KABOAT 2025 대회 규정에 따른 탐지 클래스 정의

        Returns:
            list: 탐지할 객체 클래스 리스트
        """
        classes = []

        # 1. 도킹 임무 표식 (형상 × 색상)
        docking_shapes = ["triangle", "circle", "square", "triangular marker", "circular marker", "square marker"]
        docking_colors = ["red", "green", "blue", "orange", "yellow"]

        # 단일 색상 형상
        classes.extend([f"{color} {shape}" for color in docking_colors for shape in ["triangle", "circle", "square"]])

        # 2. 부표 (Buoy) - 모든 임무에서 사용
        buoy_types = [
            # 장애물 회피
            "orange buoy", "orange cone", "orange marker", "orange obstacle",

            # 위치유지
            "yellow buoy", "yellow marker", "yellow waypoint",

            # 탐색 및 항로추종
            "red buoy", "green buoy", "blue buoy",

            # 게이트 부표
            "navigation buoy", "gate buoy", "marker buoy",

            # 일반 부표
            "buoy", "floating marker", "marine marker"
        ]
        classes.extend(buoy_types)

        # 3. 도킹 스테이션
        docking_station = [
            "docking station", "dock", "landing pad",
            "docking platform", "berthing area"
        ]
        classes.extend(docking_station)

        # 4. 배 및 장애물
        obstacles = [
            "boat", "ship", "vessel", "other boat",
            "obstacle", "barrier"
        ]
        classes.extend(obstacles)

        # 중복 제거
        return list(set(classes))

    def detect_markers(self, image_path, text_queries=None):
        """표식 탐지 (3가지 프롬프트 방식 지원)

        Args:
            image_path: 이미지 경로
            text_queries: 탐지할 표식 리스트 (옵션, None이면 모든 KABOAT 클래스 사용)

        Returns:
            detections: 탐지 결과
            image: PIL Image
            inference_time: 추론 시간
        """
        start_time = time.time()

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 1️⃣ 텍스트 프롬프트 방식
        if self.use_text_prompt and hasattr(self.model, 'set_classes'):
            if text_queries is not None:
                # 특정 쿼리만 탐지
                self.model.set_classes(text_queries)
            else:
                # 모든 KABOAT 클래스 탐지
                self.model.set_classes(self.kaboat_classes)

        # YOLOE로 탐지 수행
        results = self.model.predict(
            source=image_path,
            conf=self.threshold,
            device=self.device,
            verbose=False,
            imgsz=640  # 이미지 크기 (빠른 추론: 640, 정확도: 1280)
        )

        inference_time = time.time() - start_time

        # 결과 파싱
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # 바운딩 박스 (xyxy 형식)
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())

                # 클래스 ID 및 이름
                class_id = int(boxes.cls[i].cpu().numpy())

                # 클래스 이름 가져오기
                if hasattr(self.model, 'names') and class_id < len(self.model.names):
                    label = self.model.names[class_id]
                else:
                    # 텍스트 프롬프트 모드인 경우
                    query_list = text_queries if text_queries else self.kaboat_classes
                    label = query_list[class_id] if class_id < len(query_list) else f"object_{class_id}"

                if confidence >= self.threshold:
                    x1, y1, x2, y2 = bbox

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    detections.append({
                        "label": label,
                        "confidence": float(confidence),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [float(center_x), float(center_y)],
                        "width": float(width),
                        "height": float(height),
                        "class_id": class_id
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
        info_text = f"{command} | {inference_time*1000:.0f}ms | YOLOE"
        info_bbox = draw.textbbox((8, 8), info_text)
        draw.rectangle(info_bbox, fill="black")
        draw.text((10, 10), info_text, fill="lime")

        if save_path:
            image.save(save_path)

        return image

    def batch_test(self, input_dir, output_dir, target_queries=None):
        """배치 테스트"""
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations_yoloe")
        os.makedirs(vis_dir, exist_ok=True)

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
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

                vis_path = os.path.join(vis_dir, f"yoloe_{image_file}")
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
        print(f"YOLOE 성능 통계")
        print(f"{'='*80}")
        print(f"평균 추론 시간: {avg_time*1000:.0f}ms")
        print(f"예상 FPS: {fps:.1f}")
        print(f"{'='*80}\n")

        # 결과 저장
        output_file = os.path.join(
            output_dir,
            f"yoloe_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "performance": {
                    "avg_inference_time_ms": avg_time * 1000,
                    "fps": fps,
                    "total_images": len(image_files),
                    "model": "YOLOE"
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"결과 저장: {output_file}")
        print(f"시각화: {vis_dir}\n")

        return results

    def process_video(self, video_path, output_path, target_queries=None, display=False):
        """비디오 실시간 탐지

        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            target_queries: 탐지할 표식 리스트 (None이면 KABOAT 클래스 사용)
            display: 실시간 화면 표시 여부

        Returns:
            dict: 처리 결과 통계
        """
        print(f"\n{'='*80}")
        print(f"비디오 실시간 탐지 시작")
        print(f"{'='*80}")
        print(f"입력: {video_path}")
        print(f"출력: {output_path}")

        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"오류: 비디오를 열 수 없습니다: {video_path}")
            return None

        # 비디오 속성
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"해상도: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"총 프레임: {total_frames}")
        print(f"{'='*80}\n")

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 탐지 쿼리 설정
        if target_queries is None:
            target_queries = []
            docking_shapes = ["triangle", "circle", "square"]
            docking_colors = ["red", "green", "blue", "orange", "yellow"]
            target_queries.extend([f"{color} {shape}" for color in docking_colors for shape in docking_shapes])
            target_queries.extend([
                "orange buoy", "orange cone", "orange marker",
                "yellow buoy", "yellow marker",
                "red buoy", "green buoy", "blue buoy"
            ])

        # 텍스트 프롬프트 설정
        if self.use_text_prompt and hasattr(self.model, 'set_classes'):
            self.model.set_classes(target_queries)

        # 프레임 처리
        frame_count = 0
        total_inference_time = 0
        detection_stats = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                # BGR -> RGB 변환 (YOLO는 RGB 사용)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # YOLO 탐지 (NumPy 배열 직접 입력)
                results = self.model.predict(
                    source=frame_rgb,
                    conf=self.threshold,
                    device=self.device,
                    verbose=False,
                    imgsz=640
                )

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # 탐지 결과 파싱
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())

                        query_list = target_queries if target_queries else self.kaboat_classes
                        label = query_list[class_id] if class_id < len(query_list) else f"object_{class_id}"

                        if confidence >= self.threshold:
                            x1, y1, x2, y2 = bbox
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1

                            detections.append({
                                "label": label,
                                "confidence": float(confidence),
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "center": [float(center_x), float(center_y)],
                                "width": float(width),
                                "height": float(height),
                                "class_id": class_id
                            })

                # 탐지 결과 정렬
                detections_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)
                best_detection = detections_sorted[0] if detections_sorted else None

                # 항법 명령
                command = self.calculate_navigation_command(best_detection, (width, height))

                # 프레임에 시각화
                frame = self._draw_detections_on_frame(
                    frame, detections_sorted, command, inference_time, frame_count, total_frames
                )

                # 비디오 저장
                out.write(frame)

                # 통계 저장
                detection_stats.append({
                    "frame": frame_count,
                    "detections": len(detections_sorted),
                    "inference_time_ms": inference_time * 1000,
                    "command": command
                })

                # 실시간 표시
                if display:
                    cv2.imshow('YOLOE Real-time Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # 진행 상황 출력 (10% 단위)
                if frame_count % max(1, total_frames // 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                    print(f"진행: {progress:.0f}% ({frame_count}/{total_frames}) | "
                          f"평균 FPS: {avg_fps:.1f} | 탐지: {len(detections_sorted)}개")

        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()

        # 최종 통계
        avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0

        print(f"\n{'='*80}")
        print(f"비디오 처리 완료")
        print(f"{'='*80}")
        print(f"처리 프레임: {frame_count}/{total_frames}")
        print(f"평균 추론 시간: {avg_inference_time*1000:.0f}ms")
        print(f"평균 FPS: {avg_fps:.1f}")
        print(f"출력 저장: {output_path}")
        print(f"{'='*80}\n")

        return {
            "total_frames": frame_count,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "avg_fps": avg_fps,
            "detection_stats": detection_stats
        }

    def _draw_detections_on_frame(self, frame, detections, command, inference_time, frame_num, total_frames):
        """프레임에 탐지 결과 그리기"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # 중심 십자선
        cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 255), 2)

        # 색상 매핑
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "buoy": (255, 0, 255),
        }

        # 탐지된 객체 표시
        for idx, det in enumerate(detections[:10]):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = det["label"]
            conf = det["confidence"]

            # 색상 선택
            box_color = (0, 0, 255)  # 기본: 빨강
            for color_name, color_value in color_map.items():
                if color_name in label.lower():
                    box_color = color_value
                    break

            # 바운딩 박스
            thickness = 3 if idx == 0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # 중심점 (최우선 타겟만)
            if idx == 0:
                cx, cy = [int(v) for v in det["center"]]
                cv2.circle(frame, (cx, cy), 6, box_color, -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)

            # 라벨
            text = f"{label}: {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # 상단 정보: 명령 + 성능
        info_text = f"{command} | {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f}"
        cv2.rectangle(frame, (5, 5), (width - 5, 35), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 하단 정보: 프레임 번호
        frame_text = f"Frame: {frame_num}/{total_frames}"
        cv2.putText(frame, frame_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame


def main():
    """메인 함수"""
    import sys

    test_img_dir = "/home/yune/Kaboat2025_sim/test_img"
    test_vla_dir = "/home/yune/Kaboat2025_sim/test_vla"
    test_video_dir = "/home/yune/Kaboat2025_sim/test_video"

    # 모드 선택: 이미지 또는 비디오
    mode = "video" if len(sys.argv) > 1 and sys.argv[1] == "video" else "image"

    print("\n" + "=" * 80)
    print("YOLOE KABOAT 도킹 시스템 [ICCV 2025]")
    print("=" * 80)
    print("\n[KABOAT 2025 임무]")
    print("🎯 도킹: 삼각형/원형/네모 × 5가지 색상")
    print("🟠 장애물 회피: 주황색 부표/원뿔")
    print("🟡 위치유지: 노란색 부표")
    print("🔴🟢🔵 탐색: 빨강/초록/파랑 부표")
    print("🚢 항로추종: 빨강/초록 게이트 부표\n")

    print("[YOLOE 특징]")
    print("✓ Real-Time Seeing Anything")
    print("✓ 3가지 프롬프트 방식 (텍스트/비주얼/프롬프트 없음)")
    print("✓ YOLO-Worldv2 대비 1.4배 빠른 추론")
    print("✓ Zero-shot 성능 SOTA\n")

    # YOLOE 탐지기 초기화
    # 모델 선택:
    # - yolov8l.pt: 기본 YOLO (COCO 클래스만)
    # - yolov8l-world.pt: YOLO-World (텍스트 프롬프트 지원)
    # - yoloe-*.pt: YOLOE 공식 모델 (권장)
    detector = YOLOEDockingDetector(
        model_name="yoloe-v8l-seg.pt",  # TODO: yoloe-v8l.pt로 교체
        device="cuda",
        confidence_threshold=0.25,
        use_text_prompt=True  # 텍스트 프롬프트 사용
    )

    if mode == "video":
        # 비디오 처리 모드
        print("\n[모드: 비디오 실시간 탐지]\n")

        # test_video 디렉토리 생성 (없으면)
        os.makedirs(test_video_dir, exist_ok=True)

        # 비디오 파일 찾기
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = [
            f for f in os.listdir(test_video_dir)
            if os.path.splitext(f.lower())[1] in video_extensions
        ] if os.path.exists(test_video_dir) else []

        if not video_files:
            print(f"경고: {test_video_dir}에 비디오 파일이 없습니다.")
            print(f"비디오 파일을 {test_video_dir}에 추가한 후 다시 실행하세요.")
            print(f"지원 형식: {', '.join(video_extensions)}")
            return

        # 각 비디오 처리
        for video_file in video_files:
            video_path = os.path.join(test_video_dir, video_file)
            output_name = f"yoloe_detected_{os.path.splitext(video_file)[0]}.mp4"
            output_path = os.path.join(test_vla_dir, output_name)

            print(f"\n처리 중: {video_file}")
            stats = detector.process_video(
                video_path=video_path,
                output_path=output_path,
                target_queries=None,
                display=True  # True로 변경하면 실시간 화면 표시
            )

            if stats:
                print(f"✓ 비디오 처리 완료: {output_path}")

        print("\n" + "=" * 80)
        print(f"✓ 전체 비디오 처리 완료: {len(video_files)}개")
        print("=" * 80)

    else:
        # 이미지 배치 처리 모드
        print("\n[모드: 이미지 배치 처리]\n")
        print("💡 비디오 처리 모드: python test_yoloe_docking.py video\n")

        if not os.path.exists(test_img_dir):
            print(f"오류: {test_img_dir} 디렉토리가 존재하지 않습니다.")
            return

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
