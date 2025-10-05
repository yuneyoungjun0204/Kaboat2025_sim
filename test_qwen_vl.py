"""
Qwen2.5-VL-7B 모델을 사용한 이미지 분석 테스트 스크립트
test_img 폴더의 이미지들을 분석하고 결과를 test_vla 폴더에 저장

Qwen2.5-VL-7B 특징:
- 정확도: GPT-4o와 거의 동등 (1점 차이)
- 메모리: 7B 모델, 양자화 시 ~4GB
- 복잡한 지시 이해 가능
- 2025년 최고 성능 오픈소스 VLM 중 하나

선박 자동 주차 내비게이션 전용 설정
"""

import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import json
from datetime import datetime


class QwenVLAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", use_quantization=True):
        """Qwen2.5-VL 모델 초기화

        Args:
            model_id: 모델 ID
            use_quantization: 4-bit 양자화 사용 여부 (메모리 절약)
        """
        print(f"모델 로딩 중: {model_id}")
        print(f"양자화 사용: {use_quantization}")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")

        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(model_id)

        # 양자화 설정
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

        print("모델 로딩 완료!")

        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.2f}GB (예약: {reserved:.2f}GB)")

    def analyze_image(self, image_path, prompt="Find red shape on dock and tell boat direction"):
        """이미지 분석 (선박 내비게이션 최적화)"""
        # 메시지 구성 (Qwen2.5-VL 형식)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 준비
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 생성 (속도 최적화)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
            )

        # 입력 토큰 제거하고 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def batch_analyze(self, input_dir, output_dir, custom_prompts=None):
        """test_img 폴더의 모든 이미지 분석"""
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

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

        print(f"\n총 {len(image_files)}개의 이미지를 분석합니다.\n")

        results = []

        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, image_file)
            print(f"[{idx}/{len(image_files)}] 분석 중: {image_file}")

            try:
                # 커스텀 프롬프트가 있으면 사용
                prompt = custom_prompts.get(image_file) if custom_prompts else None
                if not prompt:
                    # 선박 자동 주차 내비게이션 프롬프트
                    prompt = "Find red shape on dock. If red shape is not at image center, tell boat direction to move: 'Move LEFT/RIGHT/UP/DOWN/FORWARD/BACK'. If centered, say 'STOP - RED SHAPE CENTERED'. Be brief."

                # 이미지 분석
                analysis = self.analyze_image(image_path, prompt)

                result = {
                    "image_file": image_file,
                    "image_path": image_path,
                    "prompt": prompt,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

                # 개별 결과 출력
                print(f"결과: {analysis[:200]}..." if len(analysis) > 200 else f"결과: {analysis}")
                print("-" * 80)

            except Exception as e:
                print(f"오류 발생 ({image_file}): {str(e)}")
                results.append({
                    "image_file": image_file,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # 결과 저장
        output_file = os.path.join(output_dir, f"qwen_vl_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n분석 완료! 결과 저장: {output_file}")

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
    print("Qwen2.5-VL-7B 이미지 분석 테스트 (선박 자동 주차)")
    print("=" * 80)

    # 분석기 초기화
    analyzer = QwenVLAnalyzer()

    # 배치 분석 실행
    custom_prompts = {
        # 필요시 특정 이미지에 대한 커스텀 프롬프트 지정 가능
    }

    results = analyzer.batch_analyze(test_img_dir, test_vla_dir, custom_prompts)

    print("\n" + "=" * 80)
    print(f"전체 분석 완료: {len(results)}개 이미지 처리됨")
    print("=" * 80)


if __name__ == "__main__":
    main()
