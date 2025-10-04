"""
SmolVLM-2.2B 모델을 사용한 이미지 분석 테스트 스크립트
test_img 폴더의 이미지들을 분석하고 결과를 test_vla 폴더에 저장

SmolVLM 특징:
- 메모리: 4.9GB (5.78GB GPU에 최적)
- 속도: Qwen2-VL 대비 3.3-4.5배 빠름
- 2025년 최신 경량 Vision-Language 모델
"""

import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import json
from datetime import datetime


class SmolVLMAnalyzer:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-Instruct"):
        """SmolVLM 모델 초기화"""
        print(f"모델 로딩 중: {model_id}")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")

        # 모델과 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
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

    def analyze_image(self, image_path, prompt="이 이미지를 상세히 분석해주세요. 주요 객체, 색상, 위치 등을 설명해주세요."):
        """이미지 분석"""
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 입력 준비
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 텍스트 생성을 위한 입력 준비
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            return_tensors="pt"
        ).to(self.device)

        # 생성
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.7,
                top_p=0.9
            )

        # 디코딩
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)

        return generated_text

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
                    prompt = "이 이미지를 상세히 분석해주세요. 주요 객체, 색상, 위치, 그리고 해양 환경에서의 특징을 설명해주세요."

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
        output_file = os.path.join(output_dir, f"smolvlm_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
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
    print("SmolVLM-2.2B 이미지 분석 테스트")
    print("=" * 80)

    # 분석기 초기화
    analyzer = SmolVLMAnalyzer()

    # 배치 분석 실행
    # 필요시 특정 이미지에 대한 커스텀 프롬프트 지정 가능
    custom_prompts = {
        # 예시: "buoy.jpg": "이 이미지에서 부표(buoy)를 찾아 위치와 색상을 설명해주세요.",
    }

    results = analyzer.batch_analyze(test_img_dir, test_vla_dir, custom_prompts)

    print("\n" + "=" * 80)
    print(f"전체 분석 완료: {len(results)}개 이미지 처리됨")
    print("=" * 80)


if __name__ == "__main__":
    main()
