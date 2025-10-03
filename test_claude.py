"""
Claude API를 사용한 이미지 분석 테스트 스크립트
test_img 폴더의 이미지들을 분석하고 결과를 test_vla 폴더에 저장
"""

import os
import anthropic
from PIL import Image
import json
from datetime import datetime
import base64


class ClaudeVisionAnalyzer:
    def __init__(self, api_key=None):
        """Claude Vision 모델 초기화"""
        print("Claude API 클라이언트 초기화 중...")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY 환경 변수를 설정하거나 api_key를 전달해주세요.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        print("Claude API 클라이언트 초기화 완료!")

    def encode_image(self, image_path):
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def get_media_type(self, image_path):
        """이미지 파일의 미디어 타입 반환"""
        ext = os.path.splitext(image_path.lower())[1]
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(ext, 'image/jpeg')

    def analyze_image(self, image_path, prompt="이 이미지를 상세히 분석해주세요. 주요 객체, 색상, 위치 등을 설명해주세요."):
        """이미지 분석"""
        # 이미지 인코딩
        image_data = self.encode_image(image_path)
        media_type = self.get_media_type(image_path)

        # Claude API 호출
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        return message.content[0].text

    def batch_analyze(self, input_dir, output_dir, custom_prompts=None):
        """test_img 폴더의 모든 이미지 분석"""
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 지원하는 이미지 확장자
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

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
        output_file = os.path.join(output_dir, f"claude_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
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
    print("Claude Vision 이미지 분석 테스트")
    print("=" * 80)

    # 분석기 초기화
    analyzer = ClaudeVisionAnalyzer()

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
