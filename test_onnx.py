#!/usr/bin/env python3
"""
ONNX 모델 출력값 테스트 파일
- main_onnx_v9.py에서 사용하는 모델의 출력값 확인
- 더미 데이터로 모델 입력 후 출력값 분석
"""

import numpy as np
import onnxruntime as ort
import json

def test_onnx_model():
    """ONNX 모델의 입력/출력 구조와 더미 데이터로 출력값 테스트"""
    
    # 모델 경로 (v9와 동일)
    model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-7315183.onnx'
    
    try:
        # ONNX 모델 로드
        session = ort.InferenceSession(model_path)
        print("✅ ONNX 모델 로드 성공!")
        print(f"📁 모델 경로: {model_path}")
        print()
        
        # 입력 정보 확인
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        input_type = input_info.type
        
        print("🔍 모델 입력 정보:")
        print(f"  - 이름: {input_name}")
        print(f"  - 형태: {input_shape}")
        print(f"  - 타입: {input_type}")
        
        # input_shape 처리 (동적 차원 처리)
        processed_shape = []
        for dim in input_shape:
            if isinstance(dim, str):
                # 동적 차원인 경우 기본값 사용
                if dim == 'batch_size' or dim == 'N':
                    processed_shape.append(1)  # 배치 크기 1
                else:
                    processed_shape.append(1)  # 기타 동적 차원은 1로 설정
            else:
                processed_shape.append(dim)
        
        input_shape = tuple(processed_shape)
        print(f"  - 처리된 형태: {input_shape}")
        print()
            
        # 출력 정보 확인
        output_infos = session.get_outputs()
        print("🔍 모델 출력 정보:")
        for i, output_info in enumerate(output_infos):
            print(f"  - 출력 {i+1}:")
            print(f"    이름: {output_info.name}")
            print(f"    형태: {output_info.shape}")
            print(f"    타입: {output_info.type}")
        print()
        
        # 더미 데이터 생성 (v9의 426개 입력 구조)
        print("🎯 더미 데이터 생성 중...")
        
        # Unity와 동일한 구조: 213개 관찰값 사용
        
        # 213개 관찰값 생성 (Unity와 동일한 구조)
        observation_values = []
        
        # 1. LiDAR 거리 (201개) - 0~50m 범위의 랜덤 값
        lidar_data = np.random.uniform(0.1, 50.0, 201)
        observation_values.extend(lidar_data.tolist())
        print(f"  - LiDAR 거리 (201개): {lidar_data[:5]}... (첫 5개)")
        
        # 2. 에이전트 헤딩 (1개) - 0~360도
        heading = np.random.uniform(0.0, 360.0)
        observation_values.append(heading)
        print(f"  - 에이전트 헤딩: {heading}")
        
        # 3. 각속도 Y (1개) - -10~10 rad/s
        angular_vel = np.random.uniform(-10.0, 10.0)
        observation_values.append(angular_vel)
        print(f"  - 각속도 Y: {angular_vel}")
        
        # 4. 에이전트 위치 (2개) - X, Z 좌표
        agent_pos = np.random.uniform(-1000.0, 1000.0, 2)
        observation_values.extend(agent_pos.tolist())
        print(f"  - 에이전트 위치 (X, Z): {agent_pos}")
        
        # 5. 현재 타겟 위치 (2개) - X, Z 좌표
        current_target = np.random.uniform(-1000.0, 1000.0, 2)
        observation_values.extend(current_target.tolist())
        print(f"  - 현재 타겟 위치 (X, Z): {current_target}")
        
        # 6. 이전 타겟 위치 (2개) - X, Z 좌표
        prev_target = np.random.uniform(-1000.0, 1000.0, 2)
        observation_values.extend(prev_target.tolist())
        print(f"  - 이전 타겟 위치 (X, Z): {prev_target}")
        
        # 7. 다음 타겟 위치 (2개) - X, Z 좌표
        next_target = np.random.uniform(-1000.0, 1000.0, 2)
        observation_values.extend(next_target.tolist())
        print(f"  - 다음 타겟 위치 (X, Z): {next_target}")
        
        # 8. 이전 모멘트 입력 (1개) - -1~1
        prev_moment = np.random.uniform(-1.0, 1.0)
        observation_values.append(prev_moment)
        print(f"  - 이전 모멘트 입력: {prev_moment}")
        
        # 9. 이전 포스 입력 (1개) - -1~1
        prev_force = np.random.uniform(-1.0, 1.0)
        observation_values.append(prev_force)
        print(f"  - 이전 포스 입력: {prev_force}")
        
        print(f"  - 관찰값 총 크기: {len(observation_values)} (예상: 211)")
        
        # ONNX 모델이 426개 입력을 기대하므로 stacked input 구조 사용
        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([
            observation_array,  # 첫 번째 211개 데이터
            observation_array   # 두 번째 211개 데이터 (동일한 데이터 반복)
        ]).reshape(1, 426)
        
        dummy_input = stacked_input.flatten()
        print(f"  - 최종 입력 크기: {len(dummy_input)} (예상: 426)")
        print()
        
        # 입력 데이터를 모델 형태로 변환 - 안전한 reshape
        try:
            model_input = dummy_input.reshape(input_shape)
        except ValueError as e:
            print(f"⚠️  reshape 오류: {e}")
            print(f"  - dummy_input 크기: {dummy_input.shape}")
            print(f"  - input_shape: {input_shape}")
            # 기본적으로 (1, -1) 형태로 reshape
            model_input = dummy_input.reshape(1, -1)
            print(f"  - 대체 형태로 reshape: {model_input.shape}")
        
        print("🚀 모델 추론 실행 중...")
        
        # 모델 추론 실행
        outputs = session.run(None, {input_name: model_input})
        
        print("✅ 모델 추론 완료!")
        print()
        
        # 출력 결과 분석
        print("📊 모델 출력 결과:")
        for i, output in enumerate(outputs):
            output_info = output_infos[i]
            print(f"  - 출력 {i+1} ({output_info.name}):")
            print(f"    형태: {output.shape}")
            print(f"    데이터 타입: {output.dtype}")
            print(f"    값 범위: [{np.min(output):.6f}, {np.max(output):.6f}]")
            print(f"    평균: {np.mean(output):.6f}")
            print(f"    표준편차: {np.std(output):.6f}")
            
            # 출력값 상세 정보
            if output.size <= 10:
                print(f"    모든 값: {output.flatten()}")
            else:
                flat_output = output.flatten()
                print(f"    첫 5개 값: {flat_output[:5]}")
                print(f"    마지막 5개 값: {flat_output[-5:]}")
            
            # 특별한 값들 확인
            if np.any(np.isnan(output)):
                print("    ⚠️  NaN 값 발견!")
            if np.any(np.isinf(output)):
                print("    ⚠️  무한대 값 발견!")
            
            print()
        
        # 출력값 해석 시도
        print("🔍 출력값 해석:")
        if len(outputs) >= 2:
            # 일반적으로 첫 번째는 linear velocity, 두 번째는 angular velocity
            linear_vel = outputs[0].flatten()
            angular_vel = outputs[1].flatten()
            
            print(f"  - Linear Velocity: {linear_vel}")
            print(f"  - Angular Velocity: {angular_vel}")
            
            # v9에서 사용하는 스케일링 적용
            v_scale = 1.0
            w_scale = 1.0
            
            scaled_linear = linear_vel * v_scale
            scaled_angular = angular_vel * w_scale
            
            print(f"  - 스케일링된 Linear Velocity: {scaled_linear}")
            print(f"  - 스케일링된 Angular Velocity: {scaled_angular}")
        
        # 결과를 JSON 파일로 저장
        result = {
            "model_path": model_path,
            "input_shape": list(input_shape),
            "input_size": len(dummy_input),
            "outputs": []
        }
        
        for i, output in enumerate(outputs):
            output_info = output_infos[i]
            result["outputs"].append({
                "name": output_info.name,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "min": float(np.min(output)),
                "max": float(np.max(output)),
                "mean": float(np.mean(output)),
                "std": float(np.std(output)),
                "values": output.flatten().tolist()
            })
        
        with open('onnx_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("💾 결과가 'onnx_test_results.json' 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_scenarios():
    """여러 시나리오로 모델 테스트"""
    print("=" * 60)
    print("🎯 여러 시나리오 테스트")
    print("=" * 60)
    
    try:
        model_path = '/home/yuneyoungjun/vrx_ws/src/vrx/scripts/models/Ray-7315183.onnx'
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        raw_input_shape = session.get_inputs()[0].shape
        
        # input_shape 처리 (동적 차원 처리)
        processed_shape = []
        for dim in raw_input_shape:
            if isinstance(dim, str):
                # 동적 차원인 경우 기본값 사용
                if dim == 'batch_size' or dim == 'N':
                    processed_shape.append(1)  # 배치 크기 1
                else:
                    processed_shape.append(1)  # 기타 동적 차원은 1로 설정
            else:
                processed_shape.append(dim)
        
        input_shape = tuple(processed_shape)
        
        scenarios = [
            ("정상 상황", "일반적인 센서 데이터"),
            ("장애물 근처", "LiDAR 거리가 매우 짧은 상황"),
            ("목표 근처", "타겟 위치와 에이전트 위치가 가까운 상황"),
            ("극한 상황", "모든 값이 극값인 상황")
        ]
        
        for scenario_name, description in scenarios:
            print(f"\n🔍 시나리오: {scenario_name}")
            print(f"   설명: {description}")
            
            # 시나리오별 더미 데이터 생성 (v9의 실제 구조 사용)
            observation_values = []
            
            if scenario_name == "정상 상황":
                # 일반적인 값들
                observation_values.extend(np.random.uniform(5.0, 30.0, 201).tolist())  # LiDAR
                observation_values.append(45.0)  # 헤딩
                observation_values.append(0.1)  # 각속도
                observation_values.extend([100.0, 200.0])  # 에이전트 위치
                observation_values.extend([150.0, 250.0])  # 현재 타겟
                observation_values.extend([50.0, 150.0])   # 이전 타겟
                observation_values.extend([200.0, 300.0])  # 다음 타겟
                observation_values.append(0.0)  # 이전 모멘트
                observation_values.append(0.0)  # 이전 포스
                
            elif scenario_name == "장애물 근처":
                # LiDAR 거리가 매우 짧음
                observation_values.extend(np.random.uniform(0.1, 2.0, 201).tolist())  # LiDAR
                observation_values.append(90.0)  # 헤딩
                observation_values.append(-0.5)  # 각속도
                observation_values.extend([0.0, 0.0])  # 에이전트 위치
                observation_values.extend([10.0, 10.0])  # 현재 타겟
                observation_values.extend([-10.0, -10.0])  # 이전 타겟
                observation_values.extend([20.0, 20.0])  # 다음 타겟
                observation_values.append(0.5)  # 이전 모멘트
                observation_values.append(-0.3)  # 이전 포스
                
            elif scenario_name == "목표 근처":
                # 타겟과 매우 가까움
                observation_values.extend(np.random.uniform(10.0, 50.0, 201).tolist())  # LiDAR
                observation_values.append(180.0)  # 헤딩
                observation_values.append(0.0)  # 각속도
                observation_values.extend([100.0, 100.0])  # 에이전트 위치
                observation_values.extend([101.0, 101.0])  # 현재 타겟 (매우 가까움)
                observation_values.extend([99.0, 99.0])   # 이전 타겟
                observation_values.extend([102.0, 102.0])  # 다음 타겟
                observation_values.append(0.0)  # 이전 모멘트
                observation_values.append(0.0)  # 이전 포스
                
            elif scenario_name == "극한 상황":
                # 극값들
                observation_values.extend(np.random.uniform(0.01, 100.0, 201).tolist())  # LiDAR
                observation_values.append(359.9)  # 헤딩
                observation_values.append(10.0)  # 각속도
                observation_values.extend([1000.0, -1000.0])  # 에이전트 위치
                observation_values.extend([-1000.0, 1000.0])  # 현재 타겟
                observation_values.extend([500.0, -500.0])   # 이전 타겟
                observation_values.extend([-500.0, 500.0])  # 다음 타겟
                observation_values.append(1.0)  # 이전 모멘트
                observation_values.append(-1.0)  # 이전 포스
            
        # ONNX 모델이 426개 입력을 기대하므로 stacked input 구조 사용
        observation_array = np.array(observation_values, dtype=np.float32)
        stacked_input = np.concatenate([
            observation_array,  # 첫 번째 211개 데이터
            observation_array   # 두 번째 211개 데이터 (동일한 데이터 반복)
        ]).reshape(1, 426)
        
        dummy_input = stacked_input.flatten()
            
            # 모델 추론 - 안전한 reshape
            try:
                model_input = dummy_input.reshape(input_shape)
            except ValueError as e:
                print(f"   ⚠️  reshape 오류: {e}")
                print(f"   - dummy_input 크기: {dummy_input.shape}")
                print(f"   - input_shape: {input_shape}")
                # 기본적으로 (1, -1) 형태로 reshape
                model_input = dummy_input.reshape(1, -1)
                print(f"   - 대체 형태로 reshape: {model_input.shape}")
            
            outputs = session.run(None, {input_name: model_input})
            
            # 결과 출력
            for i, output in enumerate(outputs):
                flat_output = output.flatten()
                print(f"   출력 {i+1}: {flat_output} (범위: [{np.min(output):.3f}, {np.max(output):.3f}])")
                
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 ONNX 모델 테스트 시작")
    print("=" * 60)
    
    # 기본 테스트
    test_onnx_model()
    
    # 여러 시나리오 테스트
    test_multiple_scenarios()
    
    print("\n✅ 모든 테스트 완료!")