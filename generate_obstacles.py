#!/usr/bin/env python3
"""
Sydney Regatta CA에서 custom_obstacle들을 배의 정면을 기준으로
1~2미터 간격으로 랜덤하게 3개씩 15줄 배치하는 스크립트
"""

import random
import numpy as np

def generate_obstacle_positions():
    """장애물 위치 생성"""
    
    # 배의 초기 위치 (추정: -530, 150 부근)
    boat_x = -530.0
    boat_y = 150.0
    
    # 배의 정면 방향 (동쪽으로 가정)
    boat_heading = 0.0  # 0도 = 동쪽
    
    obstacles = []
    
    # 15줄, 각 줄마다 3개씩
    for row in range(15):
        # 배의 정면에서 시작하여 앞으로 이동
        row_distance = 20.0 + row * 8.0  # 첫 줄은 20m 앞, 각 줄마다 8m씩 간격
        
        # 배의 정면 방향으로 이동
        row_x = boat_x + row_distance * np.cos(np.radians(boat_heading))
        row_y = boat_y + row_distance * np.sin(np.radians(boat_heading))
        
        # 각 줄에서 3개의 장애물 배치
        for col in range(3):
            # 좌우로 1~2m 랜덤 간격
            lateral_offset = random.uniform(-3.0, 3.0)  # 좌우로 최대 3m
            forward_offset = random.uniform(-1.0, 1.0)  # 앞뒤로 1m
            
            # 장애물 위치 계산
            obstacle_x = row_x + lateral_offset * np.sin(np.radians(boat_heading))
            obstacle_y = row_y + lateral_offset * np.cos(np.radians(boat_heading))
            obstacle_z = -0.3
            
            # 장애물 이름
            obstacle_name = f"custom_obstacle_{row * 3 + col}"
            
            obstacles.append({
                'name': obstacle_name,
                'x': obstacle_x,
                'y': obstacle_y,
                'z': obstacle_z
            })
    
    return obstacles

def generate_xml_sections(obstacles):
    """XML 섹션 생성"""
    xml_sections = []
    
    for obstacle in obstacles:
        xml_section = f"""    <include>
      <name>{obstacle['name']}</name>
      <pose>{obstacle['x']:.1f} {obstacle['y']:.1f} {obstacle['z']:.1f} 0 0 0</pose>
      <uri>model://custom_obstacle</uri>
    </include>"""
        xml_sections.append(xml_section)
    
    return xml_sections

def main():
    """메인 함수"""
    print("🚢 Sydney Regatta CA 장애물 배치 생성기")
    print("=" * 50)
    
    # 장애물 위치 생성
    obstacles = generate_obstacle_positions()
    
    print(f"생성된 장애물 개수: {len(obstacles)}개")
    print("배치 위치:")
    for i, obstacle in enumerate(obstacles):
        print(f"  {obstacle['name']}: ({obstacle['x']:.1f}, {obstacle['y']:.1f}, {obstacle['z']:.1f})")
    
    # XML 섹션 생성
    xml_sections = generate_xml_sections(obstacles)
    
    print("\n" + "=" * 50)
    print("XML 코드:")
    print("=" * 50)
    
    for xml_section in xml_sections:
        print(xml_section)
    
    # 파일로 저장
    with open('custom_obstacles.xml', 'w') as f:
        for xml_section in xml_sections:
            f.write(xml_section + '\n')
    
    print(f"\n✅ XML 코드가 'custom_obstacles.xml' 파일에 저장되었습니다!")

if __name__ == '__main__':
    main()
