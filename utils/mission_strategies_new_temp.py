    def execute(self, detected_objects: List[Dict], current_image: np.ndarray,
                agent_heading: float, mission_params: Dict, logger=None,
                raw_detections: List[Dict] = None, **kwargs) -> Tuple[float, float]:
        """
        파란색 부표 주변을 회전 (main_circle.py 로직 사용)

        Args:
            detected_objects: 추적된 객체 (IMM-PDAF 출력, 추정값)
            raw_detections: 원본 탐지 결과 (NanoOWL 직접 출력, 측정값)
            mission_params: 미션 파라미터
        """
        # 트랙바 파라미터 업데이트 (동적 조정)
        circle_base_speed = mission_params.get('circle_base_speed', self.base_speed)
        circle_min_speed = mission_params.get('circle_min_speed', self.min_speed)
        circle_max_turn = mission_params.get('circle_max_turn', self.max_turn_thrust)
        circle_pid_kp = mission_params.get('circle_pid_kp', self.pid_controller.kp)

        # target_x 결정식 파라미터 업데이트 (main_circle.py와 동일)
        self.tx_base_x = mission_params.get('circle_tx_base_x', self.tx_base_x)
        self.tx_slope = mission_params.get('circle_tx_slope', self.tx_slope)
        tx_min_x = mission_params.get('circle_tx_min_x', self.tx_min_x)
        tx_max_x = mission_params.get('circle_tx_max_x', self.tx_max_x)
        # min/max 순서 보장
        self.tx_min_x = min(tx_min_x, tx_max_x - 1.0)
        self.tx_max_x = max(tx_max_x, self.tx_min_x + 1.0)

        # PID Kp 값이 변경되었으면 업데이트
        if circle_pid_kp != self.pid_controller.kp:
            self.pid_controller.kp = circle_pid_kp

        # 속도 및 회전 파라미터 업데이트
        self.base_speed = circle_base_speed
        self.min_speed = circle_min_speed
        self.max_turn_thrust = circle_max_turn

        # 파란색 부표 찾기 (추정값 우선)
        blue_buoy = None
        data_source = "TRACKED"

        # 1. 추정값(tracked)에서 먼저 찾기
        for det in detected_objects:
            if det['label'] == 'blue_buoy':
                blue_buoy = det
                break

        # 2. 추정값에 없으면 측정값(raw)에서 찾기
        if not blue_buoy and raw_detections:
            for det in raw_detections:
                if det['label'] == 'blue_buoy':
                    blue_buoy = det
                    data_source = "RAW"
                    if logger:
                        logger.info("⚠️ 추정값 없음 -> 측정값 사용 (끊김 방지)")
                    break

        # 회전 시작 시간 기록 (완료 확인용)
        if self.circle_start_time is None:
            self.circle_start_time = time.time()
            self.circle_initial_heading = agent_heading if agent_heading else 0.0
            self.previous_heading = agent_heading if agent_heading else 0.0
            self.total_rotation = 0.0

        # 누적 회전 각도 계산
        if agent_heading is not None:
            heading_diff = agent_heading - self.previous_heading

            # 각도 차이 정규화 (-180 ~ 180)
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360

            self.total_rotation += abs(heading_diff)
            self.previous_heading = agent_heading

        # 360도 회전 완료 확인
        if self.total_rotation >= 350:
            self.target_x = None
            if logger:
                logger.info("부표 회전 완료!")
            return 0.0, 0.0

        # 부표 미탐지 시 이전 명령 사용 (main_circle.py와 동일)
        if not blue_buoy:
            self.target_x = None
            left_cmd = self.last_known_left_cmd
            right_cmd = self.last_known_right_cmd
            if logger:
                logger.warn(f"파란색 부표 미탐지: 이전 명령 사용 L={left_cmd:.1f}, R={right_cmd:.1f}")
            return left_cmd, right_cmd

        # 미션 파라미터
        rotation_direction = mission_params.get('rotation_direction', 1)

        # 부표 측정값
        buoy_depth = blue_buoy['depth']
        buoy_x = blue_buoy['center'][0]

        # 회전 모드: 부표를 기준으로 일정한 방향으로 회전
        target_x = self.calculate_rotation_target(rotation_direction, buoy_depth)
        self.target_x = target_x
        error = target_x - buoy_x

        # 조향 명령 계산 (PID)
        steering_command = self.calculate_steering_command(error)

        # 회전 추력 계산
        turn_thrust = steering_command * self.max_turn_thrust

        # 각도에 따른 적응형 속도 계산
        turn_angle = abs(steering_command * 90)
        forward_thrust = self.calculate_rotation_speed(turn_angle)

        # 스러스터 명령 계산
        left_command = forward_thrust - turn_thrust
        right_command = forward_thrust + turn_thrust

        # 마지막으로 성공한 명령 저장 (main_circle.py와 동일)
        self.last_known_left_cmd = left_command
        self.last_known_right_cmd = right_command

        if logger:
            direction_name = "시계방향" if rotation_direction == 1 else "반시계방향"
            logger.info(
                f"Circle [ROTATE][{data_source}]: rotation={self.total_rotation:.1f}°, "
                f"부표 위치=({buoy_x:.1f}px), 깊이={buoy_depth:.3f}m, "
                f"목표={target_x:.1f}px, 오차={error:.1f}px, "
                f"조향={steering_command:.3f}, 방향={direction_name}"
            )

        # 스러스터를 thrust_scale로 변환하여 반환
        left_thrust = (left_command / 1000.0) * self.thrust_scale
        right_thrust = (right_command / 1000.0) * self.thrust_scale

        return left_thrust, right_thrust
