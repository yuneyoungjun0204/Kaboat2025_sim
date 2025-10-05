#!/usr/bin/env python3
"""
Configuration Manager
- YAML 기반 설정 관리
- 토픽명, 파라미터 중앙 관리
- 환경변수 지원
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import re


class ConfigManager:
    """설정 파일 관리 클래스"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: 설정 파일 디렉토리 경로 (기본: 프로젝트 루트/config)
        """
        if config_dir is None:
            # 현재 파일 기준으로 config 디렉토리 찾기
            current_dir = Path(__file__).parent.parent
            self.config_dir = current_dir / 'config'
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        # 설정 파일 로드
        self.topics = self._load_config('topics.yaml')
        self.mission = self._load_config('mission_config.yaml')

        # 환경변수 치환
        self._expand_env_vars()

    def _load_config(self, filename: str) -> Dict:
        """YAML 설정 파일 로드"""
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config if config else {}

    def _expand_env_vars(self):
        """환경변수 치환 (${VAR} 형식)"""
        self.mission = self._expand_dict(self.mission)

    def _expand_dict(self, d: Any) -> Any:
        """재귀적으로 딕셔너리 내 환경변수 치환"""
        if isinstance(d, dict):
            return {k: self._expand_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._expand_dict(item) for item in d]
        elif isinstance(d, str):
            # ${VAR} 패턴 찾아서 치환
            pattern = re.compile(r'\$\{(\w+)\}')
            return pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), d)
        else:
            return d

    # ==================== 토픽 관련 메서드 ====================

    def get_topic(self, *keys: str) -> str:
        """토픽명 가져오기

        Args:
            *keys: 계층적 키 (예: 'sensors', 'lidar')

        Returns:
            토픽명 문자열

        Example:
            config.get_topic('sensors', 'lidar')  # '/wamv/sensors/lidars/lidar_wamv_sensor/scan'
        """
        value = self.topics
        for key in keys:
            if not isinstance(value, dict):
                raise KeyError(f"Invalid topic path: {'.'.join(keys)}")
            value = value.get(key)
            if value is None:
                raise KeyError(f"Topic not found: {'.'.join(keys)}")
        return value

    def get_sensor_topic(self, sensor_name: str) -> str:
        """센서 토픽 가져오기"""
        return self.get_topic('sensors', sensor_name)

    def get_actuator_topic(self, actuator_type: str, name: str) -> str:
        """액추에이터 토픽 가져오기"""
        return self.get_topic('actuators', actuator_type, name)

    def get_vrx_topic(self, topic_name: str) -> str:
        """VRX 토픽 가져오기"""
        return self.get_topic('vrx', topic_name)

    def get_qos(self, qos_type: str) -> int:
        """QoS 설정 가져오기"""
        return self.topics.get('qos', {}).get(qos_type, 10)

    # ==================== 미션 설정 관련 메서드 ====================

    def get_param(self, *keys: str, default: Any = None) -> Any:
        """미션 파라미터 가져오기

        Args:
            *keys: 계층적 키
            default: 기본값

        Returns:
            파라미터 값

        Example:
            config.get_param('control', 'thrust_scale')  # 800
        """
        value = self.mission
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
            if value is None:
                return default
        return value

    def get_model_path(self) -> str:
        """ONNX 모델 경로 가져오기"""
        return self.get_param('model', 'path', default='')

    def get_control_params(self) -> Dict:
        """제어 파라미터 전체 가져오기"""
        return self.get_param('control', default={})

    def get_mission_params(self, mission_type: str) -> Dict:
        """미션별 파라미터 가져오기

        Args:
            mission_type: 'gate', 'circle', 'avoid'
        """
        return self.get_param('missions', mission_type, default={})

    def get_sensor_params(self, sensor_name: str) -> Dict:
        """센서 파라미터 가져오기"""
        return self.get_param('sensors', sensor_name, default={})

    def get_timer_period(self, timer_name: str) -> float:
        """타이머 주기 가져오기 (초 단위)

        Args:
            timer_name: 'control_update', 'visualization'

        Returns:
            주기 (초)
        """
        hz = self.get_param('timers', timer_name, default=100)
        return 1.0 / hz

    # ==================== 유틸리티 메서드 ====================

    def reload(self):
        """설정 파일 다시 로드"""
        self.topics = self._load_config('topics.yaml')
        self.mission = self._load_config('mission_config.yaml')
        self._expand_env_vars()

    def print_config(self):
        """설정 내용 출력 (디버깅용)"""
        print("=" * 50)
        print("📋 Topics Configuration:")
        print("=" * 50)
        self._print_dict(self.topics, indent=0)

        print("\n" + "=" * 50)
        print("⚙️  Mission Configuration:")
        print("=" * 50)
        self._print_dict(self.mission, indent=0)

    def _print_dict(self, d: Dict, indent: int = 0):
        """딕셔너리 예쁘게 출력"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


# ==================== 전역 인스턴스 ====================

# 싱글톤 패턴으로 전역 설정 관리자 제공
_global_config: Optional[ConfigManager] = None


def get_config(config_dir: Optional[str] = None) -> ConfigManager:
    """전역 설정 관리자 가져오기

    Args:
        config_dir: 설정 디렉토리 (최초 호출 시만 사용)

    Returns:
        ConfigManager 인스턴스
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigManager(config_dir)

    return _global_config


def reload_config():
    """전역 설정 다시 로드"""
    global _global_config
    if _global_config is not None:
        _global_config.reload()


# ==================== 사용 예제 ====================

if __name__ == '__main__':
    # 사용 예제
    config = ConfigManager()

    # 설정 출력
    config.print_config()

    print("\n" + "=" * 50)
    print("📝 Usage Examples:")
    print("=" * 50)

    # 토픽 가져오기
    print(f"LiDAR topic: {config.get_sensor_topic('lidar')}")
    print(f"Left thruster: {config.get_actuator_topic('thrusters', 'left')}")
    print(f"Waypoint topic: {config.get_vrx_topic('waypoint')}")

    # 파라미터 가져오기
    print(f"\nModel path: {config.get_model_path()}")
    print(f"Thrust scale: {config.get_param('control', 'thrust_scale')}")
    print(f"Gate mission params: {config.get_mission_params('gate')}")

    # 타이머 주기
    print(f"\nControl update period: {config.get_timer_period('control_update')}s")
    print(f"Visualization period: {config.get_timer_period('visualization')}s")
