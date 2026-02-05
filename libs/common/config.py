# libs/common/config.py
import yaml
import os
import sys
from pathlib import Path

def load_config():
    """
    프로젝트 루트의 conf/config.yaml을 읽어서 딕셔너리로 반환합니다.
    """
    # 현재 파일(libs/common/config.py) 기준으로 프로젝트 루트(chandra/) 찾기
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    config_path = project_root / "conf" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 편의를 위해 project_root 경로도 config에 주입
    config["project_root"] = str(project_root)
    return config
