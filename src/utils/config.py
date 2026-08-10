"""读取并规范化项目配置。"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置，并将路径字段转换为项目绝对路径。"""
    config_path = path.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key, value in config["paths"].items():
        config["paths"][key] = (config_path.parent / value).resolve()

    return config
