from pathlib import Path

from src.utils.config import load_config


def test_load_config_resolves_project_relative_paths() -> None:
    config = load_config(Path("config.yaml"))

    assert config["random_seed"] == 2026
    assert config["paths"]["figures"].is_absolute()
    assert config["paths"]["figures"].name == "figures"
