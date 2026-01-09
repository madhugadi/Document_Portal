import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load a YAML configuration file relative to the project root.

    This works reliably from:
    - notebooks
    - scripts
    - tests
    - FastAPI / Streamlit apps

    :param config_path: Path relative to project root (default: config/config.yaml)
    :return: Dictionary containing configuration values
    """
    # Resolve project root: utils/config_loader.py -> utils -> project root
    project_root = Path(__file__).resolve().parents[1]
    full_path = project_root / config_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Config file not found.\nExpected location: {full_path}"
        )

    with open(full_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
