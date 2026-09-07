from pathlib import Path
import yaml
from typing import Any, Dict

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_data_config(cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_config()
    return cfg["data"]