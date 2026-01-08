from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_THRESHOLDS_PATH = Path(__file__).with_name("thresholds.yml")


def load_thresholds(path: str | Path | None = None) -> Dict[str, Any]:
    thresholds_path = Path(path) if path else DEFAULT_THRESHOLDS_PATH
    data = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Thresholds config must be a mapping.")
    return data
