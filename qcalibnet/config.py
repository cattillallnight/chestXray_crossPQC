"""YAML experiment config."""

from __future__ import annotations

from pathlib import Path

import yaml

from qcalibnet.paths import CONFIG_DIR


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else CONFIG_DIR / "default.yaml"
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {cfg_path}")
    return cfg
