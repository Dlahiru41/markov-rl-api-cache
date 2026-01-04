"""Configuration loader for markov-rl-api-cache.

This module loads YAML configuration from `configs/default.yaml`, applies
environment variable overrides prefixed with `MARKOV_RL_`, and exposes the
configuration as a nested attribute-accessible object.

Usage:
    from src.utils.config import get_config
    config = get_config()
    print(config.rl.learning_rate)

Environment override convention:
- Use the prefix MARKOV_RL_.
- For nested keys use double underscores to separate levels, e.g.
  MARKOV_RL_RL__LEARNING_RATE=0.001

The loader attempts to preserve types from the YAML file when applying
overrides (ints/floats/bools/lists/dicts can be provided as JSON strings).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class AttrDict(dict):
    """Dict that supports attribute-style access for nested keys.

    Example: d = AttrDict({'a': {'b': 1}}); d.a.b == 1
    """

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - trivial
        self[name] = value


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level configuration must be a mapping (YAML dict)")
    return data


def _set_nested(d: Dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a nested value in dict `d` given a list of keys.

    Creates intermediate dicts if necessary.
    """
    cur = d
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _get_nested(d: Dict[str, Any], keys: list[str]) -> Optional[Any]:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _cast_value(original: Any, new_value_str: str) -> Any:
    """Cast the override string to the type of `original` when possible."""
    if original is None:
        # Try JSON parse, fall back to string
        try:
            return json.loads(new_value_str)
        except Exception:
            return new_value_str
    if isinstance(original, bool):
        lowered = new_value_str.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Cannot cast {new_value_str!r} to bool")
    if isinstance(original, int) and not isinstance(original, bool):
        return int(new_value_str)
    if isinstance(original, float):
        return float(new_value_str)
    if isinstance(original, (list, dict)):
        # Expect a JSON representation
        return json.loads(new_value_str)
    # Default fallback to original-like: try JSON, then raw string
    try:
        return json.loads(new_value_str)
    except Exception:
        return new_value_str


def _apply_env_overrides(cfg: Dict[str, Any], prefix: str = "MARKOV_RL_") -> None:
    """Apply environment variable overrides to `cfg` in-place.

    Nested keys are specified using double underscores. Example:
      MARKOV_RL_RL__LEARNING_RATE=0.001
    will override cfg['rl']['learning_rate'].
    """
    for name, val in os.environ.items():
        if not name.startswith(prefix):
            continue
        key_path = name[len(prefix) :]
        if not key_path:
            continue
        # Use double underscores as nested separator (common conv)
        if "__" in key_path:
            parts = [p.lower() for p in key_path.split("__")]
        else:
            # Fallback: treat single underscores as separators and lower-case
            parts = [p.lower() for p in key_path.split("_")]
        # Attempt to locate original to infer typing
        original = _get_nested(cfg, parts)
        try:
            casted = _cast_value(original, val) if original is not None else _cast_value(None, val)
        except Exception:
            # If casting fails, fall back to raw string
            casted = val
        _set_nested(cfg, parts, casted)


_CONFIG_LOCK = threading.Lock()
_CONFIG: Optional[AttrDict] = None


def get_config(yaml_path: Optional[str | Path] = None, env_prefix: str = "MARKOV_RL_") -> AttrDict:
    """Load and return the singleton configuration object.

    Parameters
    ----------
    yaml_path:
        Optional path to a YAML configuration file. If not provided, the
        function will look for `configs/default.yaml` in the project root.
    env_prefix:
        Environment variable prefix used to override configuration values.

    Returns
    -------
    AttrDict
        The configuration exposed as a nested attribute-accessible mapping.

    Example
    -------
    >>> cfg = get_config()
    >>> isinstance(cfg, AttrDict)
    True
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    with _CONFIG_LOCK:
        if _CONFIG is not None:
            return _CONFIG
        # Resolve path
        if yaml_path is None:
            # src/utils/config.py -> project root is parents[3]
            root = Path(__file__).resolve().parents[3]
            yaml_path = root / "configs" / "default.yaml"
        else:
            yaml_path = Path(yaml_path)
        cfg_dict = _load_yaml(yaml_path)
        # Apply environment overrides
        _apply_env_overrides(cfg_dict, prefix=env_prefix)
        # Wrap into AttrDict recursively
        def _wrap(obj: Any) -> Any:
            if isinstance(obj, dict):
                return AttrDict({k: _wrap(v) for k, v in obj.items()})
            if isinstance(obj, list):
                return [_wrap(v) for v in obj]
            return obj

        _CONFIG = _wrap(cfg_dict)
        return _CONFIG


__all__ = ["get_config"]

