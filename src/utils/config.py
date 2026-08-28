# src/utils/config.py

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml


def _deep_merge(base, override):
    """
    Recursively merge override into base.

    Nested dictionaries are merged recursively.
    Scalar/list values are replaced by override values.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_yaml(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    return data


def load_config(config_path, default_path=None):
    """
    Load default config and then override with config_path.

    If default_path is None, default.yaml is searched in the same
    directory as config_path.
    """
    config_path = Path(config_path)

    if default_path is None:
        default_path = config_path.parent / "default.yaml"

    default_config = load_yaml(default_path)
    user_config = load_yaml(config_path)

    merged = _deep_merge(default_config, user_config)

    return SimpleNamespace(**_dict_to_namespace(merged))


def _dict_to_namespace(value):
    if isinstance(value, dict):
        return {
            key: _dict_to_namespace(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [_dict_to_namespace(item) for item in value]

    return value


def namespace_to_dict(namespace):
    """
    Recursively convert SimpleNamespace back to a normal dictionary.
    Useful for wandb.config.
    """
    if isinstance(namespace, SimpleNamespace):
        return {
            key: namespace_to_dict(value)
            for key, value in vars(namespace).items()
        }

    if isinstance(namespace, list):
        return [namespace_to_dict(value) for value in namespace]

    return namespace