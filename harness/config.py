"""Config: harness.toml if a TOML parser exists, env vars otherwise (env wins)."""
import json
import os
from typing import Any, Dict

try:
    import tomllib  # py3.11+
except ImportError:  # py3.9/3.10 fallback for local dev; containers are py3.12
    tomllib = None

from .models import DEFAULT_POLICY

DEFAULTS: Dict[str, Any] = {
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5:7b",
    "core_url": "http://core:8000",
    "sandbox_url": "http://sandbox:8001",
    "workspace": "/workspace",
    "data_dir": "/data",
    "policy": DEFAULT_POLICY,
}


def load(path: str = "harness.toml") -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    cfg["policy"] = dict(DEFAULT_POLICY)
    if tomllib and os.path.exists(path):
        with open(path, "rb") as f:
            file_cfg = tomllib.load(f)
        cfg["policy"].update(file_cfg.pop("policy", {}))
        cfg.update(file_cfg)
    env_json = os.environ.get("HARNESS_CONFIG")
    if env_json:
        cfg.update(json.loads(env_json))
    for key in ("ollama_url", "model", "core_url", "sandbox_url"):
        env = os.environ.get(key.upper())
        if env:
            cfg[key] = env
    return cfg
