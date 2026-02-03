from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from anymind.config.schemas import MCPConfig, ModelConfig, PricingConfig

MODEL_CONFIG_ENV = "AM_MODEL_CONFIG"
PRICING_CONFIG_ENV = "AM_PRICING_CONFIG"
MCP_CONFIG_ENV = "AM_MCP_CONFIG"

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_CONFIG_CANDIDATES = [
    Path.cwd() / "model.json",
    Path.cwd() / "config" / "model.json",
    BASE_DIR / "config" / "model.json",
    Path.home() / ".config" / "anymind" / "model.json",
]

PRICING_CONFIG_CANDIDATES = [
    Path.cwd() / "pricing.json",
    Path.cwd() / "config" / "pricing.json",
    BASE_DIR / "config" / "pricing.json",
    Path.home() / ".config" / "anymind" / "pricing.json",
]

MCP_CONFIG_CANDIDATES = [
    Path.cwd() / "mcp_servers.json",
    Path.cwd() / "config" / "mcp_servers.json",
    BASE_DIR / "config" / "mcp_servers.json",
    Path.home() / ".config" / "anymind" / "mcp_servers.json",
]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_config(env_var: str, candidates: Iterable[Path]) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        path = Path(env_value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Config path from {env_var} not found: {path}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No config file found. Set {env_var} or place one at: "
        + ", ".join(str(path) for path in candidates)
    )


def load_model_config() -> ModelConfig:
    path = _find_config(MODEL_CONFIG_ENV, MODEL_CONFIG_CANDIDATES)
    data = _read_json(path)
    return ModelConfig(**data)


def load_model_config_from_path(path: Path) -> ModelConfig:
    data = _read_json(path)
    return ModelConfig(**data)


def load_pricing_config() -> PricingConfig:
    path = _find_config(PRICING_CONFIG_ENV, PRICING_CONFIG_CANDIDATES)
    data = _read_json(path)
    return PricingConfig(**data)


def load_mcp_config() -> MCPConfig:
    path = _find_config(MCP_CONFIG_ENV, MCP_CONFIG_CANDIDATES)
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            "MCP config must be a JSON object mapping server names to configs."
        )
    return MCPConfig(servers=data)
