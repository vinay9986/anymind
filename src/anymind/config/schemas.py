from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    redis_url: Optional[str] = None
    ttl_seconds: int = 300


class CheckpointConfig(BaseModel):
    backend: str = "sqlite"
    path: Optional[Path] = None
    redis_url: Optional[str] = None


class ModelConfig(BaseModel):
    model: str
    model_provider: Optional[str] = None
    model_parameters: Dict[str, Any] = Field(default_factory=dict)
    thread_id: str = "default"
    tools_enabled: bool = True
    tools_policy: str = "auto"
    budget_tokens: Optional[int] = None
    state_dir: Optional[Path] = None
    checkpoint: Optional[CheckpointConfig] = None
    cache: Optional[CacheConfig] = None


class PricingConfig(BaseModel):
    currency: str = "USD"
    prices_per_1k_tokens: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    default: Dict[str, float] = Field(
        default_factory=lambda: {"input": 0.0, "output": 0.0}
    )


class MCPConfig(BaseModel):
    servers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
