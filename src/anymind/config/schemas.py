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


class AIoTConfig(BaseModel):
    min_iterations: int = 1
    self_consistency_samples: int = 5
    trace_steps: bool = True
    trace_max_chars: int = 300
    trace_samples: bool = True


class GIoTConfig(BaseModel):
    n_agents: int = 3
    self_consistency_samples: int = 1
    sim_start: float = 0.9
    sim_decay: float = 0.03
    sim_min: float = 0.8
    vote_ratio: float = 0.6
    trace_steps: bool = True
    trace_max_chars: int = 300


class AGoTConfig(BaseModel):
    d_max: int = 1
    l_max: int = 3
    n_max: int = 3
    max_concurrency: int = 3
    trace_steps: bool = True
    trace_max_chars: int = 300


class GoTConfig(BaseModel):
    max_layers: int = 3
    beam_width: int = 2
    children_per_expand: int = 3
    max_concurrency: int = 6

    reflection: bool = True
    reflection_weight: float = 0.5

    diversity_lambda: float = 0.1
    relevance_weight: float = 0.2

    prune_delta: float = 0.1
    success_threshold: float = 0.8

    verify: bool = True
    verify_threshold: float = 0.8
    verify_max_retries: int = 2

    max_diversity_samples: int = 20
    tool_feedback_max_chars: int = 8000
    context_max_chars: int = 1800
    final_top_k: int = 12


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
    aiot: Optional[AIoTConfig] = None
    giot: Optional[GIoTConfig] = None
    agot: Optional[AGoTConfig] = None
    got: Optional[GoTConfig] = None


class PricingConfig(BaseModel):
    currency: str = "USD"
    prices_per_1k_tokens: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    default: Dict[str, float] = Field(
        default_factory=lambda: {"input": 0.0, "output": 0.0}
    )


class MCPConfig(BaseModel):
    servers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
