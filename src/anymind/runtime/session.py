from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict

from anymind.config.schemas import ModelConfig
from anymind.runtime.evidence import EvidenceLedger
from anymind.runtime.usage import PricingTable, UsageTotals


@dataclass
class Session:
    agent_name: str
    model_config: ModelConfig
    pricing: PricingTable
    tool_policy_name: str
    model_client: Any
    tools: list[Any]
    agent_with_tools: Any
    agent_no_tools: Any
    checkpointer: Any
    checkpointer_conn: Any
    totals_by_model: Dict[str, UsageTotals] = field(default_factory=dict)
    agent_cache: Dict[tuple[str, ...], Any] = field(default_factory=dict)
    budget_exhausted: bool = False
    evidence_ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    chat_history: list[tuple[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def close(self) -> None:
        if self.checkpointer_conn is not None:
            await self.checkpointer_conn.close()
