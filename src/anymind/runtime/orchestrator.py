from __future__ import annotations

from typing import Any, Dict, Optional

from anymind.config.schemas import MCPConfig, ModelConfig, PricingConfig
from anymind.runtime.session import Session
from anymind.runtime.session_factory import SessionFactory
from anymind.runtime.session_usage import session_summary
from anymind.runtime.turn_runner import BudgetExceededError, TurnRunner


class Orchestrator:
    def __init__(
        self,
        *,
        session_factory: Optional[SessionFactory] = None,
        turn_runner: Optional[TurnRunner] = None,
    ) -> None:
        self._session_factory = session_factory or SessionFactory()
        self._turn_runner = turn_runner or TurnRunner()

    async def create_session(
        self,
        *,
        agent_name: str,
        model_config: Optional[ModelConfig] = None,
        pricing_config: Optional[PricingConfig] = None,
        mcp_config: Optional[MCPConfig] = None,
    ) -> Session:
        return await self._session_factory.create_session(
            agent_name=agent_name,
            model_config=model_config,
            pricing_config=pricing_config,
            mcp_config=mcp_config,
        )

    async def run_turn(
        self,
        session: Session,
        *,
        user_input: str,
        thread_id: Optional[str] = None,
        pause_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return await self._turn_runner.run_turn(
            session,
            user_input=user_input,
            thread_id=thread_id,
            pause_event=pause_event,
        )

    def session_summary(self, session: Session) -> Dict[str, Any]:
        return session_summary(session)
