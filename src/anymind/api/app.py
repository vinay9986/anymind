from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from anymind.runtime.logging import configure_logging
from anymind.runtime.jobs import Job, JobManager, JobStatus
from anymind.runtime.orchestrator import BudgetExceededError, Orchestrator, Session


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    agent: str = "chat_agent"


class ChatResponse(BaseModel):
    response: str
    usage: dict
    tokens: dict
    evidence: Optional[list[dict]] = None


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


def create_app() -> FastAPI:
    configure_logging("INFO")
    app = FastAPI(title="AnyMind API", version="0.1.0")

    orchestrator = Orchestrator()
    session_holder: dict[str, Session] = {}
    job_manager = JobManager()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    async def _get_session(agent_name: str) -> Session:
        if agent_name in session_holder:
            return session_holder[agent_name]
        session = await orchestrator.create_session(agent_name=agent_name)
        session_holder[agent_name] = session
        return session

    @app.post("/agents/run", response_model=ChatResponse)
    async def run_agent(payload: ChatRequest) -> ChatResponse:
        try:
            session = await _get_session(payload.agent)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            result = await orchestrator.run_turn(
                session, user_input=payload.message, thread_id=payload.thread_id
            )
        except BudgetExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return ChatResponse(**result)

    @app.post("/jobs", response_model=JobResponse)
    async def run_agent_async(payload: ChatRequest) -> JobResponse:
        try:
            session = await _get_session(payload.agent)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def _runner(job: Job):
            return await orchestrator.run_turn(
                session,
                user_input=payload.message,
                thread_id=payload.thread_id,
                pause_event=job.pause_event,
            )

        job = job_manager.start(_runner)
        return JobResponse(job_id=job.id, status=job.status.value)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        try:
            job = job_manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result = job.result if job.status == JobStatus.completed else None
        return JobStatusResponse(
            job_id=job.id,
            status=job.status.value,
            result=result,
            error=job.error,
        )

    @app.post("/jobs/{job_id}/pause", response_model=JobStatusResponse)
    async def pause_job(job_id: str) -> JobStatusResponse:
        try:
            job = job_manager.pause(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JobStatusResponse(job_id=job.id, status=job.status.value)

    @app.post("/jobs/{job_id}/resume", response_model=JobStatusResponse)
    async def resume_job(job_id: str) -> JobStatusResponse:
        try:
            job = job_manager.resume(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JobStatusResponse(job_id=job.id, status=job.status.value)

    @app.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
    async def cancel_job(job_id: str) -> JobStatusResponse:
        try:
            job = job_manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JobStatusResponse(job_id=job.id, status=job.status.value)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        for job in job_manager.list().values():
            job_manager.cancel(job.id)
        for session in session_holder.values():
            await session.close()

    return app
