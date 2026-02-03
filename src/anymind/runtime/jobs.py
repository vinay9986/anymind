from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Any] = None
    error: Optional[str] = None
    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None

    def touch(self) -> None:
        self.updated_at = time.time()

    async def wait_if_paused(self) -> None:
        await self.pause_event.wait()


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    def get(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found")
        return self._jobs[job_id]

    def list(self) -> Dict[str, Job]:
        return dict(self._jobs)

    def start(self, runner: Callable[[Job], Awaitable[Any]]) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, status=JobStatus.queued)
        job.pause_event.set()
        self._jobs[job_id] = job

        async def _wrap() -> None:
            job.status = JobStatus.running
            job.touch()
            try:
                await job.wait_if_paused()
                job.result = await runner(job)
                job.status = JobStatus.completed
            except asyncio.CancelledError:
                job.status = JobStatus.canceled
                raise
            except Exception as exc:  # pragma: no cover - defensive
                job.error = str(exc)
                job.status = JobStatus.failed
            finally:
                job.touch()

        job.task = asyncio.create_task(_wrap())
        return job

    def pause(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in {JobStatus.completed, JobStatus.failed, JobStatus.canceled}:
            return job
        job.pause_event.clear()
        job.status = JobStatus.paused
        job.touch()
        return job

    def resume(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == JobStatus.paused:
            job.pause_event.set()
            job.status = JobStatus.running
            job.touch()
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.task and not job.task.done():
            job.task.cancel()
        job.status = JobStatus.canceled
        job.touch()
        return job
