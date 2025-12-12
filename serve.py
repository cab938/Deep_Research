import asyncio
import json
import os
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select

from deep_research.db import TaskRecordDB, init_db, session_scope, utcnow
from deep_research.research_agent_full import agent

TaskStatus = Literal["pending", "running", "succeeded", "failed"]


class ResearchRequest(BaseModel):
    query: str
    async_mode: bool = Field(default=False, description="Run asynchronously and poll later when true")


class ResearchResponse(BaseModel):
    task_id: Optional[str] = None
    status: TaskStatus = "succeeded"
    error: Optional[str] = None
    research_brief: str | None = None
    draft_report: str | None = None
    notes: List[str] = []
    final_report: str | None = None
    messages: List[str] = []


class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus
    request: ResearchRequest
    result: Optional[ResearchResponse] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


def _record_from_db(row: TaskRecordDB) -> TaskRecord:
    request_data = json.loads(row.request_json)
    result_data = json.loads(row.result_json) if row.result_json else None
    return TaskRecord(
        task_id=row.task_id,
        status=row.status,  # type: ignore[arg-type]
        request=ResearchRequest(**request_data),
        result=ResearchResponse(**result_data) if result_data else None,
        error=row.error,
        created_at=row.created_at.timestamp(),
        updated_at=row.updated_at.timestamp(),
    )


class TaskStore:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def create(self, request: ResearchRequest) -> TaskRecord:
        async with self.lock:
            task_id = uuid4().hex
            now = utcnow()
            db_row = TaskRecordDB(
                task_id=task_id,
                status="pending",
                request_json=request.model_dump_json(),
                result_json=None,
                error=None,
                pending_action_json=None,
                created_at=now,
                updated_at=now,
            )
            async with session_scope() as session:
                session.add(db_row)
                await session.commit()
            return _record_from_db(db_row)

    async def update(self, task_id: str, status: TaskStatus, result: Optional[ResearchResponse] = None, error: Optional[str] = None) -> Optional[TaskRecord]:
        async with self.lock:
            async with session_scope() as session:
                row = await session.get(TaskRecordDB, task_id)
                if not row:
                    return None
                row.status = status  # type: ignore[assignment]
                row.result_json = result.model_dump_json() if result else None
                row.error = error
                row.updated_at = utcnow()
                await session.commit()
                await session.refresh(row)
                return _record_from_db(row)

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        async with self.lock:
            async with session_scope() as session:
                result = await session.execute(select(TaskRecordDB).where(TaskRecordDB.task_id == task_id))
                row = result.scalar_one_or_none()
                if not row:
                    return None
                return _record_from_db(row)


store = TaskStore()
app = FastAPI(title="ThinkDepth.ai Deep Research")


@app.on_event("startup")
async def _startup() -> None:
    await init_db()


def _build_state(query: str) -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "supervisor_messages": [],
        "raw_notes": [],
        "notes": [],
        "draft_report": "",
        "final_report": "",
        "research_brief": "",
        "user_request": query,
    }


def _response_from_agent_result(result: Dict[str, Any], task_id: Optional[str], status: TaskStatus = "succeeded", error: Optional[str] = None) -> ResearchResponse:
    return ResearchResponse(
        task_id=task_id,
        status=status,
        error=error,
        research_brief=result.get("research_brief"),
        draft_report=result.get("draft_report"),
        notes=result.get("notes", []),
        final_report=result.get("final_report"),
        messages=[m.content if hasattr(m, "content") else str(m) for m in result.get("messages", [])],
    )


async def _run_task(task_id: str, request: ResearchRequest) -> None:
    await store.update(task_id, status="running")
    try:
        state = _build_state(request.query)
        result = await agent.ainvoke(state)
        response = _response_from_agent_result(result, task_id=task_id, status="succeeded")
        await store.update(task_id, status="succeeded", result=response)
    except Exception as exc:  # pragma: no cover
        await store.update(task_id, status="failed", error=str(exc))


@app.post("/research", response_model=ResearchResponse, status_code=200)
async def run_research(payload: ResearchRequest) -> ResearchResponse:
    if payload.async_mode:
        record = await store.create(payload)
        asyncio.create_task(_run_task(record.task_id, payload))
        return ResearchResponse(task_id=record.task_id, status="pending")

    try:
        state = _build_state(payload.query)
        result = await agent.ainvoke(state)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))

    return _response_from_agent_result(result, task_id=None, status="succeeded")


@app.get("/research/{task_id}", response_model=ResearchResponse)
async def get_research(task_id: str) -> ResearchResponse:
    record = await store.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    if record.result:
        return record.result

    return ResearchResponse(
        task_id=record.task_id,
        status=record.status,
        error=record.error,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.getenv("THINKDEPTH_PORT", "8000")),
        reload=False,
    )
