import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from deep_research.research_agent_full import agent

TASK_DIR = Path(os.getenv("THINKDEPTH_TASK_DIR", "/tmp/thinkdepthai/tasks"))
TASK_DIR.mkdir(parents=True, exist_ok=True)

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


class TaskStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.tasks: Dict[str, TaskRecord] = {}
        self.lock = asyncio.Lock()
        self._load_existing()

    def _task_path(self, task_id: str) -> Path:
        return self.base_dir / f"{task_id}.json"

    def _persist(self, record: TaskRecord) -> None:
        path = self._task_path(record.task_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f)

    def _load_existing(self) -> None:
        if not self.base_dir.exists():
            return
        for path in self.base_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                record = TaskRecord(**data)
                # mark any in-flight tasks as failed because we cannot resume them after restart
                if record.status in ("pending", "running"):
                    record.status = "failed"
                    record.error = "Server restarted while task was in progress"
                self.tasks[record.task_id] = record
            except Exception:
                # skip corrupted entries
                continue

    async def create(self, request: ResearchRequest) -> TaskRecord:
        async with self.lock:
            task_id = uuid4().hex
            now = asyncio.get_event_loop().time()
            record = TaskRecord(
                task_id=task_id,
                status="pending",
                request=request,
                result=None,
                error=None,
                created_at=now,
                updated_at=now,
            )
            self.tasks[task_id] = record
            self._persist(record)
            return record

    async def update(self, task_id: str, status: TaskStatus, result: Optional[ResearchResponse] = None, error: Optional[str] = None) -> Optional[TaskRecord]:
        async with self.lock:
            record = self.tasks.get(task_id)
            if not record:
                return None
            record.status = status
            record.result = result
            record.error = error
            record.updated_at = asyncio.get_event_loop().time()
            self._persist(record)
            return record

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        async with self.lock:
            return self.tasks.get(task_id)


store = TaskStore(TASK_DIR)
app = FastAPI(title="ThinkDepth.ai Deep Research")


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
