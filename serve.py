import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select

from deep_research.db import TaskRecordDB, init_db, session_scope, utcnow
from deep_research.logging_setup import get_logger
from deep_research.research_agent_full import agent

TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

logger = get_logger(__name__)
MAX_QUERY_LOG_CHARS = 200
MAX_TITLE_WORDS = 7
TASK_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _summarize_query(query: str) -> dict[str, object]:
    query = query or ""
    preview = query[:MAX_QUERY_LOG_CHARS]
    return {
        "query_len": len(query),
        "query_preview": preview,
        "query_truncated": len(query) > MAX_QUERY_LOG_CHARS,
    }


def _format_task_date(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _normalize_title_tokens(query: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (query or "").lower())
    tokens = [token for token in normalized.split() if token and token not in TASK_TITLE_STOPWORDS]
    if not tokens:
        return ["research"]
    return tokens


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages or []:
        if hasattr(message, "type") and hasattr(message, "model_dump"):
            serialized.append({"type": message.type, "data": message.model_dump()})
        else:
            serialized.append({"type": "text", "data": {"content": str(message)}})
    return serialized


def _langgraph_recursion_limit() -> int:
    explicit = os.getenv("LANGGRAPH_RECURSION_LIMIT")
    if explicit:
        try:
            return max(1, int(explicit))
        except ValueError:
            logger.warning("Invalid LANGGRAPH_RECURSION_LIMIT, using fallback", extra={"value": explicit})

    max_iterations = int(os.getenv("DEEP_RESEARCH_MAX_ITERATIONS", "15"))
    # The supervisor subgraph typically consumes ~2 steps per iteration (plan + tools),
    # plus a handful of outer graph steps. Keep some headroom.
    return max(25, (2 * max_iterations) + 20)


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


class ResearchTaskSummary(BaseModel):
    task_id: str
    status: TaskStatus
    query: str
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
        self.running_tasks: Dict[str, asyncio.Task] = {}

    async def _task_id_exists(self, task_id: str) -> bool:
        async with session_scope() as session:
            row = await session.get(TaskRecordDB, task_id)
            return row is not None

    async def _build_task_id(self, query: str, now: datetime) -> str:
        date_str = _format_task_date(now)
        base_tokens = _normalize_title_tokens(query)[:MAX_TITLE_WORDS]
        title_tokens = base_tokens
        candidate = f"{date_str}_{'_'.join(title_tokens)}"
        suffix = 2

        while await self._task_id_exists(candidate):
            max_tokens = MAX_TITLE_WORDS - 1 if len(base_tokens) >= MAX_TITLE_WORDS else len(base_tokens)
            title_tokens = base_tokens[:max_tokens] + [str(suffix)]
            candidate = f"{date_str}_{'_'.join(title_tokens)}"
            suffix += 1

        return candidate

    async def create(self, request: ResearchRequest) -> TaskRecord:
        async with self.lock:
            now = utcnow()
            task_id = await self._build_task_id(request.query, now)
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
            logger.info(
                "Task id assigned",
                extra={
                    "task_id": task_id,
                    "task_date": _format_task_date(now),
                },
            )
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

    async def list(self, status: TaskStatus | None = None) -> List[TaskRecord]:
        async with self.lock:
            async with session_scope() as session:
                stmt = select(TaskRecordDB).order_by(TaskRecordDB.created_at.desc())
                if status is not None:
                    stmt = stmt.where(TaskRecordDB.status == status)
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [_record_from_db(row) for row in rows]

    async def attach_task_handle(self, task_id: str, task: asyncio.Task) -> None:
        async with self.lock:
            self.running_tasks[task_id] = task

    async def pop_task_handle(self, task_id: str) -> Optional[asyncio.Task]:
        async with self.lock:
            return self.running_tasks.pop(task_id, None)

    async def cancel_task(self, task_id: str) -> Optional[TaskRecord]:
        async with self.lock:
            task = self.running_tasks.get(task_id)
            if task:
                task.cancel()
            async with session_scope() as session:
                row = await session.get(TaskRecordDB, task_id)
                if not row:
                    return None
                row.status = "cancelled"  # type: ignore[assignment]
                row.error = "cancelled by user request"
                row.result_json = None
                row.updated_at = utcnow()
                await session.commit()
                await session.refresh(row)
                return _record_from_db(row)


store = TaskStore()
app = FastAPI(title="ThinkDepth.ai Deep Research")


@app.on_event("startup")
async def _startup() -> None:
    await init_db()


def _build_state(query: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "supervisor_messages": [],
        "raw_notes": [],
        "notes": [],
        "draft_report": "",
        "final_report": "",
        "research_brief": "",
        "user_request": query,
        "task_id": task_id,
        "research_iterations": 0,
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


def _task_artifact_root() -> Path:
    root = os.getenv("THINKDEPTH_TASK_DIR", "/tmp/thinkdepthai/tasks")
    return Path(root)


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _ensure_trailing_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _export_task_artifacts(
    task_id: str,
    request: ResearchRequest,
    response: ResearchResponse,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    root = _task_artifact_root()
    task_dir = root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    _write_text_file(task_dir / "query.md", _ensure_trailing_newline(request.query))
    _write_json_file(task_dir / "request.json", request.model_dump())
    _write_json_file(task_dir / "response.json", response.model_dump())

    if response.research_brief:
        _write_text_file(task_dir / "research_brief.md", _ensure_trailing_newline(response.research_brief))
    if response.draft_report:
        _write_text_file(task_dir / "draft_report.md", _ensure_trailing_newline(response.draft_report))
    if response.final_report:
        _write_text_file(task_dir / "final_report.md", _ensure_trailing_newline(response.final_report))

    notes_dir = task_dir / "notes"
    for idx, note in enumerate(response.notes or []):
        _write_text_file(notes_dir / f"note{idx}.md", _ensure_trailing_newline(note or ""))

    messages_dir = task_dir / "messages"
    for idx, message in enumerate(response.messages or []):
        _write_text_file(messages_dir / f"message{idx}.md", _ensure_trailing_newline(message or ""))

    if state:
        raw_notes_dir = task_dir / "raw_notes"
        for idx, note in enumerate(state.get("raw_notes", []) or []):
            _write_text_file(raw_notes_dir / f"note{idx}.md", _ensure_trailing_newline(note or ""))

        supervisor_messages = _serialize_messages(list(state.get("supervisor_messages", []) or []))
        _write_json_file(task_dir / "supervisor_messages.json", supervisor_messages)

        state_snapshot = {
            "task_id": task_id,
            "research_iterations": state.get("research_iterations"),
            "research_brief": state.get("research_brief"),
            "draft_report": state.get("draft_report"),
            "final_report": state.get("final_report"),
            "notes": state.get("notes", []),
            "raw_notes": state.get("raw_notes", []),
            "supervisor_messages": supervisor_messages,
            "messages": _serialize_messages(list(state.get("messages", []) or [])),
        }
        _write_json_file(task_dir / "state.json", state_snapshot)


async def _run_task(task_id: str, request: ResearchRequest) -> None:
    task_logger = get_logger(__name__, task_id=task_id)
    started_at = time.monotonic()
    task_logger.info(
        "Task started",
        extra=_summarize_query(request.query),
    )
    await store.update(task_id, status="running")
    try:
        state = _build_state(request.query, task_id=task_id)
        result = await agent.ainvoke(state, config={"recursion_limit": _langgraph_recursion_limit()})
        response = _response_from_agent_result(result, task_id=task_id, status="succeeded")
        await store.update(task_id, status="succeeded", result=response)
        task_logger.info(
            "Task succeeded",
            extra={
                "elapsed_s": round(time.monotonic() - started_at, 2),
                "final_report_len": len(response.final_report or ""),
            },
        )
        try:
            await asyncio.to_thread(_export_task_artifacts, task_id, request, response, result)
        except Exception:
            task_logger.exception("Failed to export task artifacts")
    except asyncio.CancelledError:
        await store.update(task_id, status="cancelled", error="cancelled by user request")
        task_logger.info(
            "Task cancelled",
            extra={"elapsed_s": round(time.monotonic() - started_at, 2)},
        )
        raise
    except Exception as exc:  # pragma: no cover
        error_text = f"{type(exc).__name__}: {exc}"
        response = ResearchResponse(task_id=task_id, status="failed", error=error_text)
        await store.update(task_id, status="failed", result=response, error=error_text)
        task_logger.exception(
            "Task failed",
            extra={"elapsed_s": round(time.monotonic() - started_at, 2)},
        )
        try:
            await asyncio.to_thread(_export_task_artifacts, task_id, request, response, None)
        except Exception:
            task_logger.exception("Failed to export task artifacts after failure")
    finally:
        await store.pop_task_handle(task_id)


@app.post("/research", response_model=ResearchResponse, status_code=200)
async def run_research(payload: ResearchRequest) -> ResearchResponse:
    logger.info(
        "Research request received",
        extra={
            "async_mode": payload.async_mode,
            **_summarize_query(payload.query),
        },
    )
    if payload.async_mode:
        record = await store.create(payload)
        logger.info(
            "Async task created",
            extra={"task_id": record.task_id},
        )
        task = asyncio.create_task(_run_task(record.task_id, payload))
        await store.attach_task_handle(record.task_id, task)
        return ResearchResponse(task_id=record.task_id, status="pending")

    try:
        state = _build_state(payload.query)
        result = await agent.ainvoke(state, config={"recursion_limit": _langgraph_recursion_limit()})
    except Exception as exc:  # pragma: no cover
        logger.exception("Sync research request failed")
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


@app.get("/research", response_model=List[ResearchTaskSummary])
async def list_research(status: TaskStatus | None = None) -> List[ResearchTaskSummary]:
    records = await store.list(status=status)
    return [
        ResearchTaskSummary(
            task_id=record.task_id,
            status=record.status,
            query=record.request.query,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        for record in records
    ]


@app.post("/research/{task_id}/cancel", response_model=ResearchResponse)
async def cancel_research(task_id: str) -> ResearchResponse:
    record = await store.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    if record.status in {"succeeded", "failed", "cancelled"}:
        return record.result or ResearchResponse(task_id=task_id, status=record.status, error=record.error)

    if record.status == "pending":
        cancelled = await store.cancel_task(task_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail="Task not found")
        return ResearchResponse(task_id=task_id, status="cancelled", error="cancelled by user request")

    # running
    await store.cancel_task(task_id)
    handle = await store.pop_task_handle(task_id)
    if handle:
        try:
            await asyncio.wait_for(handle, timeout=5)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass
    return ResearchResponse(task_id=task_id, status="cancelled", error="cancelled by user request")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.getenv("THINKDEPTH_PORT", "8000")),
        reload=False,
    )
