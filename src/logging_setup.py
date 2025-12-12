"""Logging setup for ThinkDepth.ai agents."""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_LOG_DIR = os.getenv("THINKDEPTH_LOG_DIR", "/tmp/thinkdepthai/logs")
RUN_ID = os.getenv("THINKDEPTH_RUN_ID", uuid4().hex[:8])


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that appends contextual key value pairs to the message."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        if merged:
            context_str = " ".join(f"{k}={v}" for k, v in merged.items())
            msg = f"{msg} [{context_str}]"
        return msg, kwargs


def _configure_base_logger() -> logging.Logger:
    """Configure the shared thinkdepthai logger with file rotation and stdout."""
    logger = logging.getLogger("thinkdepthai")
    if logger.handlers:
        return logger

    level_name = os.getenv("THINKDEPTH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    log_dir = Path(DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "thinkdepthai.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=1,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a logger adapter with optional contextual fields."""
    base_logger = _configure_base_logger()
    logger = logging.getLogger(name)

    if not logger.handlers:
        for handler in base_logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(base_logger.level)
        logger.propagate = False

    context_with_run = {"run_id": RUN_ID}
    context_with_run.update(context)
    return ContextAdapter(logger, context_with_run)
