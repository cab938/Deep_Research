FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
ENV THINKDEPTH_LOG_DIR=/tmp/thinkdepthai/logs \
    THINKDEPTH_LOG_LEVEL=INFO \
    THINKDEPTH_DB_URL=sqlite:////data/thinkdepthai/thinkdepthai.sqlite3

WORKDIR /srv/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${THINKDEPTH_LOG_DIR} /data/thinkdepthai

COPY pyproject.toml README.md ./ 
COPY src ./src
COPY serve.py ./serve.py

RUN python -m pip install --no-cache-dir fastapi uvicorn \
 && python -m pip install --no-cache-dir .

EXPOSE 8000

ENV THINKDEPTH_PORT=8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
