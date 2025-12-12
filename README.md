### ThinkDepth.ai Deep Research
Our ThinkDepth.ai deep research 1) addresses the issue of balancing multiple factors for long horizon and complex tasks and 2) addresses the issue of balancing between model capability and structural flexibility. It solves those issues by explicitly reasoning about the self-balancing rules that guide the interaction of different requirements at different stages. For example, those self-balancing rules allow ThinkDepth.ai deep research to explicitly guide the interaction between information gap closing and generation gap closing at different stages.


In the information collection stage, it focuses on closing the information gap by making external web search tool calls while doing a bit of generation gap closing by refining the draft report. Once the information gap is fully closed, it transitions to the final report generation stage. In the final report generation stage, it then fully optimizes for closing the generation gap. This explicit multi-stage self-balancing rules reasoning leads to the development of Self-Balancing Test-Time Diffusion Deep Research algorithm and more effective context engineering. We call this paradigm Self-Balancing Agentic AI. 

Check out our <a href="https://paichunlin.substack.com/p/self-balancing-agentic-ai-test-time">blog post</a> for more technical details.

Primary Contact: <a href="https://www.linkedin.com/in/paichunjimlin">Paichun Lin's LinkedIn</a> | paichul@cs.stanford.edu

### Setup
Please follow the instructions to run the demo:
1. Install uv
```
pip install uv
```
2. in ~/.zshrc or ~/.bashrc, enter your API keys info:
```
export OPENAI_API_KEY='Your OpenAI API Key'

export TAVILY_API_KEY='Your Tavily API Key'
```
3. Install all packages
```
uv sync
```
4. Run the demo in the notebook
```
uv run jupyter notebook thinkdepthai_deepresearch.ipynb
```

### Async API usage
- Default behavior: `POST /research` runs synchronously and returns a `ResearchResponse`.
- Async mode: set `"async_mode": true` in the request body to get back a `task_id` and `status: pending`. Poll `GET /research/{task_id}` to retrieve status and the completed response when ready. Task files are stored under `/tmp/thinkdepthai/tasks` by default (override with `THINKDEPTH_TASK_DIR`).
- Example:
  - Submit: `curl -X POST http://localhost:8005/research -H "Content-Type: application/json" -d '{"query":"...", "async_mode":true}'`
  - Poll: `curl http://localhost:8005/research/<task_id>`

### Model configuration
- The agent uses OpenAI-compatible chat endpoints. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` to point at your self-hosted gateway (e.g., vLLM, llama.cpp server).
- Choose models via `DEEP_RESEARCH_MODEL` (defaults to `openai:gpt-5`). Optional overrides: `DEEP_RESEARCH_SUMMARY_MODEL`, `DEEP_RESEARCH_COMPRESS_MODEL`, and `DEEP_RESEARCH_WRITER_MODEL` (defaults fall back to `DEEP_RESEARCH_MODEL`). `DEEP_RESEARCH_WRITER_MAX_TOKENS` and `DEEP_RESEARCH_COMPRESS_MAX_TOKENS` control output lengths.

### Logging
- Plain text logs write to `/tmp/thinkdepthai/logs/thinkdepthai.log` by default (configure with `THINKDEPTH_LOG_DIR` and `THINKDEPTH_LOG_LEVEL`).
- Logs rotate at midnight with one backup kept for roughly one day of retention.
- Tail inside the container with `docker exec -it <container_name> tail -f /tmp/thinkdepthai/logs/thinkdepthai.log`.

### Environment variables

Set these before starting the service to control model selection, paths, and networking:

| Variable | Example | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `sk-your-key` | API key for the OpenAI-compatible endpoint. Required for model calls. |
| `OPENAI_BASE_URL` | `http://localhost:1337/v1` | Base URL for the OpenAI-compatible endpoint (vLLM, llama.cpp, Azure). Defaults to the official API if unset. |
| `TAVILY_API_KEY` | `tvly-dev-abc123` | API key for Tavily search, used by the research agents. |
| `DEEP_RESEARCH_MODEL` | `openai:gpt-5` | Default chat model used across the pipeline. |
| `DEEP_RESEARCH_SUMMARY_MODEL` | `openai:gpt-5` | Model for summarization steps; falls back to `DEEP_RESEARCH_MODEL` if unset. |
| `DEEP_RESEARCH_COMPRESS_MODEL` | `openai:gpt-5` | Model for compressing research notes; falls back to `DEEP_RESEARCH_MODEL` if unset. |
| `DEEP_RESEARCH_CREATIVE_MODEL` | `openai:gpt-5` | Creative model for draft generation; falls back to `DEEP_RESEARCH_MODEL` if unset. |
| `DEEP_RESEARCH_WRITER_MODEL` | `openai:gpt-5` | Model for final report writing; falls back to `DEEP_RESEARCH_MODEL` if unset. |
| `DEEP_RESEARCH_COMPRESS_MAX_TOKENS` | `32000` | Max tokens for compressed research responses. |
| `DEEP_RESEARCH_WRITER_MAX_TOKENS` | `40000` | Max tokens for the final report writer. |
| `DEEP_RESEARCH_MAX_ITERATIONS` | `15` | Maximum supervisor tool-call iterations before ending a run. |
| `DEEP_RESEARCH_MAX_CONCURRENCY` | `3` | Maximum concurrent researcher agents launched per iteration. |
| `THINKDEPTH_PORT` | `8000` | Port exposed by the FastAPI service inside the container. |
| `THINKDEPTH_TASK_DIR` | `/tmp/thinkdepthai/tasks` | Directory for async task metadata used by `/research` in async mode. |
| `THINKDEPTH_LOG_DIR` | `/tmp/thinkdepthai/logs` | Directory for log files. |
| `THINKDEPTH_LOG_LEVEL` | `INFO` | Log level for all ThinkDepth.ai modules (`DEBUG`, `INFO`, `WARNING`, etc.). |
| `THINKDEPTH_RUN_ID` | `demo1234` | Optional identifier appended to log entries for correlation; defaults to a random ID. |

### Experiments
<a href="https://thinkdepth.ai">ThinkDepth.ai</a> deep research is ranked #1 and established a new state-of-art result on <a href="https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard/discussions/4/files">DeepResearch  Bench</a> on Nov 17th, 2025.
* It outperformed Google Gemini 2.5 pro deep research by 2.78%.
* It outperformed OpenAI deep research by 6.04%.
* It outperformed Anthropic Claude deep research by  7.45%.

<img width="899" height="463" alt="benchmark" src="https://github.com/user-attachments/assets/1ddd8bd0-1d04-467e-a00d-394e9dc967f8" />

### DeepResearch Bench Leaderboard Screenshot

<img width="1178" height="751" alt="huggingface_leaderboard" src="https://github.com/user-attachments/assets/2d88256a-5e77-46f8-bd51-fe083bfcc780" />

<a href="https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard"> DeepResearch Bench Leaderboard </a> 

### Example Generated Report
For the task "Write a paper to discuss the influence of AI interaction on interpersonal relations, considering AI's potential to fundamentally change how and why individuals relate to each other.", a snapshot of the generated report is shared below:


<img width="1005" height="645" alt="report" src="https://github.com/user-attachments/assets/7fccc245-a83b-4b95-9abe-f1d56fef607d" />
