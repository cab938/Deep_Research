

"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

import os

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

from deep_research.state_research import Summary
from deep_research.prompts import summarize_webpage_prompt, report_generation_with_draft_insight_prompt
from deep_research.logging_setup import get_logger

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

# ===== CONFIGURATION =====

MODEL_ID = os.getenv("DEEP_RESEARCH_MODEL", "openai:gpt-5")
SUMMARY_MODEL_ID = os.getenv("DEEP_RESEARCH_SUMMARY_MODEL", MODEL_ID)
WRITER_MODEL_ID = os.getenv("DEEP_RESEARCH_WRITER_MODEL", MODEL_ID)
WRITER_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_WRITER_MAX_TOKENS", "32000"))

summarization_model = init_chat_model(model=SUMMARY_MODEL_ID)
writer_model = init_chat_model(model=WRITER_MODEL_ID, max_tokens=WRITER_MAX_TOKENS)
tavily_client = TavilyClient()
DEFAULT_TAVILY_RAW_CONTENT_MAX_CHARS = 250000
logger = get_logger(__name__)

# ===== SEARCH FUNCTIONS =====

def _get_env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None

def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return None

def _get_env_bool(name: str) -> bool | None:
    value = _get_env_str(name)
    if value is None:
        return None
    parsed = _parse_bool(value)
    if parsed is None:
        logger.warning("Invalid bool env var", extra={"name": name, "value": value})
    return parsed

def _get_env_int(name: str) -> int | None:
    value = _get_env_str(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int env var", extra={"name": name, "value": value})
        return None

def _get_env_float(name: str) -> float | None:
    value = _get_env_str(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env var", extra={"name": name, "value": value})
        return None

def _get_env_csv(name: str) -> list[str] | None:
    value = _get_env_str(name)
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    items = [part for part in parts if part]
    return items or None

def _get_env_tavily_search_depth() -> str | None:
    value = _get_env_str("TAVILY_SEARCH_DEPTH")
    if value is None:
        return None
    lowered = value.lower()
    allowed = {"basic", "advanced", "fast", "ultra-fast"}
    if lowered in allowed:
        return lowered
    logger.warning("Invalid Tavily search_depth", extra={"value": value})
    return None

def _get_env_tavily_topic() -> str | None:
    value = _get_env_str("TAVILY_TOPIC")
    if value is None:
        return None
    lowered = value.lower()
    allowed = {"general", "news", "finance"}
    if lowered in allowed:
        return lowered
    logger.warning("Invalid Tavily topic", extra={"value": value})
    return None

def _get_env_tavily_time_range() -> str | None:
    value = _get_env_str("TAVILY_TIME_RANGE")
    if value is None:
        return None
    lowered = value.lower()
    allowed = {"day", "week", "month", "year"}
    if lowered in allowed:
        return lowered
    logger.warning("Invalid Tavily time_range", extra={"value": value})
    return None

def _get_env_tavily_include_answer() -> bool | str | None:
    value = _get_env_str("TAVILY_INCLUDE_ANSWER")
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"basic", "advanced"}:
        return lowered
    parsed_bool = _parse_bool(value)
    if parsed_bool is not None:
        return parsed_bool
    logger.warning("Invalid Tavily include_answer", extra={"value": value})
    return None

def _get_env_tavily_include_raw_content() -> bool | str | None:
    value = _get_env_str("TAVILY_INCLUDE_RAW_CONTENT")
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"markdown", "text"}:
        return lowered
    parsed_bool = _parse_bool(value)
    if parsed_bool is not None:
        return parsed_bool
    logger.warning("Invalid Tavily include_raw_content", extra={"value": value})
    return None

def _build_tavily_search_kwargs(*, max_results: int, topic: str, include_raw_content: bool | str) -> dict[str, object]:
    env_search_depth = _get_env_tavily_search_depth()
    env_topic = _get_env_tavily_topic()
    env_time_range = _get_env_tavily_time_range()
    env_start_date = _get_env_str("TAVILY_START_DATE")
    env_end_date = _get_env_str("TAVILY_END_DATE")
    env_days = _get_env_int("TAVILY_DAYS")
    env_max_results = _get_env_int("TAVILY_MAX_RESULTS")
    env_include_domains = _get_env_csv("TAVILY_INCLUDE_DOMAINS")
    env_exclude_domains = _get_env_csv("TAVILY_EXCLUDE_DOMAINS")
    env_include_answer = _get_env_tavily_include_answer()
    env_include_raw_content = _get_env_tavily_include_raw_content()
    env_timeout = _get_env_float("TAVILY_TIMEOUT_SECONDS")
    env_country = _get_env_str("TAVILY_COUNTRY")
    env_auto_parameters = _get_env_bool("TAVILY_AUTO_PARAMETERS")

    resolved_topic = env_topic or topic
    resolved_max_results = env_max_results if env_max_results is not None else max_results
    resolved_include_raw_content = env_include_raw_content if env_include_raw_content is not None else include_raw_content

    kwargs: dict[str, object] = {
        "topic": resolved_topic,
        "max_results": resolved_max_results,
        "include_raw_content": resolved_include_raw_content,
    }

    if env_search_depth is not None:
        kwargs["search_depth"] = env_search_depth
    if env_time_range is not None:
        kwargs["time_range"] = env_time_range
    if env_start_date is not None:
        kwargs["start_date"] = env_start_date
    if env_end_date is not None:
        kwargs["end_date"] = env_end_date
    if env_days is not None:
        kwargs["days"] = env_days
    if env_include_domains is not None:
        kwargs["include_domains"] = env_include_domains
    if env_exclude_domains is not None:
        kwargs["exclude_domains"] = env_exclude_domains
    if env_include_answer is not None:
        kwargs["include_answer"] = env_include_answer
    if env_timeout is not None:
        kwargs["timeout"] = env_timeout
    if env_country is not None:
        kwargs["country"] = env_country
    if env_auto_parameters is not None:
        kwargs["auto_parameters"] = env_auto_parameters

    return kwargs

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool | Literal["markdown", "text"] = True,
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """

    # Execute searches sequentially. Note: yon can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    kwargs = _build_tavily_search_kwargs(
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )
    logger.info(
        "Executing Tavily search batch",
        extra={
            "query_count": len(search_queries),
            "search_depth": kwargs.get("search_depth"),
            "topic": kwargs.get("topic"),
            "time_range": kwargs.get("time_range"),
            "days": kwargs.get("days"),
            "max_results": kwargs.get("max_results"),
            "include_domains": kwargs.get("include_domains"),
            "exclude_domains": kwargs.get("exclude_domains"),
            "include_answer": kwargs.get("include_answer"),
            "include_raw_content": kwargs.get("include_raw_content"),
            "timeout": kwargs.get("timeout"),
            "country": kwargs.get("country"),
            "auto_parameters": kwargs.get("auto_parameters"),
        },
    )
    for query in search_queries:
        try:
            result = tavily_client.search(query, **kwargs)
        except Exception:
            logger.exception(
                "Tavily query failed",
                extra={"query": query},
            )
            raise
        logger.info(
            "Tavily query complete",
            extra={
                "query": query,
                "result_count": len(result.get("results", [])),
            },
        )
        search_docs.append(result)

    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)
        logger.info(
            "Summarizing webpage content",
            extra={"content_length": len(webpage_content)},
        )

        # Generate summary
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])

        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        logger.info(
            "Webpage summarization complete",
            extra={
                "summary_len": len(summary.summary or ""),
                "excerpt_len": len(summary.key_excerpts or ""),
            },
        )

        return formatted_summary

    except Exception as e:
        logger.exception("Failed to summarize webpage content")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    logger.info(
        "Search results deduplicated",
        extra={
            "input_count": sum(len(r.get("results", [])) for r in search_results),
            "unique_count": len(unique_results),
        },
    )
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}
    logger.info(
        "Processing search results for summarization",
        extra={"unique_count": len(unique_results)},
    )

    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            raw_content_max_chars = _get_env_int("TAVILY_RAW_CONTENT_MAX_CHARS")
            if raw_content_max_chars is None:
                raw_content_max_chars = DEFAULT_TAVILY_RAW_CONTENT_MAX_CHARS
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'][:raw_content_max_chars])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== RESEARCH TOOLS =====

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    # Execute search for single query
    logger.info(
        "Tavily search started",
        extra={
            "query": query,
            "max_results": max_results,
            "topic": topic,
        },
    )
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

@tool(parse_docstring=True)
def refine_draft_report(research_brief: Annotated[str, InjectedToolArg], 
                        findings: Annotated[str, InjectedToolArg], 
                        draft_report: Annotated[str, InjectedToolArg]):
    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: user's research request
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request

    Returns:
        refined draft report
    """

    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str()
    )

    logger.info(
        "Refining draft report with findings",
        extra={
            "research_brief_len": len(research_brief or ""),
            "findings_len": len(findings or ""),
            "draft_report_len": len(draft_report or ""),
        },
    )
    try:
        draft_report = writer_model.invoke([HumanMessage(content=draft_report_prompt)])
    except Exception:
        logger.exception("Refine draft report failed")
        raise

    logger.info(
        "Refined draft report complete",
        extra={"draft_report_len": len(draft_report.content or "")},
    )

    return draft_report.content
