
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str
from deep_research.prompts import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from deep_research.state_scope import AgentState, AgentInputState
from deep_research.research_agent_scope import clarify_with_user, write_research_brief, write_draft_report
from deep_research.multi_agent_supervisor import supervisor_agent
from deep_research.logging_setup import get_logger

# ===== Config =====

from langchain.chat_models import init_chat_model
import os

WRITER_MODEL_ID = os.getenv("DEEP_RESEARCH_WRITER_MODEL", os.getenv("DEEP_RESEARCH_MODEL", "openai:gpt-5"))
WRITER_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_WRITER_MAX_TOKENS", "40000"))
writer_model = init_chat_model(model=WRITER_MODEL_ID, max_tokens=WRITER_MAX_TOKENS)
logger = get_logger(__name__)

# ===== FINAL REPORT GENERATION =====

from deep_research.state_scope import AgentState

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    logger.info(
        "Final report generation started",
        extra={
            "research_brief_len": len(state.get("research_brief", "") or ""),
            "findings_count": len(notes),
            "draft_report_len": len(state.get("draft_report", "") or ""),
        },
    )

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=state.get("user_request", "")
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    logger.info(
        "Final report generation complete",
        extra={
            "final_report_len": len(final_report.content or ""),
            "notes_used": len(notes),
        },
    )

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
