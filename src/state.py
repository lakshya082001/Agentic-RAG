from typing import TypedDict

from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class ClinicalAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sub_queries: list[dict]   # [{"query": str, "type": "chat"|"rag"|"hybrid"}]
    missing_info: list[str]   # e.g. ["payer name", "LOB"]
    tool_results: dict        # {sub_query_text: result_text}
    final_response: str
    needs_dt: bool
    blocked: bool             # True if input_guard rejected the message
