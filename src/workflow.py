from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.state import ClinicalAgentState
from src.pipeline import rag_pipeline
from src.mcp_client import search_and_fetch


# ---------- structured outputs for the LLM-only nodes ----------

class SubQuery(BaseModel):
    query: str
    type: str = Field(description="one of: chat, rag, hybrid")


class PlannerOutput(BaseModel):
    sub_queries: list[SubQuery]


class ValidatorOutput(BaseModel):
    missing_info: list[str] = Field(
        default_factory=list,
        description="Missing fields needed for PA/rag sub-queries, e.g. 'payer name', 'LOB', 'code or symptom'",
    )


class ClinicalExpertOutput(BaseModel):
    needs_dt: bool = Field(description="True if this is a prior-auth eligibility decision")
    response: str


PLANNER_PROMPT = """You are a clinical assistant query planner.
Split the user's message into one or more sub-queries. Tag each as:
- "chat": general clinical knowledge, no fresh/PA-specific context needed
- "rag": prior-authorization (PA) guideline question, needs payer/policy lookup
- "hybrid": needs both general knowledge and PA guideline lookup
Keep sub-queries minimal — usually just one unless the user clearly asked multiple things."""

VALIDATOR_PROMPT = """You validate prior-authorization (PA) sub-queries before lookup.
For every sub-query tagged "rag" or "hybrid", check the full conversation for these three
required fields: payer name (insurer), code or symptom/condition, and LOB (line of business,
e.g. Medicare/Medicaid/Commercial). List any that are missing in `missing_info` using exactly
these labels: "payer name", "code or symptom", "LOB". If nothing is missing, or there are no
rag/hybrid sub-queries, return an empty list."""

CLINICAL_EXPERT_PROMPT = """You are the final clinical expert reviewing a draft answer.
Decide if this is a prior-authorization eligibility question (needs_dt=True) or a normal
clinical answer (needs_dt=False).

If needs_dt is False: just clean up `response` into a clear final answer. Do not mention
decision trees.

If needs_dt is True: build a decision tree using the draft's criteria, following the
skill below. The skill text is instructions for YOU only — never copy its section headings
(Trigger, Build Rules, Common Patterns, etc.) or any instructional prose into `response`.
`response` must contain ONLY the filled-in tree, matching the "Output Format" block's
structure exactly (the ROOT/[qNNN] lines), with real questions substituted in — nothing else.

--- SKILL (instructions only, not output content) ---
{dt_skill}
--- END SKILL ---"""


def build_graph(llm, dt_skill_text: str = ""):
    planner_llm = llm.with_structured_output(PlannerOutput)
    validator_llm = llm.with_structured_output(ValidatorOutput)
    expert_llm = llm.with_structured_output(ClinicalExpertOutput)

    async def planner(state: ClinicalAgentState):
        result = planner_llm.invoke(
            [{"role": "system", "content": PLANNER_PROMPT}] + state["messages"]
        )
        return {"sub_queries": [sq.model_dump() for sq in result.sub_queries]}

    async def validator(state: ClinicalAgentState):
        context = [{"role": "system", "content": VALIDATOR_PROMPT}] + state["messages"] + [
            {"role": "user", "content": f"Sub-queries: {state['sub_queries']}"}
        ]
        result = validator_llm.invoke(context)
        return {"missing_info": result.missing_info}

    async def ask_user(state: ClinicalAgentState):
        # Chat-native human-in-the-loop: end the turn with a clarifying question.
        # MemorySaver keeps history, so the next user message re-enters "planner" with
        # full context. (langgraph.types.interrupt() is the alternative if you want a
        # true mid-run pause/resume instead of a new turn.)
        fields = ", ".join(state["missing_info"])
        msg = f"To look up the prior-authorization guidelines, I still need: {fields}. Could you provide that?"
        return {"final_response": msg, "messages": [{"role": "assistant", "content": msg}]}

    async def router(state: ClinicalAgentState):
        tool_results = {}
        for sq in state["sub_queries"]:
            query, qtype = sq["query"], sq["type"]
            if qtype == "chat":
                answer = llm.invoke([{"role": "user", "content": query}])
                tool_results[query] = answer.content
            else:
                raw_texts = await search_and_fetch(query)
                docs = await rag_pipeline.ainvoke({"user_query": query, "raw_texts": raw_texts})
                tool_results[query] = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
        return {"tool_results": tool_results}

    async def combiner(state: ClinicalAgentState):
        combined = "\n\n".join(f"Q: {q}\n{a}" for q, a in state["tool_results"].items())
        summary = llm.invoke([
            {"role": "system", "content": "Combine these findings into one coherent draft answer for the user."},
            {"role": "user", "content": combined},
        ])
        return {"final_response": summary.content}

    async def clinical_expert(state: ClinicalAgentState):
        prompt = CLINICAL_EXPERT_PROMPT.format(dt_skill=dt_skill_text)
        result = expert_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Original question(s): {state['sub_queries']}\n\nDraft: {state['final_response']}"},
        ])
        return {
            "needs_dt": result.needs_dt,
            "final_response": result.response,
            "messages": [{"role": "assistant", "content": result.response}],
        }

    def route_after_validation(state: ClinicalAgentState):
        return "ask_user" if state.get("missing_info") else "router"

    workflow = StateGraph(ClinicalAgentState)
    workflow.add_node("planner", planner)
    workflow.add_node("validator", validator)
    workflow.add_node("ask_user", ask_user)
    workflow.add_node("router", router)
    workflow.add_node("combiner", combiner)
    workflow.add_node("clinical_expert", clinical_expert)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "validator")
    workflow.add_conditional_edges("validator", route_after_validation, ["ask_user", "router"])
    workflow.add_edge("ask_user", END)
    workflow.add_edge("router", "combiner")
    workflow.add_edge("combiner", "clinical_expert")
    workflow.add_edge("clinical_expert", END)

    return workflow.compile(checkpointer=MemorySaver())
