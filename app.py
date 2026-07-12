import gradio as gr

from config import llm
from src.workflow import build_graph

with open("DT_SKILL.md") as f:
    DT_SKILL_TEXT = f.read()

graph = build_graph(llm, dt_skill_text=DT_SKILL_TEXT)


async def chat(user_input: str, history):
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
    )
    return result["final_response"]


if __name__ == "__main__":
    gr.ChatInterface(chat).launch()
