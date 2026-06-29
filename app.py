import gradio as gr

from config import llm
from src.workflow import build_graph

graph = build_graph(llm)


def chat(user_input: str, history):
    result = graph.invoke(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the rag_pipeline tool when the user asks about prior authorization guidelines."},
                {"role": "user", "content": user_input},
            ]
        },
        config={"configurable": {"thread_id": "1"}},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    gr.ChatInterface(chat).launch()
