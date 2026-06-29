from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.pipeline import rag_pipeline


def build_graph(llm):
    tools = [rag_pipeline]
    tools_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", tools_node)
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    return workflow.compile(checkpointer=MemorySaver())
