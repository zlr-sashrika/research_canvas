from typing import cast, Literal, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel
from langchain.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from app.utils.web import get_resource, download_resource
from app.models import get_model
import json

# State definition
class AgentState(BaseModel):
    messages: List[dict] = []
    research_question: str = ""
    report: str = ""
    resources: List[dict] = []
    logs: List[dict] = []
    model: str = "openai"

# Tool definitions
@tool
def Search(queries: List[str]):
    """A list of one or more search queries to find good resources to support the research."""

@tool
def WriteReport(report: str):
    """Write the research report."""

@tool
def WriteResearchQuestion(research_question: str):
    """Write the research question."""

@tool
def DeleteResources(urls: List[str]):
    """Delete the URLs from the resources."""

# Node implementations
async def chat_node(state: AgentState) -> Command[Literal["search_node", "chat_node", "delete_node", "__end__"]]:
    resources = []
    for resource in state.resources:
        content = get_resource(resource["url"])
        if content != "ERROR":
            resources.append({**resource, "content": content})

    model = get_model(state.model)
    response = await model.bind_tools([
        Search,
        WriteReport,
        WriteResearchQuestion,
        DeleteResources
    ]).ainvoke([
        SystemMessage(content=f"""
        You are a research assistant. Help the user with writing a research report.
        Use the resources to answer questions but don't just recite them.
        Use the search tool to find resources when needed.
        Use WriteReport to save the report - never just respond with it.
        If there's a research question, don't ask for it again.

        Research question: {state.research_question}
        Current report: {state.report}
        Resources: {json.dumps(resources)}
        """),
        *state.messages
    ])

    ai_message = cast(AIMessage, response)
    goto = "__end__"

    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        if tool_call.name == "Search":
            goto = "search_node"
        elif tool_call.name == "DeleteResources":
            goto = "delete_node"
        elif tool_call.name in ["WriteReport", "WriteResearchQuestion"]:
            new_state = {
                "messages": [*state.messages, ai_message]
            }
            
            if tool_call.name == "WriteReport":
                new_state["report"] = tool_call.args.get("report", "")
            else:
                new_state["research_question"] = tool_call.args.get("research_question", "")

            return Command(
                goto="chat_node",
                update=new_state
            )

    return Command(
        goto=goto,
        update={"messages": [*state.messages, response]}
    )

async def search_node(state: AgentState) -> AgentState:
    # Search implementation here
    return state

async def delete_node(state: AgentState) -> AgentState:
    # Delete implementation here
    return state

# Graph construction
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chat_node", chat_node)
workflow.add_node("search_node", search_node)
workflow.add_node("delete_node", delete_node)

# Add edges
workflow.add_edge("chat_node", "search_node")
workflow.add_edge("chat_node", "delete_node")
workflow.add_edge("search_node", "chat_node")
workflow.add_edge("delete_node", "chat_node")

# Set entry point
workflow.set_entry_point("chat_node")

# Compile graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)