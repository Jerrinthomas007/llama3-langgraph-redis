from typing import TypedDict
from src.shared.types import ChatRequest, ChatResponse
from src.backend.core.memory import redis_client
from src.backend.tools.online_search import duckduckgo  # Your custom search tool

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

import json

# Define structured state for LangGraph
class ChatState(TypedDict):
    input: str
    user_id: str
    history: str
    output: str
    action: str  # "search" or "respond"
    search_result: str

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant named Jerrin. Respond using prior chat history and, if needed, include online search results.

Chat History:
{history}

Relevant Info:
{context}

User: {input}
""".strip())

# LLM instance
llm = ChatGroq(model="llama3-8b-8192")

# Step 1: Load chat history from Redis
def load_history(state: ChatState) -> ChatState:
    key = f"chat:{state['user_id']}:history"
    messages = redis_client.lrange(key, 0, -1)
    history = "\n".join(
        f"{json.loads(m)['user']}: {json.loads(m)['bot']}" for m in messages
    )
    return {**state, "history": history}

# Step 2: Decide if online search is needed
def decide_action(state: ChatState) -> ChatState:
    decision_prompt = f"""
You're an AI assistant. Given this user message, decide whether you need to search online or you can answer from your own knowledge.

Message: {state['input']}

Respond with one word: "search" or "respond".
""".strip()
    decision = llm.invoke(decision_prompt).content.strip().lower()
    return {**state, "action": decision}

# Step 3: Online search

def online_search_tool(state: ChatState) -> ChatState:
    result = duckduckgo(state["input"])  # Assume this returns a string
    return {**state, "search_result": result}

# Step 4: Generate final response

def generate_response(state: ChatState) -> ChatState:
    if state["action"] == "search":
        context = f"Search result:\n{state['search_result']}"
    else:
        context = "No external context required."

    prompt_msg = prompt.format_messages(
        input=state["input"],
        history=state["history"],
        context=context
    )
    response = llm.invoke(prompt_msg)
    return {**state, "output": response.content}

# Step 5: Store chat history

def store_history(state: ChatState) -> ChatState:
    key = f"chat:{state['user_id']}:history"
    redis_client.rpush(key, json.dumps({
        "user": state["input"],
        "bot": state["output"]
    }))
    return state

# Main chat function

def run_chat(request: ChatRequest) -> ChatResponse:
    if not request.message or not request.user_id:
        return ChatResponse(response="Invalid input or user ID")

    builder = StateGraph(ChatState)

    builder.add_node("load_history", RunnableLambda(load_history))
    builder.add_node("decide_action", RunnableLambda(decide_action))
    builder.add_node("search_online", RunnableLambda(online_search_tool))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("store_history", RunnableLambda(store_history))

    builder.set_entry_point("load_history")
    builder.add_edge("load_history", "decide_action")

    builder.add_conditional_edges(
        "decide_action",
        lambda state: state["action"],
        {
            "search": "search_online",
            "respond": "generate"
        }
    )

    builder.add_edge("search_online", "generate")
    builder.add_edge("generate", "store_history")
    builder.add_edge("store_history", END)

    graph = builder.compile()

    # Initial empty state
    state = graph.invoke({
        "input": request.message,
        "user_id": request.user_id,
        "history": "",
        "output": "",
        "action": "",
        "search_result": ""
    })

    return ChatResponse(response=state["output"])
