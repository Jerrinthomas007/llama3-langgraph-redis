from typing import TypedDict
from src.shared.types import ChatRequest, ChatResponse
from src.backend.core.memory import redis_client
from src.backend.core.vector_store import qdrant

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
    context: str
    output: str

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Respond using prior chat history and relevant context if any.

Chat History:
{history}

Relevant Info:
{context}

User: {input}
""".strip())

# Reuse the model to avoid repeated initialization
llm = ChatGroq(model="llama3-8b-8192")

# Load user chat history from Redis
def load_history(state: ChatState) -> ChatState:
    key = f"chat:{state['user_id']}:history"
    messages = redis_client.lrange(key, 0, -1)
    history = "\n".join(
        f"{json.loads(m)['user']}: {json.loads(m)['bot']}" for m in messages
    )
    return {**state, "history": history}

# Retrieve relevant context from Qdrant
def retrieve_context(state: ChatState) -> ChatState:
    docs = qdrant.similarity_search(state["input"], k=2)
    context = "\n".join(doc.page_content for doc in docs)
    return {**state, "context": context}

# Generate AI response from LLM
def generate_response(state: ChatState) -> ChatState:
    prompt_msg = prompt.format_messages(
        input=state["input"],
        history=state["history"],
        context=state["context"]
    )
    response = llm.invoke(prompt_msg)
    return {**state, "output": response.content}

# Store user-bot message pair into Redis
def store_memory(state: ChatState) -> ChatState:
    key = f"chat:{state['user_id']}:history"
    redis_client.rpush(key, json.dumps({
        "user": state["input"],
        "bot": state["output"]
    }))
    return state

# Store both user query and bot reply in vector DB
def store_vector(state: ChatState) -> ChatState:
    qdrant.add_texts([state["input"], state["output"]])
    return state

# Main chat pipeline
def run_chat(request: ChatRequest) -> ChatResponse:
    # Validate basic request inputs
    if not request.message or not request.user_id:
        return ChatResponse(response="Invalid input or user ID")

    builder = StateGraph(ChatState)

    builder.add_node("load_history", RunnableLambda(load_history))
    builder.add_node("search_memory", RunnableLambda(retrieve_context))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("store_history", RunnableLambda(store_memory))
    builder.add_node("store_vector", RunnableLambda(store_vector))

    builder.set_entry_point("load_history")
    builder.add_edge("load_history", "search_memory")
    builder.add_edge("search_memory", "generate")
    builder.add_edge("generate", "store_history")
    builder.add_edge("store_history", "store_vector")
    builder.add_edge("store_vector", END)

    graph = builder.compile()

    state = graph.invoke({
        "input": request.message,
        "user_id": request.user_id,
        "history": "",
        "context": "",
        "output": ""
    })

    return ChatResponse(response=state["output"])

