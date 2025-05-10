import os
import redis
import json
from typing import TypedDict
import getpass
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# ----------- Redis Setup -----------
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ----------- Qdrant Setup -----------
embedding_model = HuggingFaceEmbeddings()
qdrant = Qdrant(
    client=QdrantClient(host="localhost", port=6333),
    collection_name="chatbot-memory",
    embeddings=embedding_model,
)

# ----------- LLM Setup (Groq + LLaMA3) -----------
llm = ChatGroq(model="llama3-8b-8192")

# ----------- Prompt -----------
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Respond using prior chat history and relevant context if any.

Chat History:
{history}

Relevant Info:
{context}

User: {input}
""")

# ----------- State Definition -----------
class ChatState(TypedDict):
    input: str
    history: str
    context: str
    output: str

# ----------- Nodes (LangGraph Functions) -----------

def load_history(state: ChatState) -> ChatState:
    messages = redis_client.lrange("chat_history", 0, -1)
    history = "\n".join([json.loads(msg)["user"] + ": " + json.loads(msg)["bot"] for msg in messages])
    return {**state, "history": history}


def retrieve_context(state: ChatState) -> ChatState:
    docs = qdrant.similarity_search(state["input"], k=2)
    context = "\n".join(doc.page_content for doc in docs)
    return {**state, "context": context}


def generate_response(state: ChatState) -> ChatState:
    prompt_msg = prompt.format_messages(
        input=state["input"],
        history=state["history"],
        context=state["context"]
    )
    response = llm.invoke(prompt_msg)
    return {**state, "output": response.content}


def store_memory(state: ChatState) -> ChatState:
    entry = json.dumps({"user": state["input"], "bot": state["output"]})
    redis_client.rpush("chat_history", entry)
    return state
 

def index_vector_memory(state: ChatState) -> ChatState:
    # Save both user input and bot output as semantic memory
    qdrant.add_texts([state["input"], state["output"]])
    return state

# ----------- Build LangGraph -----------

builder = StateGraph(ChatState)

builder.add_node("load_history", RunnableLambda(load_history))
builder.add_node("search_memory", RunnableLambda(retrieve_context))
builder.add_node("generate", RunnableLambda(generate_response))
builder.add_node("store_history", RunnableLambda(store_memory))
builder.add_node("store_vector", RunnableLambda(index_vector_memory))

builder.set_entry_point("load_history")
builder.add_edge("load_history", "search_memory")
builder.add_edge("search_memory", "generate")
builder.add_edge("generate", "store_history")
builder.add_edge("store_history", "store_vector")
builder.add_edge("store_vector", END)

graph = builder.compile()

# ----------- Chat Loop -----------

if __name__ == "__main__":
    print("💬 Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        state = graph.invoke({"input": user_input})
        print(f"🤖 Bot: {state['output']}")
