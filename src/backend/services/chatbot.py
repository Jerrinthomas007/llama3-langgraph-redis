from typing import TypedDict
from src.shared.types import ChatRequest, ChatResponse
from src.backend.core.memory import redis_client
from src.backend.tools.online_search import duckduckgo_search,scrape_webpage
from src.backend.tools.dnstoys import get_weather

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
    action: str
    search_result: str

# Prompt template for response
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


def query_generator(mssg):
    query_prompt = f"""
    You are not an AI assistant that responds to a user. You are an AI web search query generator model. 
    You will be given a prompt to an AI assistant with web search capabilities. If you are being used, an 
    AI has determined this prompt to the actual AI assistant, requires web search for more recent data. 
    You must determine what the data is the assistant needs from search and generate the best possible 
    DuckDuckGo query to find that data. Do not respond with anything but a query that an expert human 
    search engine user would type into DuckDuckGo to find the needed data. Keep your queries simple, 
    without any search engine code. Just type a query likely to retrieve the data we need.

    prompt : {mssg}
    """.strip()
    query = llm.invoke(query_prompt).content.strip().lower()
    if query[0]=='"':
        query=query[1:-1]
    return query

def best_search_result(search_results,user_message,query):
    query_prompt = f"""
    You are not an AI assistant that responds to a user. You are an AI model trained to select the best '
    'search result out of a list of ten results. The best search result is the link an expert human search '
    'engine user would click first to find the data to respond to a USER_PROMPT after searching DuckDuckGo '
    'for the SEARCH_QUERY. \nAll user messages you receive in this conversation will have the format of: \n'
    '  SEARCH_RESULTS: [] \n'
    '  USER_PROMPT: "this will be an actual prompt to a web search enabled AI assistant" \n'
    '  SEARCH_QUERY: "search query ran to get the above 10 links" \n\n'
    'You must select the index from the 0 indexed SEARCH_RESULTS list and only respond with the index of '
    'the best search result to check for the data the AI assistant needs to respond. That means your responses '
    'to this conversation should always be 1 token, being an integer between 0–9.'

    the SEARCH_RESULTS : {search_results},
    the  USER_PROMPT : {user_message},
    the  SEARCH_QUERY: {query}
    """.strip()

    for _ in range(2):
        try:
            best_result = llm.invoke(query_prompt).content.strip().lower()
            return int(best_result)
        except:
            continue
    
    return 0




# Step 1: Load history
def load_history(state: ChatState) -> ChatState:
    key = f"chat:{state['user_id']}:history"
    messages = redis_client.lrange(key, 0, -1)
    history = "\n".join(f"{json.loads(m)['user']}: {json.loads(m)['bot']}" for m in messages)
    return {**state, "history": history}

# Step 2: Decide action
def decide_action(state: ChatState) -> ChatState:
    decision_prompt = f"""
You're an AI assistant. Given this user message, decide whether you need to search online or can answer from your own knowledge.

Message: {state['input']}
Respond with one word: "search" or "respond".
""".strip()
    decision = llm.invoke(decision_prompt).content.strip().lower()
    return {**state, "action": decision}

# Step 3: Use LLM to extract city if weather is being asked
def get_city_name(user_input: str) -> str:
    extract_prompt = f"""
Extract only the city name from the following message if it's asking about weather.
Message: "{user_input}"
City (return only the city name, no extra text):
""".strip()
    return llm.invoke(extract_prompt).content.strip()

# Step 4: Online search tool — fallback for weather or general query
def online_search_tool(state: ChatState) -> ChatState:
    if "weather" in state["input"].lower():
        city = get_city_name(state["input"])
        if city:
            weather = get_weather(city)
            return {**state, "search_result": f"Weather in {city}:\n{weather}"}
    # fallback to general search

    query = query_generator(state["input"])
    search_results = duckduckgo_search(query)
    bs_results = best_search_result(search_results,state["input"],query)
    result = scrape_webpage(search_results[bs_results]['link'])
    if result == None:
        result = search_results[bs_results]['search_description']
    return {**state, "search_result": result}

# Step 5: Generate response
def generate_response(state: ChatState) -> ChatState:
    if state["action"] == "search":
        context = f"{state['search_result']}"
    else:
        context = "No external context required."

    prompt_msg = prompt.format_messages(
        input=state["input"],
        history=state["history"],
        context=context
    )
    response = llm.invoke(prompt_msg)
    return {**state, "output": response.content}

# Step 6: Store in Redis
def store_history(state: ChatState) -> ChatState:
    key = f"chat:{state['user_id']}:history"
    redis_client.rpush(key, json.dumps({
        "user": state["input"],
        "bot": state["output"]
    }))
    return state

# Final chat endpoint
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

    state = graph.invoke({
        "input": request.message,
        "user_id": request.user_id,
        "history": "",
        "output": "",
        "action": "",
        "search_result": ""
    })

    return ChatResponse(response=state["output"])
