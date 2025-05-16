from typing import TypedDict, Optional, List, Dict, Any, Union
from src.shared.types import ChatRequest, ChatResponse
from src.backend.core.memory import redis_client
from src.backend.tools.online_search import duckduckgo_search, scrape_webpage
from src.backend.tools.dnstoys import get_weather

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

import json
import logging
import time
import traceback
from functools import lru_cache
from datetime import datetime
import re
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants with improved flexibility
GENERAL_KNOWLEDGE_CACHE_TTL = 86400  # 24 hours for general knowledge
PERSONALIZED_CACHE_TTL = 3600  # 1 hour for personalized responses
WEATHER_CACHE_TTL = 900  # 15 minutes for weather
SEARCH_RESULTS_CACHE_TTL = 1800  # 30 minutes for search results
MAX_HISTORY_LENGTH = 50  # Maximum number of messages to store in history
MAX_PROMPT_LENGTH = 8000  # Maximum length for prompts
MAX_RETRIES = 3  # Maximum retries for LLM calls
RETRY_DELAY = 1  # Delay between retries in seconds
MAX_CONCURRENT_LLM_CALLS = 5  # Limit concurrent LLM calls

# Rate limiting settings
RATE_LIMIT_PERIOD = 60  # 1 minute
RATE_LIMIT_MAX_CALLS = 10  # Max 10 calls per minute

# Define structured state for LangGraph
class ChatState(TypedDict):
    input: str
    user_id: str
    history: str
    output: str
    action: str
    search_result: str
    is_general: bool
    error: Optional[str]
    start_time: float
    rate_limited: bool
    city: Optional[str]
    query: Optional[str]

# Initialize LLM with error handling
def get_llm():
    try:
        return ChatGroq(model="llama3-8b-8192", temperature=0.7)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise RuntimeError(f"Could not initialize language model: {str(e)}")

# LLM instance with retry mechanism
llm = get_llm()

# Improved prompt template with better security
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant named Jerrin. Respond using prior chat history and, if needed, include online search results.

Chat History:
{history}

Relevant Info:
{context}

User: {input}

Respond in a helpful, accurate, and conversational manner. If you don't know the answer, be honest about it.
""".strip())

# Rate limiter implementation
class RateLimiter:
    def __init__(self, period: int, max_calls: int):
        self.period = period
        self.max_calls = max_calls
        self.calls = {}  # user_id -> list of timestamps
    
    def check_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        now = time.time()
        if user_id not in self.calls:
            self.calls[user_id] = []
        
        # Remove old timestamps
        self.calls[user_id] = [t for t in self.calls[user_id] if now - t < self.period]
        
        # Check if limit exceeded
        if len(self.calls[user_id]) >= self.max_calls:
            return False
        
        # Add new timestamp
        self.calls[user_id].append(now)
        return True

# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_PERIOD, RATE_LIMIT_MAX_CALLS)

# Enhanced input sanitization
def sanitize_input(text: str) -> str:
    """Sanitize input to prevent prompt injection"""
    # Replace quotes with safer alternatives
    text = text.replace('"', "'")
    # Remove potential control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # Limit length to prevent attacks
    return text[:MAX_PROMPT_LENGTH]

# Improved LLM invocation with retries and error handling
def invoke_llm_prompt(prompt: str, retries: int = MAX_RETRIES) -> Tuple[str, Optional[str]]:
    """Invoke LLM with retries and error handling"""
    sanitized_prompt = sanitize_input(prompt)
    
    for attempt in range(retries):
        try:
            response = llm.invoke(sanitized_prompt)
            return response.content.strip(), None
        except Exception as e:
            logger.warning(f"LLM invocation failed (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            else:
                error_msg = f"Failed to generate response after {retries} attempts"
                logger.error(f"{error_msg}: {str(e)}")
                return "", error_msg
    
    return "", "Unknown error in LLM invocation"

# Enhanced cache key generation
def get_cache_key(user_id: str, message: str, is_general: bool, action: str = "") -> str:
    """Generate cache key with prefix for better organization"""
    message_hash = hash(message) % 10000000  # Use hash to avoid too long keys
    
    if is_general:
        return f"llm:global:{action}:{message_hash}"
    else:
        return f"llm:user:{user_id}:{action}:{message_hash}"

# Load history with pagination
def load_history(state: ChatState) -> ChatState:
    """Load chat history with pagination"""
    try:
        key = f"chat:{state['user_id']}:history"
        # Get only the last MAX_HISTORY_LENGTH messages
        messages = redis_client.lrange(key, -MAX_HISTORY_LENGTH, -1)
        
        # Parse messages with error handling
        history_entries = []
        for m in messages:
            try:
                entry = json.loads(m)
                if isinstance(entry, dict) and 'user' in entry and 'bot' in entry:
                    history_entries.append(f"User: {entry['user']}\nJerrin: {entry['bot']}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse history entry: {m}")
                continue
        
        history = "\n\n".join(history_entries)
        # Limit history size to prevent token overflow
        if len(history) > MAX_PROMPT_LENGTH:
            history = history[-MAX_PROMPT_LENGTH:]
            
        return {**state, "history": history}
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        return {**state, "history": "", "error": f"Failed to load chat history: {str(e)}"}

# Improved action decision with better prompting
def decide_action(state: ChatState) -> ChatState:
    """Decide whether to search or directly respond"""
    if state.get("error"):
        return state
    
    # Check rate limit
    if not rate_limiter.check_limit(state["user_id"]):
        return {**state, "rate_limited": True, "action": "respond", 
                "error": "Rate limit exceeded. Please try again later."}
    
    # For weather queries, set action directly
    if "weather" in state["input"].lower():
        return {**state, "action": "weather"}
    
    decision_prompt = f"""
As a decision-making AI, analyze the following user request and determine if external information is needed.

User request: {sanitize_input(state['input'])}

Choose one option:
1. "search" - if recent or external information is required (news, current events, specific data)
2. "respond" - if the request can be answered with general knowledge

Return only one word: "search" or "respond"
"""
    
    result, error = invoke_llm_prompt(decision_prompt)
    
    if error:
        return {**state, "action": "respond", "error": error}
    
    action = result.lower().strip()
    # Default to respond if unexpected output
    if action not in ["search", "respond"]:
        action = "respond"
        
    return {**state, "action": action}

# Determine if question is general knowledge with improved prompt
def is_general_knowledge(state: ChatState) -> ChatState:
    """Check if question is general knowledge that can be globally cached"""
    if state.get("error") or state.get("rate_limited"):
        return {**state, "is_general": False}
    
    # Weather is always treated as non-general due to locality
    if state["action"] == "weather":
        return {**state, "is_general": False}
    
    check_prompt = f"""
Determine if this question is general knowledge that can be safely cached for all users.

User Question: {sanitize_input(state['input'])}

General knowledge includes:
- Verifiable facts (scientific, historical)
- Definitions
- Common processes/procedures
- Mathematical concepts/calculations

NOT general knowledge includes:
- Personal opinions or preferences
- Personalized advice
- User-specific information
- Requests containing personal identifiers
- Time-sensitive questions

Return exactly "yes" if general knowledge, "no" if personalized or uncertain.
"""
    
    result, error = invoke_llm_prompt(check_prompt)
    
    if error:
        return {**state, "is_general": False, "error": error}
    
    is_general = result.lower().strip() == "yes"
    return {**state, "is_general": is_general}

# Enhanced city extraction for weather queries
def extract_city(state: ChatState) -> ChatState:
    """Extract city name from weather query with validation"""
    if state.get("error") or state["action"] != "weather":
        return state
    
    extract_prompt = f"""
Extract ONLY the city name from this weather query. If no specific city is mentioned, respond with "unknown".

Query: {sanitize_input(state['input'])}

Return only the city name (or "unknown"), nothing else.
"""
    
    result, error = invoke_llm_prompt(extract_prompt)
    
    if error:
        return {**state, "city": None, "error": error}
    
    city = result.strip()
    if city.lower() == "unknown":
        return {**state, "city": None, "action": "respond", 
                "error": "Could not determine city from query"}
    
    # Validate extracted city (basic check)
    if not re.match(r'^[A-Za-z\s\-]+$', city):
        return {**state, "city": None, "action": "respond", 
                "error": "Invalid city name extracted"}
    
    return {**state, "city": city}

# Improved query generation with better context
def generate_search_query(state: ChatState) -> ChatState:
    """Generate optimized search query"""
    if state.get("error") or state["action"] != "search":
        return state
    
    query_prompt = f"""
You are a web search query generator. Given a user's question, create a brief, optimized search query.

User's question: {sanitize_input(state['input'])}

Generate a concise search query (2-6 words) that will find the most relevant information.
Return only the query text, no other explanation.
"""
    
    result, error = invoke_llm_prompt(query_prompt)
    
    if error:
        return {**state, "query": None, "error": error}
    
    # Clean up query
    query = result.strip()
    query = re.sub(r'["\']', '', query)  # Remove quotes
    
    return {**state, "query": query}

# Improved search result selection
def select_best_search_result(results: List[Dict[str, Any]], user_message: str, query: str) -> Tuple[int, Optional[str]]:
    """Select best search result with improved error handling"""
    if not results:
        return 0, "No search results available"
    
    # Sanitize inputs
    safe_query = sanitize_input(query)
    safe_message = sanitize_input(user_message)
    
    # Create a simplified version of results for the prompt
    simplified_results = []
    for i, result in enumerate(results[:10]):  # Limit to 10 results
        simplified_results.append({
            "index": i,
            "title": result.get("title", "No title"),
            "snippet": result.get("search_description", "No description")
        })
    
    query_prompt = f"""
Select the most relevant search result for this user query:

USER QUERY: {safe_message}
SEARCH TERM USED: {safe_query}

SEARCH RESULTS:
{json.dumps(simplified_results, indent=2)}

Return only the index number (0-9) of the best result.
"""
    
    for attempt in range(MAX_RETRIES):
        try:
            result, error = invoke_llm_prompt(query_prompt)
            if error:
                return 0, error
            
            # Extract number from the response
            match = re.search(r'\d+', result)
            if match:
                index = int(match.group())
                if 0 <= index < len(results):
                    return index, None
        except Exception as e:
            logger.warning(f"Error selecting search result (attempt {attempt+1}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return 0, f"Failed to select search result: {str(e)}"
            time.sleep(RETRY_DELAY)
    
    # Default to first result if extraction fails
    return 0, "Could not determine best result"

# Online search tool with improved caching and error handling
def online_search_tool(state: ChatState) -> ChatState:
    """Handle online searches with caching and error handling"""
    if state.get("error"):
        return state
    
    # Handle weather queries
    if state["action"] == "weather" and state.get("city"):
        city = state["city"]
        weather_key = f"weather:{city.lower()}"
        
        try:
            # Check cache
            cached_weather = redis_client.get(weather_key)
            if cached_weather:
                logger.info(f"Cache hit for weather in {city}")
                return {**state, "search_result": f"Weather in {city}:\n{cached_weather.decode('utf-8')}"}
            
            # Get fresh weather data
            logger.info(f"Fetching weather for {city}")
            weather = get_weather(city)
            if not weather:
                return {**state, "search_result": f"Weather data not available for {city}"}
            
            # Cache the result
            redis_client.set(weather_key, weather, ex=WEATHER_CACHE_TTL)
            return {**state, "search_result": f"Weather in {city}:\n{weather}"}
        except Exception as e:
            logger.error(f"Weather API error: {str(e)}")
            return {**state, "search_result": "", "error": f"Error fetching weather: {str(e)}"}
    
    # Handle web searches
    if state["action"] == "search" and state.get("query"):
        query = state["query"]
        query_key = f"search_result:{query}"
        
        try:
            # Check cache
            cached = redis_client.get(query_key)
            if cached:
                logger.info(f"Cache hit for search query: {query}")
                return {**state, "search_result": cached.decode("utf-8")}
            
            # Perform search
            logger.info(f"Performing search for: {query}")
            search_results = duckduckgo_search(query)
            
            if not search_results:
                return {**state, "search_result": "No search results found", "error": "No search results available"}
            
            # Select best result
            best_index, error = select_best_search_result(search_results, state["input"], query)
            if error:
                logger.warning(f"Error selecting search result: {error}")
                # Continue with default index
            
            # Scrape webpage content
            result_url = search_results[best_index]['link']
            logger.info(f"Scraping content from: {result_url}")
            
            result = scrape_webpage(result_url)
            if not result:
                result = search_results[best_index].get('search_description', 'No description available.')
                
            # Limit result size
            if len(result) > MAX_PROMPT_LENGTH:
                result = result[:MAX_PROMPT_LENGTH] + "... [content truncated]"
            
            # Cache the result
            redis_client.set(query_key, result, ex=SEARCH_RESULTS_CACHE_TTL)
            return {**state, "search_result": result}
        except Exception as e:
            logger.error(f"Search error: {str(e)}\n{traceback.format_exc()}")
            return {**state, "search_result": "", "error": f"Error performing search: {str(e)}"}
    
    return {**state, "search_result": ""}

# Check LLM cache with improved key strategy
def check_llm_cache(state: ChatState) -> ChatState:
    """Check if response is cached"""
    if state.get("error"):
        return state
    
    # Generate cache key based on input and state
    key = get_cache_key(
        state["user_id"], 
        state["input"], 
        state["is_general"],
        state["action"]
    )
    
    try:
        cached = redis_client.get(key)
        if cached:
            logger.info(f"Cache hit for response: {key}")
            return {**state, "output": cached.decode("utf-8")}
    except Exception as e:
        logger.warning(f"Cache retrieval error: {str(e)}")
        # Continue without cache
    
    return state

# Generate LLM response with improved error handling
def generate_response(state: ChatState) -> ChatState:
    """Generate LLM response with improved context handling"""
    if state.get("error"):
        return {**state, "output": f"I apologize, but I encountered an error: {state['error']}. Please try again later."}
    
    # Rate limit response
    if state.get("rate_limited"):
        return {**state, "output": "I'm receiving too many requests right now. Please try again in a minute."}
    
    # Prepare context
    if state["action"] == "search" and state["search_result"]:
        context = f"Based on online search: {state['search_result']}"
    elif state["action"] == "weather" and state["search_result"]:
        context = state["search_result"]
    else:
        context = "No external context required."
    
    try:
        # Format prompt with sanitized inputs
        prompt_msg = prompt.format_messages(
            input=sanitize_input(state["input"]),
            history=state["history"],
            context=context
        )
        
        # Generate response
        response = llm.invoke(prompt_msg)
        return {**state, "output": response.content}
    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {**state, "output": "I apologize, but I'm having trouble generating a response right now. Please try again later."}

# Cache LLM response with appropriate TTL
def cache_llm_response(state: ChatState) -> ChatState:
    """Cache LLM response with appropriate TTL"""
    if not state["output"] or state.get("error") or state.get("rate_limited"):
        return state
    
    try:
        # Generate cache key
        key = get_cache_key(
            state["user_id"], 
            state["input"], 
            state["is_general"],
            state["action"]
        )
        
        # Set appropriate TTL based on content type
        if state["is_general"]:
            ttl = GENERAL_KNOWLEDGE_CACHE_TTL
        else:
            ttl = PERSONALIZED_CACHE_TTL
        
        # Cache the response
        redis_client.set(key, state["output"], ex=ttl)
    except Exception as e:
        logger.warning(f"Cache storage error: {str(e)}")
        # Continue without caching
    
    return state

# Store chat history with validation
def store_history(state: ChatState) -> ChatState:
    """Store chat history with validation"""
    if not state["output"]:
        return state
    
    try:
        # Create history entry
        entry = {
            "user": state["input"],
            "bot": state["output"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Validate entry (basic check)
        if not isinstance(entry["user"], str) or not isinstance(entry["bot"], str):
            logger.warning("Invalid history entry types")
            return state
        
        # Store in Redis
        key = f"chat:{state['user_id']}:history"
        redis_client.rpush(key, json.dumps(entry))
        
        # Trim history to prevent unbounded growth
        redis_client.ltrim(key, -MAX_HISTORY_LENGTH, -1)
    except Exception as e:
        logger.error(f"Error storing history: {str(e)}")
        # Continue without storing history
    
    return state

# Measure and log performance
def log_performance(state: ChatState) -> ChatState:
    """Log performance metrics"""
    if state.get("start_time"):
        execution_time = time.time() - state["start_time"]
        logger.info(f"Request processed in {execution_time:.2f}s - User: {state['user_id']} - Action: {state['action']}")
    
    return state

# Main chat function with improved flow
def run_chat(request: ChatRequest) -> ChatResponse:
    """Main entry point for chat processing"""
    # Input validation
    if not request.message or not request.message.strip():
        return ChatResponse(response="Please provide a message.")
    
    if not request.user_id or not isinstance(request.user_id, str):
        return ChatResponse(response="Invalid user ID.")
    
    # Sanitize inputs
    message = sanitize_input(request.message)
    user_id = re.sub(r'[^\w\-]', '', request.user_id)  # Only allow alphanumeric and dash
    
    # Initialize state graph
    builder = StateGraph(ChatState)
    
    # Add nodes
    builder.add_node("start", RunnableLambda(lambda state: {**state, "start_time": time.time()}))
    builder.add_node("load_history", RunnableLambda(load_history))
    builder.add_node("decide_action", RunnableLambda(decide_action))
    builder.add_node("check_general", RunnableLambda(is_general_knowledge))
    builder.add_node("extract_city", RunnableLambda(extract_city))
    builder.add_node("generate_query", RunnableLambda(generate_search_query))
    builder.add_node("check_llm_cache", RunnableLambda(check_llm_cache))
    builder.add_node("search_online", RunnableLambda(online_search_tool))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("cache_llm_response", RunnableLambda(cache_llm_response))
    builder.add_node("store_history", RunnableLambda(store_history))
    builder.add_node("log_performance", RunnableLambda(log_performance))
    
    # Set entry point
    builder.set_entry_point("start")
    
    # Create edges
    builder.add_edge("start", "load_history")
    builder.add_edge("load_history", "decide_action")
    
    # Conditional flows based on action
    builder.add_conditional_edges(
        "decide_action",
        lambda state: state["action"] if not state.get("error") and not state.get("rate_limited") else "generate",
        {
            "search": "check_general",
            "weather": "check_general",
            "respond": "check_general",
            "generate": "generate"
        }
    )
    
    builder.add_edge("check_general", "check_llm_cache")
    
    builder.add_conditional_edges(
        "check_llm_cache",
        lambda state: "final" if state.get("output") else state["action"],
        {
            "search": "generate_query",
            "weather": "extract_city",
            "respond": "generate",
            "final": "store_history"
        }
    )
    
    builder.add_edge("generate_query", "search_online")
    builder.add_edge("extract_city", "search_online")
    builder.add_edge("search_online", "generate")
    builder.add_edge("generate", "cache_llm_response")
    builder.add_edge("cache_llm_response", "store_history")
    builder.add_edge("store_history", "log_performance")
    builder.add_edge("log_performance", END)
    
    # Compile the graph
    graph = builder.compile()
    
    try:
        # Execute the graph
        state = graph.invoke({
            "input": message,
            "user_id": user_id,
            "history": "",
            "output": "",
            "action": "",
            "search_result": "",
            "is_general": False,
            "error": None,
            "start_time": 0,
            "rate_limited": False,
            "city": None,
            "query": None
        })
        
        # Return response
        return ChatResponse(response=state["output"])
    except Exception as e:
        logger.error(f"Unhandled exception in chat flow: {str(e)}\n{traceback.format_exc()}")
        return ChatResponse(response="I apologize, but I encountered an unexpected error. Please try again later.")