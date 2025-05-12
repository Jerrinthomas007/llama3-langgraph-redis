# llama3-redis-qdrant-chat
A scalable and production-ready chatbot using Groq's LLaMA3 for fast LLM inference, Redis for short-term chat history, and Qdrant for long-term semantic memory. Powered by LangGraph for structured, stateful conversation flow.


conda create -n llama3-redis-qdrant-chat python=3.10 -y

conda activate llama3-redis-qdrant-chat

uvicorn src.backend.main:app --reload


cd frontend/
npm run dev