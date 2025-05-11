from fastapi import APIRouter
from src.shared.types import ChatRequest, ChatResponse
from src.backend.services.chatbot import run_chat

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    return run_chat(request)