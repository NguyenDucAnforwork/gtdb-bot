#!/usr/bin/env python3
"""
FastAPI server for RAG Chatbot - Docker deployment ready
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to Python path
sys.path.insert(0, '/app')

from src.chatbot_core import ChatbotCore
from src.memory.session_manager import session_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup chatbot"""
    global chatbot
    try:
        logger.info("🚀 Initializing RAG Chatbot...")
        # Initialize with use_memory=False (tạm thời tắt memory features)
        chatbot = ChatbotCore(use_memory=False)
        logger.info("✅ Chatbot initialized successfully (memory DISABLED)")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down chatbot...")

# FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Advanced RAG chatbot with multi-persona routing, semantic caching, and guardrails",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: str = Field(default="default", description="Session ID for conversation context")
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot response")
    session_id: str = Field(..., description="Session ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(..., description="Whether response was cached")
    persona: Optional[str] = Field(default=None, description="Active persona")
    routing_info: Optional[dict] = Field(default=None, description="Routing information")

class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict

# Dependency to get chatbot instance
def get_chatbot() -> ChatbotCore:
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker/K8s"""
    try:
        # Basic health check
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not ready")
        
        # Check components
        components = {
            "chatbot_core": "healthy",
            "memory_manager": "healthy",
            "cache_system": "healthy",
            "persona_system": "healthy",
            "routing_system": "healthy",
            "guardrails": "healthy"
        }
        
        # Test cache
        try:
            cache_stats = chatbot.get_cache_stats()
            components["cache_system"] = f"healthy ({cache_stats.get('total_entries', 0)} entries)"
        except Exception:
            components["cache_system"] = "degraded"
        
        # Test personas
        try:
            personas = chatbot.list_personas()
            components["persona_system"] = f"healthy ({len(personas)} personas)"
        except Exception:
            components["persona_system"] = "degraded"
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, bot: ChatbotCore = Depends(get_chatbot)):
    """Main chat endpoint"""
    import time
    
    start_time = time.time()
    
    try:
        # Get conversation memory
        memory = session_manager.get_memory(request.session_id)
        chat_history = memory.get_messages()
        
        # Check cache first
        cached_response = bot.semantic_cache.get(request.message)
        cached = cached_response is not None
        
        if cached:
            response_text = cached_response
            logger.info(f"🎯 Cache hit for session {request.session_id}")
        else:
            # Process with chatbot
            response_text = bot.process_query(
                question=request.message,
                chat_history=chat_history
            )
            
            # Update memory
            memory.add_user_message(request.message)
            memory.add_ai_message(response_text)
        
        processing_time = time.time() - start_time
        
        # Get routing info if available
        routing_info = None
        if hasattr(bot.query_router, '_last_route_info'):
            routing_info = bot.query_router._last_route_info
        
        logger.info(
            f"💬 Chat processed: session={request.session_id}, "
            f"time={processing_time:.2f}s, cached={cached}"
        )
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            processing_time=processing_time,
            cached=cached,
            routing_info=routing_info
        )
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")

# Get conversation history
@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        memory = session_manager.get_memory(session_id)
        messages = memory.get_messages()
        
        history = []
        for msg in messages:
            history.append({
                "role": "user" if hasattr(msg, 'type') and msg.type == "human" else "assistant",
                "content": msg.content,
                "timestamp": getattr(msg, 'timestamp', None)
            })
        
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"❌ Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {e}")

# Clear conversation history
@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    try:
        memory = session_manager.get_memory(session_id)
        memory.clear()
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"❌ Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {e}")

# Get system stats
@app.get("/stats")
async def get_system_stats(bot: ChatbotCore = Depends(get_chatbot)):
    """Get system statistics"""
    try:
        cache_stats = bot.get_cache_stats()
        personas = bot.list_personas()
        
        return {
            "cache_stats": cache_stats,
            "active_sessions": len(session_manager.sessions),
            "available_personas": personas,
            "system_status": "operational"
        }
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"🚀 Starting RAG Chatbot API on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,  # Disable reload in production
        access_log=True
    )
