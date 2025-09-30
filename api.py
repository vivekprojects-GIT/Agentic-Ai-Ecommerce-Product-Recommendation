"""FastAPI backend for Commerce Agent with single unified endpoint."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Generator
import logging
import os
import json
from pathlib import Path

from src.config import get_config, setup_logging, reload_config
# Import agent lazily inside startup to allow server to run without ML deps

# Setup dynamic logging
setup_logging()
logger = logging.getLogger(__name__)

# Get dynamic configuration
config = get_config()

# Initialize FastAPI app with dynamic configuration
app = FastAPI(
    title=config.ui_title,
    description=config.ui_description,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global conversational agent instance
agent = None

# Pydantic models
class AskRequest(BaseModel):
    text_input: Optional[str] = None
    image_base64: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    conversation_context: Optional[Dict[str, Any]] = None

class AskResponse(BaseModel):
    response: str
    products: List[Dict[str, Any]]
    intent: str
    confidence: float
    metadata: Dict[str, Any]

class ReloadResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize the dynamic modular commerce agent on startup."""
    global agent
    try:
        logger.info("Initializing Dynamic Modular Commerce Agent...")

        # Lazy import here to avoid hard dependency during server boot
        from src.agent import ModularAgenticAgent  # noqa: WPS433

        # Initialize the modular agentic agent with dynamic configuration
        agent = ModularAgenticAgent()
        logger.info(f"Dynamic Modular Agentic Commerce Agent initialized successfully with {config.llm_model}")
        
    except Exception as e:
        # Do not block server start; run in degraded mode without agent
        agent = None
        logger.error(f"Failed to initialize Commerce Agent: {e}")
        logger.warning("Starting API in degraded mode without ML dependencies. Text/image search will use a simple fallback.")

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "commerce-agent"}

@app.post("/admin/reload", response_model=ReloadResponse)
async def admin_reload():
    """Reload the agent and tools with updated configuration."""
    global agent
    try:
        # Reload configuration and reinitialize agent
        reload_config()
        from src.agent import ModularAgenticAgent  # noqa: WPS433
        agent = ModularAgenticAgent()
        return ReloadResponse(status="ok", message="Dynamic modular agentic agent and tools reloaded with updated configuration.")
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@app.post("/api/v1/simple-rag/ask", response_model=AskResponse)
async def ask_agent(request: AskRequest):
    """
    Unified endpoint for all commerce agent interactions.
    
    Handles:
    - General conversation
    - Text-based product search
    - Image-based product search
    """
    if agent is None:
        # Degraded mode fallback response
        text = (request.text_input or "").strip()
        if not text and not request.image_base64:
            raise HTTPException(status_code=400, detail="No input provided")
        fallback = {
            "response": "Service running in setup mode. Please complete dependency installation for full features.",
            "products": [],
            "intent": "general_chat",
            "confidence": 0.2,
            "metadata": {"degraded_mode": True}
        }
        return AskResponse(**fallback)
    
    try:
        # Process the request using the agentic system with conversation context
        result = agent.process_request(
            message=request.text_input or "",
            image_base64=request.image_base64,
            conversation_context=request.conversation_context
        )
        
        return AskResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/commerce-agent/ask-stream")
async def ask_agent_stream(request: AskRequest):
    """
    Streaming endpoint for commerce agent interactions.
    
    Returns Server-Sent Events (SSE) stream with:
    - Status updates
    - Partial responses
    - Final results
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    def generate_stream() -> Generator[str, None, None]:
        """Generate streaming response."""
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...'})}\n\n"

            # Process the request with streaming
            for chunk in agent.process_request_stream(
                message=request.text_input or "",
                image_base64=request.image_base64
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api_host, port=config.api_port)
