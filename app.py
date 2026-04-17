import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

# Import our custom services
from services.huggingface import analyze_text_and_get_query
from services.youtube import search_youtube_videos

# Initialize FastAPI App
app = FastAPI(title="Mood-Based YouTube Video Suggestor")

# Setup generic logging to print to terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output models
class RecommendRequest(BaseModel):
    text: str

# API endpoint
@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    text = req.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    logger.info(f"Received request for text: '{text}'")
    
    # 1. Use Llama 3.3 to analyze mood AND generate query directly
    llm_output = analyze_text_and_get_query(text)
    
    sentiment = llm_output.get("mood", "unknown")
    query = llm_output.get("search_query", text)
    
    logger.info(f"LLM Detected Mood: '{sentiment}'")
    logger.info(f"LLM Generated Query: '{query}'")
    
    # 2. Search YouTube using the exact optimized query
    videos = search_youtube_videos(query, max_results=5)
    logger.info(f"Found {len(videos)} videos from YouTube")
    
    # 3. Return Output properly formatted for frontend
    return {
        "mood": sentiment,
        "query": query,
        "videos": videos
    }

# Mount static files to root AFTER api routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
