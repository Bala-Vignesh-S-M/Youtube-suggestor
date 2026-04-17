import os
import requests
from fastapi import HTTPException

YT_API_URL = "https://www.googleapis.com/youtube/v3/search"

def search_youtube_videos(query: str, max_results: int = 5) -> list:
    """
    Uses the YouTube Data API v3 to search for videos.
    Returns: A list of up to `max_results` videos containing title, videoId, thumbnail, and embed_url.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY environment variable not set.")
        
    params = {
        "part": "snippet",
        "maxResults": max_results,
        "q": query,
        "type": "video",
        "key": api_key
    }
    
    try:
        response = requests.get(YT_API_URL, params=params, timeout=10)
        data = response.json()
        
        if not response.ok:
            error_msg = data.get("error", {}).get("message", "Unknown YouTube Error")
            raise HTTPException(status_code=response.status_code, detail=f"YouTube API Error: {error_msg}")
            
        videos = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            videos.append({
                "title": item["snippet"]["title"],
                "videoId": video_id,
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                "embed_url": f"https://www.youtube.com/embed/{video_id}"
            })
            
        return videos
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"YouTube API request failed: {str(e)}")
