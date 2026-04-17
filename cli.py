import os
import argparse
import sys
from dotenv import load_dotenv

# Load locals before services
load_dotenv()

from services.huggingface import analyze_text_and_get_query
from services.youtube import search_youtube_videos

def main():
    parser = argparse.ArgumentParser(description="Mood-Based YouTube Video Suggestor CLI (Llama Edition)")
    parser.add_argument("text", nargs="*", help="Your mood or intent text")
    args = parser.parse_args()
    
    text = " ".join(args.text)
    
    if not text:
        text = input("Enter your mood or what you want to watch (e.g., 'i played cricket, it was interesting'): ")
        
    if not text.strip():
        print("Error: Text input cannot be empty.")
        sys.exit(1)
        
    print(f"\n[1] Letting Llama 3.3 analyze: '{text}'...")
    try:
        llm_output = analyze_text_and_get_query(text)
        sentiment = llm_output.get("mood", "unknown")
        query = llm_output.get("search_query", text)
        
        print(f" -> Detected Mood: {sentiment.upper()}")
        print(f" -> Optimized YouTube Query: '{query}'")
        
        print("\n[2] Searching YouTube...")
        videos = search_youtube_videos(query, max_results=5)
        
        print("\n=== RECOMMENDED VIDEOS ===\n")
        if not videos:
            print("No videos found.")
        else:
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video['title']}")
                print(f"   Link: {video['embed_url'].replace('embed/', 'watch?v=')}")
                print()
                
    except Exception as e:
        print(f"\n[X] Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
