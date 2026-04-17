import os
import json
from fastapi import HTTPException
from openai import OpenAI

def analyze_text_and_get_query(text: str) -> dict:
    """
    Calls Meta Llama 3.3 70B via Hugging Face's OpenAI-compatible router.
    Extracts the user's mood and generates an optimized YouTube search query.
    Returns: dict {"mood": "...", "search_query": "..."}
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable not set.")

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
        
        system_prompt = (
            "You are an assistant designed to parse user texts and convert them into the perfect YouTube search query. "
            "1. Read the user's text and determine their general emotional mood or intent (e.g. happy, sad, bored, motivated, curious). "
            "2. Generate an optimized, short YouTube search query that would return the most satisfying videos for that context. "
            "For example, if the input is 'i played criekt, it was interesting', the search query should be something like 'cricket interesting moments' or 'cricket highlights'. "
            "You MUST output your response strictly as valid, raw JSON in this exact format: "
            '{"mood": "assessed_mood", "search_query": "your_optimized_youtube_query"}. '
            "Do NOT wrap it in markdown block quotes, just output the raw JSON string directly."
        )

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
        )
        
        content = completion.choices[0].message.content.strip()
        
        # Clean up any potential markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM did not return proper JSON. Output was: {content}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API request failed: {str(e)}")
