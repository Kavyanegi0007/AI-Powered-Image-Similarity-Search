
import os
from serpapi import GoogleSearch
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
SERP_API_KEY = "27f976da3d0e63206d31173e361fc7e5520d513819abf08914774c7c5f13475c"

async def image_search(image_url: str, num: int = 5) -> List[Dict[str, Any]]:
    if not SERP_API_KEY:
        return [{"error": "Missing SERPAPI_API_KEY"}]

    params = {
        "engine": "google_lens",
        "type": "visual_matches",
        "url": image_url,
        "api_key": SERP_API_KEY
    }

    search = GoogleSearch(params)
    data = search.get_dict()

    if "visual_matches" not in data:
        return [{"error": "No visual matches found", "details": data.get("error", "Unknown error")}]

    images = data.get("visual_matches", [])[:num]
    return [
        {
            "title": x.get("title", "No Title"),
            "thumbnail": x.get("thumbnail") or x.get("link"),
            "url": x.get("link")
        }
        for x in images
    ]


