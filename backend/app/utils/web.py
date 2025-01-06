import httpx
import html2text
from typing import Dict

# Cache for downloaded resources
_RESOURCE_CACHE: Dict[str, str] = {}

def get_resource(url: str) -> str:
    """Get resource content from cache"""
    return _RESOURCE_CACHE.get(url, "")

async def download_resource(url: str) -> str:
    """Download and cache resource content"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            html_content = response.text
            content = html2text.html2text(html_content)
            _RESOURCE_CACHE[url] = content
            return content
    except Exception as e:
        error_message = f"Error downloading resource: {str(e)}"
        _RESOURCE_CACHE[url] = error_message
        return error_message