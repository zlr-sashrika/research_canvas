from fastapi import APIRouter, HTTPException
from app.models.research import ResearchState, Resource
from tavily import TavilyClient
import os
from typing import List

router = APIRouter()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@router.post("/search")
async def search_resources(query: str) -> List[Resource]:
    """
    Search for resources using Tavily
    """
    try:
        results = tavily_client.search(query, search_depth="advanced")
        resources = []
        
        for result in results[:5]:  # Limit to top 5 results
            resource = Resource(
                url=result["url"],
                title=result["title"],
                description=result["snippet"]
            )
            resources.append(resource)
        
        return resources
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_resources(state: ResearchState) -> ResearchState:
    """
    Analyze the current resources and update the research report
    """
    try:
        # Download any new resources
        for resource in state.resources:
            if not resource.content:
                from app.utils.web import download_resource
                content = await download_resource(resource.url)
                resource.content = content

        # Get the appropriate model
        from app.models import get_model
        model = get_model(state.model)

        # Analyze resources and generate report
        messages = [
            {
                "role": "system",
                "content": "You are a research assistant. Analyze the resources and help improve the research report."
            },
            {
                "role": "user",
                "content": f"""
                Research Question: {state.research_question}
                Current Report: {state.report}
                
                Resources:
                {[{
                    'url': r.url,
                    'title': r.title,
                    'content': r.content[:1000] + '...' if r.content else 'No content'
                } for r in state.resources]}
                
                Please analyze these resources and suggest improvements to the report.
                """
            }
        ]

        response = await model.predict(messages=messages)
        state.report = response.content
        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/resources/{url}")
async def delete_resource(url: str, state: ResearchState) -> ResearchState:
    """
    Delete a resource from the state
    """
    try:
        state.resources = [r for r in state.resources if r.url != url]
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
