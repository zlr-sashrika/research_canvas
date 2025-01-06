from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.agent import graph
from copilotkit.runtime import copilotkit_fastapi_endpoint

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route(
    "/copilotkit", 
    copilotkit_fastapi_endpoint(agents=[{
        "name": "research_agent",
        "description": "Research assistant agent",
        "graph": graph
    }])
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}