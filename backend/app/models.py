import os
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

def get_model(model_name: str) -> BaseChatModel:
    """Get the appropriate language model based on the model name"""
    if model_name == "openai":
        return ChatOpenAI(temperature=0, model="gpt-4")
    elif model_name == "anthropic":
        return ChatAnthropic(
            temperature=0,
            model_name="claude-3-opus-20240229",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif model_name == "google_genai":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    raise ValueError(f"Unknown model: {model_name}")