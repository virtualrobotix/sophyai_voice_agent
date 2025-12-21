"""
OpenRouter LLM Wrapper for LiveKit Agents.
Provides OpenRouter API access with LiveKit LLM interface compatibility.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
import httpx
from loguru import logger

# LiveKit imports
try:
    from livekit.agents import llm
    from livekit.plugins import openai as openai_plugin
except ImportError:
    logger.warning("LiveKit agents not available")
    llm = None
    openai_plugin = None


class OpenRouterLLM:
    """
    OpenRouter LLM wrapper that uses the OpenAI-compatible API.
    
    OpenRouter provides access to many models through a unified API
    that is compatible with OpenAI's API format.
    """
    
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/gpt-3.5-turbo",
        site_url: str = None,
        site_name: str = "SophyAi Voice Agent"
    ):
        """
        Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key (or from OPENROUTER_API_KEY env)
            model: Model ID (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            site_url: Your site URL for OpenRouter headers
            site_name: Your site name for OpenRouter headers
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model
        self.site_url = site_url or os.getenv("SITE_URL", "https://sophyai.local")
        self.site_name = site_name
        
        if not self.api_key:
            logger.warning("OpenRouter API key not set")
    
    def get_livekit_llm(self) -> Optional[Any]:
        """
        Get a LiveKit-compatible LLM instance using the OpenAI plugin.
        
        OpenRouter uses an OpenAI-compatible API, so we can use the
        OpenAI plugin with a custom base URL.
        """
        if openai_plugin is None:
            logger.error("LiveKit OpenAI plugin not available")
            return None
        
        if not self.api_key:
            logger.error("OpenRouter API key not configured")
            return None
        
        # Create OpenAI plugin instance with OpenRouter base URL
        return openai_plugin.LLM(
            model=self.model,
            base_url=self.OPENROUTER_API_BASE,
            api_key=self.api_key,
        )
    
    async def get_available_models(
        self,
        search: str = None,
        sort_by: str = "name"
    ) -> List[Dict[str, Any]]:
        """
        Fetch available models from OpenRouter API.
        
        Args:
            search: Optional search filter
            sort_by: Sort by 'name', 'cost', 'cost_desc', or 'context'
            
        Returns:
            List of model dictionaries with id, name, pricing, etc.
        """
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                resp = await client.get(
                    f"{self.OPENROUTER_API_BASE}/models",
                    headers=headers,
                    timeout=30.0
                )
                
                if resp.status_code != 200:
                    logger.error(f"OpenRouter API error: {resp.status_code}")
                    return []
                
                data = resp.json()
                models = []
                
                for m in data.get("data", []):
                    pricing = m.get("pricing", {})
                    prompt_cost = float(pricing.get("prompt", 0)) * 1000000
                    completion_cost = float(pricing.get("completion", 0)) * 1000000
                    
                    model_info = {
                        "id": m["id"],
                        "name": m.get("name", m["id"]),
                        "description": m.get("description", ""),
                        "context_length": m.get("context_length", 0),
                        "prompt_cost": prompt_cost,
                        "completion_cost": completion_cost,
                        "total_cost": prompt_cost + completion_cost,
                    }
                    
                    # Filter by search
                    if search:
                        search_lower = search.lower()
                        if (search_lower not in model_info["id"].lower() and 
                            search_lower not in model_info["name"].lower()):
                            continue
                    
                    models.append(model_info)
                
                # Sort
                if sort_by == "cost":
                    models.sort(key=lambda x: x["total_cost"])
                elif sort_by == "cost_desc":
                    models.sort(key=lambda x: x["total_cost"], reverse=True)
                elif sort_by == "context":
                    models.sort(key=lambda x: x["context_length"], reverse=True)
                else:
                    models.sort(key=lambda x: x["name"].lower())
                
                return models
                
        except Exception as e:
            logger.error(f"Error fetching OpenRouter models: {e}")
            return []
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Optional[str]:
        """
        Direct chat completion (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text or None on error
        """
        if not self.api_key:
            logger.error("OpenRouter API key not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.OPENROUTER_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=60.0
                )
                
                if resp.status_code != 200:
                    logger.error(f"OpenRouter chat error: {resp.status_code} - {resp.text}")
                    return None
                
                data = resp.json()
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"Error in OpenRouter chat: {e}")
            return None


def get_openrouter_llm(
    api_key: str = None,
    model: str = None
) -> Optional[Any]:
    """
    Factory function to create an OpenRouter LLM for LiveKit.
    
    Args:
        api_key: OpenRouter API key
        model: Model ID to use
        
    Returns:
        LiveKit-compatible LLM instance or None
    """
    wrapper = OpenRouterLLM(
        api_key=api_key,
        model=model or "openai/gpt-3.5-turbo"
    )
    return wrapper.get_livekit_llm()


