"""LLM module - Ollama and OpenRouter support"""
from .ollama_llm import OllamaLLM
from .openrouter_llm import OpenRouterLLM, get_openrouter_llm

__all__ = ["OllamaLLM", "OpenRouterLLM", "get_openrouter_llm"]



