"""
Ollama Client for Layer 2 - Voice AI Pipeline
Handles both embeddings and LLM inference locally.
"""

import requests
import json
from typing import Dict, List, Optional, Generator, Any

from src.config import settings

# Ollama server configuration
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
EMBED_MODEL = settings.EMBED_MODEL
LLM_MODEL = settings.LLM_MODEL


def embed_text(text: str) -> List[float]:
    """
    Generate embedding for input text using Ollama's embedding model.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    
    try:
        # Keep embedding path fast-fail to avoid delaying real-time voice turns.
        response = requests.post(url, json=payload, timeout=3)
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Embedding request failed: {e}")
        raise


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False
) -> str:
    """
    Generate LLM response using Ollama's Llama model.
    
    Args:
        prompt: User prompt/query
        system_prompt: Optional system instructions
        temperature: Creativity level (0.0-1.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream response
        
    Returns:
        Generated text response
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Build the full prompt with system context if provided
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    
    payload = {
        "model": LLM_MODEL,
        "prompt": full_prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["response"]
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM request failed: {e}")
        raise


def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False
) -> str:
    """
    Chat-style LLM interaction with conversation history.
    
    Args:
        messages: List of {"role": "user/assistant/system", "content": "..."}
        temperature: Creativity level (0.0-1.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream response
        
    Returns:
        Generated assistant response
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Chat request failed: {e}")
        raise


def check_ollama_status() -> Dict[str, Any]:
    """
    Check if Ollama server is running and models are available.
    
    Returns:
        Status dict with server_running and available_models
    """
    status = {
        "server_running": False,
        "available_models": [],
        "embed_model_ready": False,
        "llm_model_ready": False
    }
    
    try:
        # Check server
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        status["server_running"] = True
        
        # Get available models
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        status["available_models"] = models
        
        # Check required models
        status["embed_model_ready"] = any(EMBED_MODEL in m for m in models)
        status["llm_model_ready"] = any(LLM_MODEL in m for m in models)
        
    except requests.exceptions.RequestException:
        pass
    
    return status


def generate_response_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512
) -> Generator[str, None, None]:
    """
    Generate LLM response as a stream of tokens.
    
    Args:
        prompt: User prompt
        system_prompt: System instructions
        temperature: Creativity
        max_tokens: Max output length
        
    Yields:
        Token strings
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    
    payload = {
        "model": LLM_MODEL,
        "prompt": full_prompt,
        "stream": True, # ENABLE STREAMING
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        pass
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM Stream failed: {e}")
        yield f"Error: {str(e)}"

def chat_stream(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512
) -> Generator[str, None, None]:
    """
    Chat-style LLM interaction with streaming.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        pass
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Chat Stream failed: {e}")
        yield f"Error: {str(e)}"
