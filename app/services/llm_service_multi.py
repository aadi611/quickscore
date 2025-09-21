"""
Multi-provider LLM service supporting both OpenAI and Groq APIs.
"""
import logging
from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime

import openai
from openai import AsyncOpenAI
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class MultiLLMService:
    """Service for LLM API interactions supporting both OpenAI and Groq."""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
            self.backup_model = "gpt-3.5-turbo-1106"
        elif self.provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is required when using Groq provider")
            self.groq_api_key = settings.GROQ_API_KEY
            self.model = settings.GROQ_MODEL
            self.backup_model = "llama3-8b-8192"  # Groq backup model
            self.groq_base_url = "https://api.groq.com/openai/v1"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        logger.info(f"Initialized LLM service with provider: {self.provider}, model: {self.model}")

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        response_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a completion using the configured LLM provider."""
        try:
            if self.provider == "openai":
                return await self._generate_openai_completion(
                    system_prompt, user_prompt, max_tokens, temperature, response_format
                )
            elif self.provider == "groq":
                return await self._generate_groq_completion(
                    system_prompt, user_prompt, max_tokens, temperature, response_format
                )
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            return await self._retry_with_backup(system_prompt, user_prompt, max_tokens, temperature, response_format)

    async def _generate_openai_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        response_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate completion using OpenAI API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add JSON response format if requested
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**kwargs)
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "model_used": response.model,
            "provider": "openai",
            "tokens_used": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason
        }

    async def _generate_groq_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        response_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate completion using Groq API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Groq doesn't support response_format yet, so we'll handle JSON in the prompt
        if response_format == "json":
            payload["messages"][0]["content"] += "\n\nPlease respond with valid JSON only."
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.groq_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "model_used": result["model"],
                "provider": "groq",
                "tokens_used": {
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "total_tokens": result["usage"]["total_tokens"]
                },
                "finish_reason": result["choices"][0]["finish_reason"]
            }

    async def _retry_with_backup(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        response_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retry with backup model if primary fails."""
        try:
            logger.info(f"Retrying with backup model: {self.backup_model}")
            
            if self.provider == "openai":
                old_model = self.model
                self.model = self.backup_model
                try:
                    result = await self._generate_openai_completion(
                        system_prompt, user_prompt, max_tokens, temperature, response_format
                    )
                    result["fallback_used"] = True
                    return result
                finally:
                    self.model = old_model
                    
            elif self.provider == "groq":
                old_model = self.model
                self.model = self.backup_model
                try:
                    result = await self._generate_groq_completion(
                        system_prompt, user_prompt, max_tokens, temperature, response_format
                    )
                    result["fallback_used"] = True
                    return result
                finally:
                    self.model = old_model
                    
        except Exception as backup_error:
            logger.error(f"Backup model also failed: {backup_error}")
            return {
                "success": False,
                "error": f"Both primary and backup models failed. Primary: {str(backup_error)}",
                "model_used": self.backup_model,
                "provider": self.provider
            }

    async def batch_completions(
        self,
        prompts: List[Dict[str, str]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple prompts concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_prompt(prompt_data):
            async with semaphore:
                return await self.generate_completion(
                    system_prompt=prompt_data["system"],
                    user_prompt=prompt_data["user"],
                    max_tokens=prompt_data.get("max_tokens", 2000),
                    temperature=prompt_data.get("temperature", 0.3),
                    response_format=prompt_data.get("response_format")
                )
        
        tasks = [process_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "prompt_index": i
                })
            else:
                result["prompt_index"] = i
                processed_results.append(result)
        
        return processed_results


# Create a compatibility layer to maintain existing imports
class LLMService(MultiLLMService):
    """Backward compatibility wrapper for existing code."""
    pass