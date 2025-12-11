"""
LLM Gateway Service
====================
Unified interface for multiple LLM providers with fallback and rate limiting.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Optional

import httpx

from .models import LLMConfig, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rpm, self.tokens + elapsed * (self.rpm / 60))
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            wait_time = (1 - self.tokens) * (60 / self.rpm)
            await asyncio.sleep(wait_time)
            self.tokens = 0
            return True


class LLMGateway:
    """
    Gateway for LLM interactions.
    Handles routing, rate limiting, fallback between providers, and caching.
    """

    DEFAULT_RATE_LIMITS = {
        LLMProvider.OPENAI: 60,
        LLMProvider.ANTHROPIC: 60,
        LLMProvider.AZURE_OPENAI: 120,
        LLMProvider.LOCAL: 1000,
    }

    def __init__(
        self,
        configs: list[LLMConfig],
        cache_service: Optional[Any] = None,
    ):
        self.configs = {c.provider: c for c in configs}
        self.primary_provider = configs[0].provider if configs else None
        self.cache = cache_service

        self._rate_limiters = {
            provider: RateLimiter(self.DEFAULT_RATE_LIMITS.get(provider, 60))
            for provider in self.configs.keys()
        }

        self._request_counts: dict[LLMProvider, int] = defaultdict(int)
        self._error_counts: dict[LLMProvider, int] = defaultdict(int)

        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _get_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider: LLMProvider,
        **kwargs,
    ) -> str:
        """Generate cache key for LLM request."""
        key_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "provider": provider.value,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"llm:{hashlib.sha256(key_str.encode()).hexdigest()}"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            provider: Specific provider to use (optional)
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text and metadata
        """
        target_provider = provider or self.primary_provider

        if not target_provider:
            raise ValueError("No LLM provider configured")

        config = self.configs.get(target_provider)
        if not config:
            raise ValueError(f"No configuration for provider {target_provider}")

        # Check cache
        if use_cache and self.cache:
            cache_key = self._get_cache_key(prompt, system_prompt, target_provider, **kwargs)
            cached = await self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for LLM request")
                cached["cached"] = True
                return LLMResponse(**cached)

        # Rate limit
        await self._rate_limiters[target_provider].acquire()

        try:
            self._request_counts[target_provider] += 1

            if target_provider == LLMProvider.OPENAI:
                response = await self._generate_openai(prompt, system_prompt, config, **kwargs)
            elif target_provider == LLMProvider.ANTHROPIC:
                response = await self._generate_anthropic(prompt, system_prompt, config, **kwargs)
            elif target_provider == LLMProvider.AZURE_OPENAI:
                response = await self._generate_azure(prompt, system_prompt, config, **kwargs)
            elif target_provider == LLMProvider.LOCAL:
                response = await self._generate_local(prompt, system_prompt, config, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {target_provider}")

            # Cache successful response
            if use_cache and self.cache:
                await self.cache.set(cache_key, response.model_dump(), ttl=3600)

            return response

        except Exception as e:
            logger.error(f"LLM generation failed with {target_provider}: {e}")
            self._error_counts[target_provider] += 1

            # Try fallback providers
            for fallback_provider, fallback_config in self.configs.items():
                if fallback_provider != target_provider:
                    try:
                        logger.info(f"Trying fallback provider: {fallback_provider}")
                        return await self.generate(
                            prompt,
                            system_prompt,
                            provider=fallback_provider,
                            use_cache=use_cache,
                            **kwargs
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback {fallback_provider} also failed: {fallback_error}")
                        continue

            raise

    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        config: LLMConfig,
        **kwargs,
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not configured")

        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_body = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "temperature": kwargs.get("temperature", config.temperature),
            "top_p": kwargs.get("top_p", config.top_p),
        }

        if kwargs.get("response_format") == "json":
            request_body["response_format"] = {"type": "json_object"}

        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data["model"],
            provider=LLMProvider.OPENAI.value,
            usage={
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "completion_tokens": data["usage"]["completion_tokens"],
                "total_tokens": data["usage"]["total_tokens"],
            },
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        config: LLMConfig,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Anthropic API."""
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not configured")

        client = await self._get_client()

        request_body = {
            "model": config.model_name,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            request_body["system"] = system_prompt

        if kwargs.get("temperature") is not None:
            request_body["temperature"] = kwargs["temperature"]
        elif config.temperature != 0.1:
            request_body["temperature"] = config.temperature

        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data["content"][0]["text"],
            model=data["model"],
            provider=LLMProvider.ANTHROPIC.value,
            usage={
                "prompt_tokens": data["usage"]["input_tokens"],
                "completion_tokens": data["usage"]["output_tokens"],
                "total_tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
            },
            finish_reason=data.get("stop_reason"),
        )

    async def _generate_azure(
        self,
        prompt: str,
        system_prompt: Optional[str],
        config: LLMConfig,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Azure OpenAI."""
        api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI API key or endpoint not configured")

        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        url = f"{endpoint}/openai/deployments/{config.model_name}/chat/completions?api-version={api_version}"

        request_body = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "temperature": kwargs.get("temperature", config.temperature),
            "top_p": kwargs.get("top_p", config.top_p),
        }

        response = await client.post(
            url,
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", config.model_name),
            provider=LLMProvider.AZURE_OPENAI.value,
            usage={
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "completion_tokens": data["usage"]["completion_tokens"],
                "total_tokens": data["usage"]["total_tokens"],
            },
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    async def _generate_local(
        self,
        prompt: str,
        system_prompt: Optional[str],
        config: LLMConfig,
        **kwargs,
    ) -> LLMResponse:
        """Generate using local model (vLLM, Ollama, etc.)."""
        endpoint = config.endpoint or os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:8080")

        client = await self._get_client()

        # Support OpenAI-compatible local endpoints (vLLM, LocalAI, etc.)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_body = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "temperature": kwargs.get("temperature", config.temperature),
        }

        response = await client.post(
            f"{endpoint}/v1/chat/completions",
            json=request_body,
            timeout=config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", config.model_name),
            provider=LLMProvider.LOCAL.value,
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate and parse JSON response from LLM."""
        # Add JSON instruction to prompt
        json_prompt = prompt + "\n\nRespond with valid JSON only, no additional text."

        response = await self.generate(
            json_prompt,
            system_prompt,
            provider,
            response_format="json",
            **kwargs,
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            text = response.text
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(text[start_idx:end_idx])
                except json.JSONDecodeError:
                    pass

            # Try array
            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1

            if start_idx != -1 and end_idx > start_idx:
                try:
                    return {"items": json.loads(text[start_idx:end_idx])}
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON from LLM response: {text[:500]}")
            raise ValueError("LLM response is not valid JSON")

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        return {
            "request_counts": dict(self._request_counts),
            "error_counts": dict(self._error_counts),
            "providers_configured": list(self.configs.keys()),
            "primary_provider": self.primary_provider,
        }
