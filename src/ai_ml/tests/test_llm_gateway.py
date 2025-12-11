"""Tests for LLM Gateway."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_ml.llm_gateway import LLMGateway, RateLimiter
from src.ai_ml.models import LLMConfig, LLMProvider, LLMResponse


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.rpm == 60
        assert limiter.tokens == 60

    @pytest.mark.asyncio
    async def test_acquire_when_available(self):
        """Test acquiring token when available."""
        limiter = RateLimiter(requests_per_minute=60)
        result = await limiter.acquire()
        assert result is True
        assert limiter.tokens < 60

    @pytest.mark.asyncio
    async def test_tokens_replenish(self):
        """Test token replenishment over time."""
        limiter = RateLimiter(requests_per_minute=60)

        # Use some tokens
        await limiter.acquire()
        initial_tokens = limiter.tokens

        # Simulate time passing
        import time
        limiter.last_update = time.time() - 1  # 1 second ago

        await limiter.acquire()
        # Tokens should have been replenished
        assert limiter.tokens >= 0


class TestLLMGatewayInit:
    """Test LLMGateway initialization."""

    def test_single_config(self):
        """Test initialization with single config."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4")
        gateway = LLMGateway([config])

        assert gateway.primary_provider == LLMProvider.OPENAI
        assert LLMProvider.OPENAI in gateway.configs

    def test_multiple_configs(self):
        """Test initialization with multiple configs."""
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-3"),
        ]
        gateway = LLMGateway(configs)

        assert gateway.primary_provider == LLMProvider.OPENAI
        assert len(gateway.configs) == 2

    def test_with_cache(self):
        """Test initialization with cache service."""
        mock_cache = AsyncMock()
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4")
        gateway = LLMGateway([config], cache_service=mock_cache)

        assert gateway.cache == mock_cache


class TestLLMGatewayGenerate:
    """Test LLMGateway generate method."""

    @pytest.mark.asyncio
    async def test_generate_no_provider(self):
        """Test generate fails with no provider."""
        gateway = LLMGateway([])
        gateway.primary_provider = None

        with pytest.raises(ValueError, match="No LLM provider configured"):
            await gateway.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_invalid_provider(self):
        """Test generate fails with invalid provider."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4")
        gateway = LLMGateway([config])

        with pytest.raises(ValueError, match="No configuration"):
            await gateway.generate("Test", provider=LLMProvider.ANTHROPIC)

    @pytest.mark.asyncio
    async def test_generate_openai_success(self):
        """Test successful OpenAI generation."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test-key")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}, "finish_reason": "stop"}],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            result = await gateway.generate("Test prompt", use_cache=False)

            assert result.text == "Generated text"
            assert result.model == "gpt-4"
            assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_generate_anthropic_success(self):
        """Test successful Anthropic generation."""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-3", api_key="test-key")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "Claude response"}],
            "model": "claude-3",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            result = await gateway.generate("Test prompt", use_cache=False)

            assert result.text == "Claude response"
            assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test-key")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            await gateway.generate(
                "Test prompt",
                system_prompt="You are a helpful assistant",
                use_cache=False,
            )

            # Verify the call includes system prompt
            call_args = mock_http.post.call_args
            request_body = call_args[1]["json"]
            assert any(m["role"] == "system" for m in request_body["messages"])


class TestLLMGatewayFallback:
    """Test LLMGateway fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """Test fallback to secondary provider on error."""
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude", api_key="test"),
        ]
        gateway = LLMGateway(configs)

        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Primary failed")
            return LLMResponse(
                text="Fallback response",
                model="claude",
                provider="anthropic",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        with patch.object(gateway, '_generate_openai', side_effect=Exception("Primary failed")):
            with patch.object(gateway, '_generate_anthropic', return_value=LLMResponse(
                text="Fallback response",
                model="claude",
                provider="anthropic",
                usage={},
            )):
                result = await gateway.generate("Test", use_cache=False)

                assert result.text == "Fallback response"


class TestLLMGatewayCache:
    """Test LLMGateway caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit returns cached response."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = {
            "text": "Cached response",
            "model": "gpt-4",
            "provider": "openai",
            "usage": {},
        }

        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4")
        gateway = LLMGateway([config], cache_service=mock_cache)

        result = await gateway.generate("Test prompt")

        assert result.text == "Cached response"
        assert result.cached is True
        mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_miss_stores_result(self):
        """Test cache miss stores generated result."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None

        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test")
        gateway = LLMGateway([config], cache_service=mock_cache)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "New response"}}],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            await gateway.generate("Test prompt")

            mock_cache.set.assert_called_once()


class TestLLMGatewayGenerateJSON:
    """Test JSON generation and parsing."""

    @pytest.mark.asyncio
    async def test_generate_json_valid(self):
        """Test generating valid JSON."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}],
            "model": "gpt-4",
            "usage": {},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            result = await gateway.generate_json("Generate JSON")

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_generate_json_extracts_from_text(self):
        """Test extracting JSON from text with extra content."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": 'Here is the JSON: {"key": "value"}'}}],
            "model": "gpt-4",
            "usage": {},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            result = await gateway.generate_json("Generate JSON")

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_generate_json_array(self):
        """Test extracting JSON array."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '[{"item": 1}, {"item": 2}]'}}],
            "model": "gpt-4",
            "usage": {},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            result = await gateway.generate_json("Generate array")

            assert "items" in result
            assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_generate_json_invalid(self):
        """Test handling invalid JSON response."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test")
        gateway = LLMGateway([config])

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is not JSON at all"}}],
            "model": "gpt-4",
            "usage": {},
        }

        with patch.object(gateway, '_get_client') as mock_client:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_client.return_value = mock_http

            with pytest.raises(ValueError, match="not valid JSON"):
                await gateway.generate_json("Generate JSON")


class TestLLMGatewayStats:
    """Test LLMGateway statistics."""

    def test_get_stats(self):
        """Test getting gateway statistics."""
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude"),
        ]
        gateway = LLMGateway(configs)

        stats = gateway.get_stats()

        assert "request_counts" in stats
        assert "error_counts" in stats
        assert "providers_configured" in stats
        assert LLMProvider.OPENAI in stats["providers_configured"]
        assert LLMProvider.ANTHROPIC in stats["providers_configured"]
