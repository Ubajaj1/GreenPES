"""Tests for LLM providers."""
import os
import pytest
from greenprompt.llm import (
    OpenAIProvider, AnthropicProvider, TogetherProvider,
    GroqProvider, GeminiProvider, LLMResponse
)


class TestOpenAIProvider:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No API key")
    def test_generate_returns_llm_response(self):
        provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
        response = provider.generate("Say hello", max_tokens=10)

        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0
        assert "gpt-4o-mini" in response.model


class TestAnthropicProvider:
    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No API key")
    def test_generate_returns_llm_response(self):
        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = provider.generate("Say hello", max_tokens=10)

        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0
        assert "claude" in response.model  # claude-haiku-4-5-20251001


class TestTogetherProvider:
    @pytest.mark.skipif(not os.environ.get("TOGETHER_API_KEY"), reason="No API key")
    def test_generate_returns_llm_response(self):
        provider = TogetherProvider(api_key=os.environ["TOGETHER_API_KEY"])
        response = provider.generate("Say hello", max_tokens=10)

        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0


class TestGroqProvider:
    @pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No API key")
    def test_llama_8b(self):
        provider = GroqProvider(
            api_key=os.environ["GROQ_API_KEY"],
            model="llama-3.1-8b-instant"
        )
        response = provider.generate("Say hello", max_tokens=10)
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    @pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No API key")
    def test_llama_70b(self):
        provider = GroqProvider(
            api_key=os.environ["GROQ_API_KEY"],
            model="llama-3.3-70b-versatile"
        )
        response = provider.generate("Say hello", max_tokens=10)
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    @pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No API key")
    def test_qwen3_32b(self):
        provider = GroqProvider(
            api_key=os.environ["GROQ_API_KEY"],
            model="qwen/qwen3-32b"
        )
        response = provider.generate("Say hello", max_tokens=10)
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    @pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No API key")
    def test_kimi_k2(self):
        # Moonshot AI model — different lab/family from Meta/Alibaba, adds paper diversity
        # Replaces decommissioned mixtral-8x7b-32768 and gemma2-9b-it
        provider = GroqProvider(
            api_key=os.environ["GROQ_API_KEY"],
            model="moonshotai/kimi-k2-instruct"
        )
        response = provider.generate("Say hello", max_tokens=10)
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0


class TestGeminiProvider:
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No API key")
    def test_gemini_flash(self):
        provider = GeminiProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model="gemini-2.0-flash"
        )
        response = provider.generate("Say hello", max_tokens=10)
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No API key")
    @pytest.mark.skipif(not os.environ.get("GEMINI_BILLING_ENABLED"), reason="gemini-2.5-pro requires billing (free tier quota=0)")
    def test_gemini_pro(self):
        # Requires Google AI Studio billing enabled — free tier limit is 0 for this model
        provider = GeminiProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model="gemini-2.5-pro"
        )
        response = provider.generate("Say hello", max_tokens=10)
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
