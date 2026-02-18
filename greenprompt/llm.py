"""
LLM Provider wrappers for GreenPES.

Unified interface for multiple LLM providers.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini API wrapper (free tier: 60 RPM)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        start = time.time()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"max_output_tokens": max_tokens}
        )

        latency = (time.time() - start) * 1000

        # Get token counts from usage_metadata
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)

        # Fallback to estimation if not available
        if input_tokens is None:
            input_tokens = int(len(prompt.split()) * 1.3)
        if output_tokens is None:
            output_tokens = int(len(response.text.split()) * 1.3)

        return LLMResponse(
            text=response.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            model=self.model_name
        )


class GroqProvider(LLMProvider):
    """Groq API wrapper for open-source models (free tier: 30 RPM)."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency,
            model=self.model
        )


class OpenAIProvider(LLMProvider):
    """OpenAI API wrapper."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency,
            model=self.model
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API wrapper."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        start = time.time()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            model=self.model
        )


class TogetherProvider(LLMProvider):
    """Together.ai API wrapper for open-source models."""

    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        try:
            from together import Together
        except ImportError:
            raise ImportError("Install together: pip install together")

        self.client = Together(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency,
            model=self.model
        )


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""

    def __init__(self, response_text: str = "This is a mock response.", tokens_per_word: float = 1.3):
        self.response_text = response_text
        self.tokens_per_word = tokens_per_word

    def generate(self, prompt: str, max_tokens: int = 500) -> LLMResponse:
        input_tokens = int(len(prompt.split()) * self.tokens_per_word)
        output_tokens = int(len(self.response_text.split()) * self.tokens_per_word)

        return LLMResponse(
            text=self.response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=10.0,
            model="mock"
        )
