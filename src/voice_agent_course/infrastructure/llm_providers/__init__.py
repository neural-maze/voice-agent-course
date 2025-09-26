"""LLM provider factory and registry."""

from enum import Enum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


class LLMProvider(str, Enum):
    """Available LLM providers."""

    GROQ = "groq"
    OLLAMA = "ollama"


class LLMProviderFactory:
    """Factory for creating LLM instances with different providers."""

    # Default models for each provider
    DEFAULT_MODELS = {
        LLMProvider.GROQ: "meta-llama/llama-4-scout-17b-16e-instruct",
        LLMProvider.OLLAMA: "qwen3:4b-instruct-2507-q4_K_M",
    }

    # Common models for each provider
    COMMON_MODELS = {
        LLMProvider.GROQ: [
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "moonshotai/kimi-k2-instruct-0905",
        ],
        LLMProvider.OLLAMA: [
            "qwen3:4b-instruct-2507-q4_K_M",
            "mistral:7b",
        ],
    }

    @classmethod
    def create_llm(
        self, provider: LLMProvider | str, model: str | None = None, temperature: float = 0.7, **kwargs: Any
    ) -> BaseChatModel:
        """
        Create an LLM instance for the specified provider and model.

        Args:
            provider: LLM provider (groq, ollama)
            model: Model name (uses default if None)
            temperature: Model temperature
            **kwargs: Additional provider-specific arguments

        Returns:
            BaseChatModel: Configured LLM instance

        Raises:
            ValueError: If provider is not supported
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        # Use default model if none specified
        if model is None:
            model = self.DEFAULT_MODELS[provider]

        if provider == LLMProvider.GROQ:
            return ChatGroq(model=model, temperature=temperature, **kwargs)
        elif provider == LLMProvider.OLLAMA:
            return ChatOllama(model=model, temperature=temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return [provider.value for provider in LLMProvider]

    @classmethod
    def get_common_models(cls, provider: LLMProvider | str) -> list[str]:
        """Get common models for a provider."""
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        return cls.COMMON_MODELS.get(provider, [])

    @classmethod
    def get_default_model(cls, provider: LLMProvider | str) -> str:
        """Get default model for a provider."""
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        return cls.DEFAULT_MODELS[provider]


def create_llm(provider: str, model: str | None = None, temperature: float = 0.7, **kwargs: Any) -> BaseChatModel:
    """Convenience function to create an LLM instance."""
    return LLMProviderFactory.create_llm(provider, model, temperature, **kwargs)


def get_available_providers() -> list[str]:
    """Get list of available providers."""
    return LLMProviderFactory.get_available_providers()


def get_common_models(provider: str) -> list[str]:
    """Get common models for a provider."""
    return LLMProviderFactory.get_common_models(provider)
