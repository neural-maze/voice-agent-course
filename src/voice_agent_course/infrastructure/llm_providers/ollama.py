import logging

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class OllamaClient:
    """Lightweight wrapper for ChatOllama with tool capability checking"""

    # NOTE: see https://ollama.com/search?c=tools
    TOOL_CAPABLE_MODEL_FAMILIES = {
        "gemma3-tools",
        "PetrosStav/gemma3-tools",
        "mistral",
        "llama3.2",
        "qwen3",
    }

    def __init__(self, model: str = "qwen3:1.7b", temperature: float = 1.0, **kwargs):
        """
        Initialize Ollama client wrapper.

        Args:
            model: Ollama model name
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to ChatOllama
        """
        self.model = model
        self.llm = ChatOllama(
            model=self.model,
            temperature=temperature,
            reasoning=False,
            **kwargs,
        )

        # Warn if model doesn't support tools
        self._check_tool_capability()

    def _check_tool_capability(self) -> None:
        """Check if the current model supports tool calling and warn if not."""
        if not self.supports_tools():
            logger.warning(
                f"Model '{self.model}' may not support tool calling. "
                f"Tool-capable model families include: {', '.join(sorted(self.TOOL_CAPABLE_MODEL_FAMILIES))}"
            )

    def supports_tools(self) -> bool:
        """Check if the current model supports tool calling."""
        # Extract model family (everything before the first ":")
        model_family = self.model.split(":")[0]
        return model_family in self.TOOL_CAPABLE_MODEL_FAMILIES

    # Expose LangChain methods directly
    def __getattr__(self, name):
        """Delegate method calls to the underlying ChatOllama instance"""
        return getattr(self.llm, name)
