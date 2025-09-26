from langchain_groq import ChatGroq


class GroqClient:
    """Lightweight wrapper for ChatGroq"""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 1.0, **kwargs):
        """
        Initialize Groq client wrapper.

        Args:
            model: Groq model name
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to ChatGroq
        """
        self.model = model
        self.llm = ChatGroq(model=self.model, temperature=temperature, reasoning=False, **kwargs)

    # Expose LangChain methods directly
    def __getattr__(self, name):
        """Delegate method calls to the underlying ChatGroq instance"""
        return getattr(self.llm, name)
