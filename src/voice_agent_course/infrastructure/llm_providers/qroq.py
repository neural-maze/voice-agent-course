from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq


class GroqService:
    """Simple Groq service using LangChain"""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq service.

        Args:
            model: Groq model name
        """
        self.model = model
        self.llm = ChatGroq(
            model=self.model,
            temperature=1.0,
            reasoning=False,
        )
        self.conversation_history: list[dict[str, str]] = []

    async def stream_chat(self, user_message: str, system_prompt: str | None = None) -> AsyncGenerator[str, None]:
        """
        Stream chat response from Groq.

        Args:
            user_message: User's input message
            system_prompt: Optional system prompt

        Yields:
            str: Chunks of the response text
        """
        # Build message history
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # Add conversation history
        for entry in self.conversation_history:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                messages.append(AIMessage(content=entry["content"]))

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        try:
            # Stream the response
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            yield f"Error: {str(e)}"

    async def chat(self, user_message: str, system_prompt: str | None = None) -> str:
        """
        Get complete chat response (non-streaming).

        Args:
            user_message: User's input message
            system_prompt: Optional system prompt

        Returns:
            str: Complete response text
        """
        response_chunks = []
        async for chunk in self.stream_chat(user_message, system_prompt):
            response_chunks.append(chunk)

        return "".join(response_chunks)

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics"""
        return {
            "model": self.model,
            "conversation_turns": len(self.conversation_history),
            "history_length": len(self.conversation_history),
        }
