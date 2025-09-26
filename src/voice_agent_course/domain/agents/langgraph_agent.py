"""LangGraph agent using prebuilt create_react_agent with Groq."""

from collections.abc import AsyncGenerator
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from voice_agent_course.domain.prompts.system_prompts import DEFAULT_SYSTEM_PROMPT

from ..tools.mock_tools import MOCK_TOOLS

load_dotenv()


class LangGraphAgent:
    """LangGraph agent using prebuilt create_react_agent with Groq model."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize LangGraph agent.

        Args:
            model: Groq model name
        """
        self.model = model
        self.max_history_turns = 3
        self.conversation_history: list[BaseMessage] = []
        self.tools = list(MOCK_TOOLS)  # Make a copy so we can modify it

        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=self.model,
            temperature=0.7,
        )

        # Create the prebuilt ReAct agent
        self._create_agent()

    def _create_agent(self):
        """Create or recreate the agent with current tools."""
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=DEFAULT_SYSTEM_PROMPT,
        )

    async def invoke(self, user_message: str) -> str:
        """
        Invoke the agent with a user message.

        Args:
            user_message: User's input message

        Returns:
            str: Agent's response
        """
        try:
            # Prepare the input with conversation history
            messages = self._get_recent_history() + [HumanMessage(content=user_message)]
            input_data = {"messages": messages}

            # Invoke the agent
            result = await self.agent.ainvoke(input_data)

            # Extract the final message content
            response_content = ""
            if result and "messages" in result and result["messages"]:
                # Look for the final AI message (not tool calls)
                for message in reversed(result["messages"]):
                    # Skip tool messages and tool calls - we want the final AI response
                    if (
                        hasattr(message, "content")
                        and message.content
                        and message.content.strip()
                        and not (hasattr(message, "tool_calls") and message.tool_calls)
                        and getattr(message, "type", None) != "tool"
                    ):
                        response_content = message.content
                        break

                # Fallback: return the last message if no proper AI message found
                if not response_content:
                    final_message = result["messages"][-1]
                    response_content = (
                        final_message.content if hasattr(final_message, "content") else str(final_message)
                    )

            if not response_content:
                response_content = "No response generated."

            # Update conversation history
            self._update_history(user_message, response_content)

            return response_content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Still update history for context
            self._update_history(user_message, error_msg)
            return error_msg

    async def stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Stream the agent's response with token-level streaming and conversation history.

        Args:
            user_message: User's input message

        Yields:
            str: Chunks of the agent's response (individual tokens)
        """
        response_content = ""

        try:
            # Prepare the input with conversation history
            messages = self._get_recent_history() + [HumanMessage(content=user_message)]
            input_data = {"messages": messages}

            # Use astream_events to get token-level streaming
            async for event in self.agent.astream_events(input_data, version="v1"):
                # Look for streaming events from the LLM
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        response_content += chunk.content
                        yield chunk.content

                # Also show tool usage
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    tool_msg = f"Using {tool_name} tool. "
                    print(f"🛠️ {tool_msg}")
                    # response_content += tool_msg
                    # yield tool_msg

            # Update conversation history with the complete response
            if response_content.strip():
                # print(f"🔍 {response_content.strip()}")
                self._update_history(user_message, response_content.strip())

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Still update history for context
            self._update_history(user_message, error_msg)
            yield error_msg

    def _get_recent_history(self) -> list[BaseMessage]:
        """Get recent conversation history (last N turns)."""
        if not self.conversation_history:
            return []

        # Keep last max_history_turns * 2 messages (each turn = human + ai message)
        max_messages = self.max_history_turns * 2
        return self.conversation_history[-max_messages:]

    def _update_history(self, user_message: str, ai_response: str):
        """Update conversation history with new turn."""
        # Add new messages
        self.conversation_history.extend([HumanMessage(content=user_message), AIMessage(content=ai_response)])

        # Trim conversation history if needed
        max_messages = self.max_history_turns * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def add_tool(self, tool) -> None:
        """Add a new tool to the agent dynamically."""
        if tool not in self.tools:
            self.tools.append(tool)
            self._create_agent()  # Recreate agent with new tools

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent by name."""
        original_count = len(self.tools)
        self.tools = [tool for tool in self.tools if tool.name != tool_name]

        if len(self.tools) < original_count:
            self._create_agent()  # Recreate agent without the removed tool
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "model": self.model,
            "tools_count": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "max_history_turns": self.max_history_turns,
        }
