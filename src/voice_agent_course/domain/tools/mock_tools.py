"""Mock tools for testing voice agent with async operations."""

import asyncio
import random

from langchain_core.tools import tool


@tool
async def get_random_number() -> str:
    """
    Get a random number between 1-100. Slow operation.

    Returns:
        str: A random number between 1 and 100
    """
    # Simulate a slow operation (e.g., API call, database query, computation)
    # await asyncio.sleep(5)

    # Generate random number
    number = random.randint(1, 100)

    return f"Here's your random number: {number}"


@tool
async def calculate_fibonacci(n: int) -> str:
    """
    Calculate the nth Fibonacci. Slow operation.

    Args:
        n: The position in the Fibonacci sequence (1-20 only for safety)

    Returns:
        str: The nth Fibonacci number
    """
    # Convert to int if it's a string (LangChain sometimes passes string arguments)
    try:
        n = int(n)
    except (ValueError, TypeError):
        return "Please provide a valid number."

    # Validate input
    if n < 1 or n > 20:
        return "Please provide a number between 1 and 20."

    # Simulate processing time
    # await asyncio.sleep(3)

    # Calculate Fibonacci
    def fib(num: int) -> int:
        if num <= 2:
            return 1
        return fib(num - 1) + fib(num - 2)

    result = fib(n)
    return f"The {n}th Fibonacci number is {result}"


@tool
async def get_weather(city: str) -> str:
    """
    Get weather information for a city. Uses external API call.

    Args:
        city: The city to get weather for

    Returns:
        str: Weather information
    """
    # Simulate API delay
    # await asyncio.sleep(0.5)

    # Mock weather data
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy"]
    temperatures = list(range(15, 35))  # 15-35°C

    condition = random.choice(conditions)
    temp = random.choice(temperatures)

    return f"The weather in {city} is currently {condition} with a temperature of {temp}°C"


# Tool registry for easy access
MOCK_TOOLS = [
    get_random_number,
    calculate_fibonacci,
    get_weather,
]
