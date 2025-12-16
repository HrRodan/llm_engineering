import sys
import os
from ai_tools.tools import LLMQuery

# Mock the LLMQuery class to avoid making actual API calls
# We only need to test the get_chat_history_as_string method
llm = LLMQuery(system_prompt="You are a helpful assistant.")

# Manually populate chat history
llm.chat_history = [
    {"role": "user", "content": "What is the weather in London?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"location": "London"}'}}
        ],
    },
    {"role": "tool", "content": '{"temp": 15, "condition": "Cloudy"}'},
    {"role": "assistant", "content": "The weather in London is 15 degrees and cloudy."},
]

# Get the formatted string
history_str = llm.get_chat_history_as_string()

# Print the result
print(history_str)

# Assertions to verify the format
assert "**User**: What is the weather in London?" in history_str
assert '**Assistant Tool Call**: get_weather({"location": "London"})' in history_str
assert '**Tool Output**: {"temp": 15, "condition": "Cloudy"}' in history_str
assert "**Assistant**: The weather in London is 15 degrees and cloudy." in history_str

print("\nVerification Passed!")
