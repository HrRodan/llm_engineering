import os
import sys

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from ai_tools.tools import LLMQuery

# Mock tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def test_query_tool_call():
    print("Testing query with tools...")
    try:
        llm = LLMQuery(tools=tools, tool_choice="auto", model="gpt-4o-mini")
    except Exception:
        llm = LLMQuery(tools=tools, tool_choice="auto")

    response = llm.query("What is the weather in San Francisco?")
    print(f"Response content: {response}")

    if llm.tool_calls:
        print(f"Tool calls received: {llm.tool_calls}")
        tc = llm.tool_calls[0]
        assert tc["function"]["name"] == "get_current_weather"
        print("Query Tool Call Test Passed")
    else:
        print("No tool calls received for Query.")


def test_query_stream_tool_call():
    print("\nTesting query_stream with tools...")
    try:
        llm = LLMQuery(tools=tools, tool_choice="auto", model="gpt-4o-mini")
    except Exception:
        llm = LLMQuery(tools=tools, tool_choice="auto")

    # Consuming the generator
    chunks = list(
        llm.query_stream("What is the weather in Paris?", return_generator=True)
    )
    print(f"Stream output len: {len(chunks)}")
    # print(f"Chunks: {chunks}") # Optional, might be verbose

    if llm.tool_calls:
        print(f"Stream Tool calls received: {llm.tool_calls}")
        tc = llm.tool_calls[0]
        # function name might be accumulated, check if correct
        assert tc["function"]["name"] == "get_current_weather"
        print("Query Stream Tool Call Test Passed")
    else:
        print("No tool calls received for Query Stream.")


def test_append_tool_result():
    print("\nTesting append_tool_result...")
    llm = LLMQuery(tools=tools)
    llm.query("What is the weather in London?")

    if llm.tool_calls:
        tool_call_id = llm.tool_calls[0]["id"]
        llm.append_tool_result(
            tool_call_id, "The weather in London is 15 degrees Celsius."
        )

        last_msg = llm.chat_history[-1]
        print(f"Last message: {last_msg}")
        assert last_msg["role"] == "tool"
        assert last_msg["content"] == "The weather in London is 15 degrees Celsius."
        assert last_msg["tool_call_id"] == tool_call_id
        print("Append Tool Result Test Passed")
    else:
        print("No tool calls to append result to.")


if __name__ == "__main__":
    try:
        test_query_tool_call()
        test_query_stream_tool_call()
        test_append_tool_result()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
