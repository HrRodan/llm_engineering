from ai_tools.tools import handle_tool_call
import json


def test_func(a, b):
    return a + b


tool_calls = [
    {
        "id": "call_123",
        "function": {"name": "test_func", "arguments": json.dumps({"a": 1, "b": 2})},
    }
]

response = handle_tool_call(tool_calls, [test_func])
print(response)
assert response[0]["name"] == "test_func"
assert response[0]["arguments"] == {"a": 1, "b": 2}
