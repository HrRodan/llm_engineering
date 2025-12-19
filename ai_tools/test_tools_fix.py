from ai_tools.tools import LLMQuery
import json
from PIL import Image


def test_append_tool_result():
    llm = LLMQuery()

    # Test dictionary output
    dict_output = {"key": "value", "list": [1, 2, 3]}
    tool_outputs = [{"tool_call_id": "call_123", "output": dict_output}]

    llm.append_tool_result(tool_outputs)

    last_msg = llm.chat_history[-1]
    assert last_msg["role"] == "tool"
    assert last_msg["tool_call_id"] == "call_123"
    # Content should be a JSON string
    assert last_msg["content"] == json.dumps(dict_output)
    print("Dictionary test passed!")

    # Test list output
    list_output = ["a", "b", "c"]
    tool_outputs = [{"tool_call_id": "call_456", "output": list_output}]
    llm.append_tool_result(tool_outputs)
    last_msg = llm.chat_history[-1]
    assert last_msg["content"] == json.dumps(list_output)
    print("List test passed!")

    # Test int output
    int_output = 42
    tool_outputs = [{"tool_call_id": "call_789", "output": int_output}]
    llm.append_tool_result(tool_outputs)
    last_msg = llm.chat_history[-1]
    assert last_msg["content"] == "42"
    print("Int test passed!")

    # Test Image output (mock)
    img_output = Image.new("RGB", (10, 10))
    tool_outputs = [{"tool_call_id": "call_img", "output": img_output}]
    llm.append_tool_result(tool_outputs)
    last_msg = llm.chat_history[-1]
    assert last_msg["content"] == "[Image created]"
    print("Image test passed!")


if __name__ == "__main__":
    test_append_tool_result()
