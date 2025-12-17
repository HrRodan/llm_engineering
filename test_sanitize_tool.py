from ai_tools.tools import LLMQuery
from PIL import Image
import io


def test_sanitization():
    llm = LLMQuery()

    # Create fake outputs
    img = Image.new("RGB", (10, 10))
    audio = b"\x00\x01"
    obj = {"some": "data"}
    text = "Normal text"

    tool_outputs = [
        {"tool_call_id": "1", "output": img},
        {"tool_call_id": "2", "output": audio},
        {"tool_call_id": "3", "output": obj},
        {"tool_call_id": "4", "output": text},
    ]

    llm.append_tool_result(tool_outputs)

    history = llm.chat_history
    print(history)

    assert history[0]["content"] == "[Image created]"
    assert history[1]["content"] == "[Audio created]"
    assert history[2]["content"] == "[dict object created]"
    assert history[3]["content"] == "Normal text"


if __name__ == "__main__":
    test_sanitization()
