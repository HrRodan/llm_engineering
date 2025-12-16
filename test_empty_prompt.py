import sys
import os

sys.path.append(os.getcwd())
try:
    from ai_tools.tools import LLMQuery
except ImportError:
    pass


def test_empty_prompt():
    print("Testing empty user prompt...")
    llm = LLMQuery(system_prompt="Say 'Hello'", model="gemini-flash-lite-latest")

    # Test 1: Explicit empty string
    print("\nTest 1: Explicit empty string query(user_prompt='')")
    try:
        response = llm.query(user_prompt="")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Query('') failed: {e}")

    # Test 2: Implicit None with default args query()
    # Note: tools.py currently handles None by NOT adding a user message.
    print("\nTest 2: Implicit None query()")
    try:
        response = llm.query()
        print(f"Response: {response}")
    except Exception as e:
        print(f"Query() failed: {e}")

    # Test 3: None with History
    print("\nTest 3: Implicit None query() WITH history")
    # Manually inject history
    llm.chat_history.append({"role": "user", "content": "Hi"})
    llm.chat_history.append({"role": "assistant", "content": "Hello there."})

    try:
        # We need use_history=True for it to be included
        response = llm.query(use_history=True)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Query() with history failed: {e}")

    print("\nFinished tests.")


if __name__ == "__main__":
    test_empty_prompt()
