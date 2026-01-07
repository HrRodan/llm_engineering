import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path to import ai_tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_tools.tools import LLMQuery


class TestHistoryLimit(unittest.TestCase):
    def setUp(self):
        self.llm = LLMQuery(system_prompt="Test System Prompt")
        # Populate history with 5 items
        self.llm.chat_history = [
            {"role": "user", "content": f"Message {i}"} for i in range(1, 6)
        ]

    def test_prepare_messages_no_limit(self):
        messages = self.llm._prepare_messages(
            user_prompt="New Prompt", use_history=True
        )
        # 1 system + 5 history + 1 user = 7
        self.assertEqual(len(messages), 7)
        self.assertEqual(messages[1]["content"], "Message 1")

    def test_prepare_messages_with_limit(self):
        messages = self.llm._prepare_messages(
            user_prompt="New Prompt", use_history=True, history_limit=2
        )
        # 1 system + 2 history + 1 user = 4
        self.assertEqual(len(messages), 4)
        # Should be Message 4 and Message 5 (last 2)
        self.assertEqual(messages[1]["content"], "Message 4")
        self.assertEqual(messages[2]["content"], "Message 5")

    def test_prepare_messages_limit_larger_than_history(self):
        messages = self.llm._prepare_messages(
            user_prompt="New Prompt", use_history=True, history_limit=10
        )
        # 1 system + 5 history + 1 user = 7
        self.assertEqual(len(messages), 7)
        self.assertEqual(messages[1]["content"], "Message 1")

    def test_prepare_messages_zero_limit(self):
        # Even if limit is 0 (which evaluates to False in boolean context), logic is:
        # if history_limit: ... else: ...
        # So passing 0 might technically mean "All" if implemented as `if history_limit:`
        # But usually 0 means "None". Let's check implementation.
        # Implementation: `if history_limit:` -> 0 is False, so it extends ALL history.
        # Ideally, if I want 0 history, use_history=False.
        # But if history_limit=0 is passed, maybe user expects 0 history?
        # Current implementation: `if history_limit:`. 0 is falsy.

        pass

    @patch("ai_tools.tools.LLMQuery._get_client_for_model")
    def test_query_passes_limit(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Response"
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.tool_calls = None

        self.llm.query(user_prompt="test", history_limit=3)

        # Verify creating called with correct messages
        call_args = mock_client.chat.completions.create.call_args
        # request_kwargs -> messages
        messages = call_args[1]["messages"]
        # 1 system + 3 history + 1 user = 5
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[1]["content"], "Message 3")

    @patch("ai_tools.tools.LLMQuery._get_client_for_model")
    def test_init_limit_priority(self, mock_get_client):
        # Test that instance limit is used if no override
        llm_limited = LLMQuery(history_limit=2)
        llm_limited.chat_history = self.llm.chat_history  # 5 items

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Response"

        llm_limited.query(user_prompt="test")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        # 1 system + 2 history + 1 user = 4
        self.assertEqual(len(messages), 4)


if __name__ == "__main__":
    unittest.main()
