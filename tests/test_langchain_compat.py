import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path so we can import ai_tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_tools.tools import LLMQuery


class TestLLMQueryLangChainCompat(unittest.TestCase):
    def setUp(self):
        # Patch the get_client method to avoid real API calls
        self.patcher = patch("ai_tools.tools.LLMQuery._get_client_for_model")
        self.mock_get_client = self.patcher.start()

        # Setup mock client and completions
        self.mock_client = MagicMock()
        self.mock_get_client.return_value = self.mock_client

        # Mock the response structure
        self.mock_response = MagicMock()
        self.mock_response.choices[0].message.content = "Test Response"
        self.mock_response.choices[0].message.tool_calls = None

        self.mock_client.chat.completions.create.return_value = self.mock_response

        self.llm = LLMQuery(model="gpt-4o-mini")

    def tearDown(self):
        self.patcher.stop()

    def test_invoke_string(self):
        response = self.llm.invoke("Hello")
        self.assertEqual(response, "Test Response")
        # Verify query was called effectively (by checking chat history or client call)
        self.mock_client.chat.completions.create.assert_called()
        call_args = self.mock_client.chat.completions.create.call_args
        # messages should contain system prompt + user "Hello"
        messages = call_args[1]["messages"]
        self.assertEqual(messages[-1]["content"], "Hello")

    def test_invoke_dict_input(self):
        response = self.llm.invoke({"input": "Hello Dict"})
        self.assertEqual(response, "Test Response")
        messages = self.mock_client.chat.completions.create.call_args[1]["messages"]
        self.assertEqual(messages[-1]["content"], "Hello Dict")

    def test_invoke_dict_query(self):
        response = self.llm.invoke({"query": "Hello Query"})
        self.assertEqual(response, "Test Response")
        messages = self.mock_client.chat.completions.create.call_args[1]["messages"]
        self.assertEqual(messages[-1]["content"], "Hello Query")

    def test_invoke_list(self):
        history = [{"role": "user", "content": "Hello List"}]
        response = self.llm.invoke(history)
        self.assertEqual(response, "Test Response")
        messages = self.mock_client.chat.completions.create.call_args[1]["messages"]
        # It should append our list to the existing history/messages
        self.assertEqual(messages[-1]["content"], "Hello List")

    def test_invoke_kwargs(self):
        # Test passing kwargs like model override
        self.llm.invoke("Hello", model="gpt-5-nano")
        call_args = self.mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "gpt-5-nano")


if __name__ == "__main__":
    unittest.main()
