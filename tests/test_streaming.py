import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path so we can import ai_tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_tools.tools import LLMQuery


class MockChunk:
    def __init__(self, content):
        self.choices = [MagicMock()]
        self.choices[0].delta.content = content


class TestLLMQueryStreaming(unittest.TestCase):
    def setUp(self):
        # Patch the get_client method to avoid real API calls
        self.patcher = patch("ai_tools.tools.LLMQuery.get_client")
        self.mock_get_client = self.patcher.start()

        # Setup mock client and completions
        self.mock_client = MagicMock()
        self.mock_get_client.return_value = self.mock_client
        self.mock_completions = self.mock_client.chat.completions.create

        self.llm = LLMQuery(model="gpt-4o-mini")

    def tearDown(self):
        self.patcher.stop()

    def test_stream_chunks_true(self):
        # Setup mock response for streaming
        mock_chunks = [MockChunk("Hello"), MockChunk(" "), MockChunk("World")]
        self.mock_completions.return_value = iter(mock_chunks)

        # Call query with stream_chunks=True
        generator = self.llm.query("Test prompt", stream=True, stream_chunks=True)

        # Verify it returns a generator/iterator
        self.assertTrue(hasattr(generator, "__next__"))

        # Verify content
        result = list(generator)
        self.assertEqual(result, ["Hello", " ", "World"])

        # Verify chat history updated (after consumption)
        # Note: Depending on implementation, history might update after generator is exhausted
        self.assertEqual(len(self.llm.chat_history), 2)  # User + Assistant
        self.assertEqual(self.llm.chat_history[-1]["content"], "Hello World")

    def test_stream_chunks_false_legacy(self):
        # Setup mock response for streaming (internal consumption)
        mock_chunks = [MockChunk("Hello"), MockChunk(" "), MockChunk("World")]
        self.mock_completions.return_value = iter(mock_chunks)

        # Call query with stream_chunks=False (default behavior with stream=True)
        response = self.llm.query("Test prompt", stream=True, stream_chunks=False)

        # Verify it returns a string
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Hello World")

        # Verify chat history
        self.assertEqual(len(self.llm.chat_history), 2)
        self.assertEqual(self.llm.chat_history[-1]["content"], "Hello World")

    def test_no_stream(self):
        # Setup mock response for non-streaming
        mock_message = MagicMock()
        mock_message.choices[0].message.content = "Full Response"
        self.mock_completions.return_value = mock_message

        # Call query with stream=False
        response = self.llm.query("Test prompt", stream=False)

        self.assertEqual(response, "Full Response")
        self.assertEqual(self.llm.chat_history[-1]["content"], "Full Response")


if __name__ == "__main__":
    unittest.main()
