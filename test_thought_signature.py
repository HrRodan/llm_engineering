import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from ai_tools.tools import LLMQuery


class TestThoughtSignature(unittest.TestCase):
    @patch("ai_tools.tools.OpenAI")
    def test_thought_signature_extraction(self, mock_openai):
        # Setup mock client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create a mock message with extra_content containing thought_signature
        mock_message = MagicMock()
        mock_message.content = "Response content"
        mock_message.tool_calls = None

        # Mocking model_extra or extra_content (since we check both)
        # We will set 'model_extra' because OpenAI python lib usually puts extra fields there
        mock_message.model_extra = {
            "google": {"thought_signature": "test_signature_123"}
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        # Initialize LLMQuery
        query_engine = LLMQuery(model="gemini-flash-latest")

        # Run query
        response = query_engine.query("Test prompt")

        # Verify response content
        self.assertEqual(response, "Response content")

        # Verify chat history contains thought_signature
        last_entry = query_engine.chat_history[-1]
        self.assertEqual(last_entry["role"], "assistant")
        self.assertEqual(last_entry.get("thought_signature"), "test_signature_123")

        print("Test passed: thought_signature was extracted and stored correctly.")


if __name__ == "__main__":
    unittest.main()
