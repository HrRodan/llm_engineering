import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure we can import ai_tools by adding project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_tools.tools import LLMQuery


class TestEmbeddingStructure(unittest.TestCase):
    def test_embedding_model_attribute(self):
        """Test that LLMQuery has the embedding_model attribute with default value."""
        llm = LLMQuery()
        self.assertTrue(hasattr(llm, "embedding_model"))
        self.assertEqual(llm.embedding_model, "qwen/qwen3-embedding-8b")

    def test_generate_embedding_method_exists(self):
        """Test that generate_embedding method exists."""
        llm = LLMQuery()
        self.assertTrue(hasattr(llm, "generate_embedding"))
        self.assertTrue(callable(llm.generate_embedding))

    @patch("ai_tools.tools.LLMQuery._get_client_for_model")
    def test_generate_embedding_call(self, mock_get_client):
        """Test that generate_embedding calls the client correctly with a list of strings."""
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock response structure
        mock_response = MagicMock()
        mock_data1 = MagicMock()
        mock_data1.embedding = [0.1, 0.2, 0.3]
        mock_data2 = MagicMock()
        mock_data2.embedding = [0.4, 0.5, 0.6]

        mock_response.data = [mock_data1, mock_data2]
        mock_client.embeddings.create.return_value = mock_response

        llm = LLMQuery()
        texts = ["text1", "text2"]
        result = llm.generate_embedding(texts)

        # Verify client calls
        mock_get_client.assert_called_with("qwen/qwen3-embedding-8b")
        mock_client.embeddings.create.assert_called_with(
            model="qwen/qwen3-embedding-8b", input=texts
        )
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    @patch("ai_tools.tools.LLMQuery._get_client_for_model")
    def test_generate_embedding_custom_model(self, mock_get_client):
        """Test generating embedding with a custom model and list input."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.9, 0.8]
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response

        llm = LLMQuery()
        custom_model = "custom_model"
        texts = ["text"]
        result = llm.generate_embedding(texts, model=custom_model)

        mock_get_client.assert_called_with(custom_model)
        mock_client.embeddings.create.assert_called_with(
            model=custom_model, input=texts
        )
        self.assertEqual(result, [[0.9, 0.8]])


if __name__ == "__main__":
    unittest.main()
