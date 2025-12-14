from unittest.mock import MagicMock
from ai_tools.tools import LLMQuery


def test_query_stream_generator():
    print("Testing query_stream (generator)...")
    llm = LLMQuery()
    llm.client = MagicMock()

    # Mock streaming response
    mock_chunk1 = MagicMock()
    mock_chunk1.choices[0].delta.content = "Chunk1"
    mock_chunk2 = MagicMock()
    mock_chunk2.choices[0].delta.content = "Chunk2"

    llm.client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])

    # Test query_stream with return_generator=True (default)
    gen = llm.query_stream("Hi")
    chunks = list(gen)
    full_response = "".join(chunks)

    assert full_response == "Chunk1Chunk2"
    assert llm.chat_history[-1]["content"] == "Chunk1Chunk2"
    print("Query stream (generator) test passed.")


def test_query_stream_string():
    print("Testing query_stream (string)...")
    llm = LLMQuery()
    llm.client = MagicMock()

    # Mock streaming response
    mock_chunk1 = MagicMock()
    mock_chunk1.choices[0].delta.content = "ChunkA"
    mock_chunk2 = MagicMock()
    mock_chunk2.choices[0].delta.content = "ChunkB"

    llm.client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])

    # Test query_stream with return_generator=False
    response = llm.query_stream("Hi", return_generator=False)

    assert isinstance(response, str)
    assert response == "ChunkAChunkB"
    assert llm.chat_history[-1]["content"] == "ChunkAChunkB"
    print("Query stream (string) test passed.")


if __name__ == "__main__":
    test_query_stream_generator()
    test_query_stream_string()
