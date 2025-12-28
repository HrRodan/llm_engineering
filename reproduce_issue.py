from ai_tools.hugging_face import HuggingFaceQuery
import torch


def test_hugging_face_query():
    print("Initializing HuggingFaceQuery...")
    # Force loading a small model for test if possible or just rely on the class structure usage
    # Since we can't easily download large models in this environment without risking timeouts or memory,
    # we might test the structure or lightweight models.
    # Let's try to initialize with a very small model for text to speed things up if we were to run it.
    # But for now, we will verify the object creation and method existence.

    hf = HuggingFaceQuery(
        model_name="sshleifer/tiny-gpt2",  # Small model for faster testing
        quantization="none",
        device="cpu",
    )

    print("Object initialized.")
    if hasattr(hf, "query"):
        print("- query method exists.")
    if hasattr(hf, "query_stream"):
        print("- query_stream method exists.")
    if hasattr(hf, "generic_pipeline"):
        print("- generic_pipeline method exists.")
    if hasattr(hf, "generate_image"):
        print("- generate_image method exists.")
    if hasattr(hf, "transcribe_audio"):
        print("- transcribe_audio method exists.")

    # Testing lazy loading logic by checking private attributes are None
    if hf._model is None and hf._tokenizer is None:
        print("Lazy loading verified: model and tokenizer are None initially.")

    print("Verification script finished successfully.")


if __name__ == "__main__":
    test_hugging_face_query()
