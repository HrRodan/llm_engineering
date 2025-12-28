import warnings
import torch
from ai_tools.hugging_face import HuggingFaceQuery

# Filter out some warnings to see our output clearly
warnings.filterwarnings("ignore")


def test_generation():
    print("Initializing HuggingFaceQuery with gpt2 for speed...")
    # Using gpt2 as it is small.
    # Note: gpt2 does not support system prompts well with apply_chat_template usually,
    # but we just want to check if generate calls work with attention_mask.
    try:
        hf = HuggingFaceQuery(model_name="gpt2", device="cpu", quantization="none")

        # Manually set pad_token if needed (gpt2 usually doesn't have one by default)
        if hf.tokenizer.pad_token is None:
            hf.tokenizer.pad_token = hf.tokenizer.eos_token

        print("\nTesting query()...")
        response = hf.query("Hello, how are you?", max_new_tokens=20)
        print(f"Response: {response}")

        print("\nTesting query_stream()...")
        hf.query_stream("Tell me a short joke.", max_new_tokens=20)
        print("\nStream test finished.")

    except Exception as e:
        print(f"\nCaught exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_generation()
