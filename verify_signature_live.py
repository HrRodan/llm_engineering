import os
import sys
import json

# Ensure project root is in path
sys.path.append(os.getcwd())

from ai_tools.tools import LLMQuery


def verify_with_complex_prompt():
    print("Initializing LLMQuery with gemini-3-pro-preview...")
    # Using the model requested by the user
    query_engine = LLMQuery(model="gemini-3-pro-preview")

    prompt = "How many words are in your answer to this sentence? Describe how you got the value."
    print(f"Sending prompt: {prompt}")

    # We want to see the RAW response, so we'll do what LLMQuery does internally but print the object
    client = query_engine._get_client_for_model(query_engine.model)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model=query_engine.model, messages=messages
        )

        print("\n--- Full Raw Response Dump ---")
        print(json.dumps(response.model_dump(), default=str, indent=2))

        message = response.choices[0].message
        print("\n--- Message Fields ---")
        print(f"Content: {message.content}")

        # Check specific locations
        if hasattr(message, "extra_content"):
            print(f"message.extra_content: {message.extra_content}")
        if hasattr(message, "model_extra"):
            print(f"message.model_extra: {message.model_extra}")

        print("\n--- Message Attributes ---")
        print(dir(message))

        print("\n--- Raw Dict Dump (if possible) ---")
        try:
            print(message.__dict__)
        except:
            print("Could not print __dict__")

        # Check for top-level thought_signature in model_extra if it exists
        if message.model_extra and "thought_signature" in message.model_extra:
            print("FOUND: thought_signature in model_extra root!")

    except Exception as e:
        print(f"Error during query: {e}")


if __name__ == "__main__":
    # Redirect stdout to a file for reliable capture
    with open("verification_output.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        verify_with_complex_prompt()
        sys.stdout = sys.__stdout__  # Reset just in case
