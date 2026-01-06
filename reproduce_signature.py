import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.getcwd())
load_dotenv(override=True)

# Copied from tools.py to avoid import issues or just for clarity
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def inspect_response():
    print("Directly calling OpenAI client with Gemini URL...")

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found.")
        return

    client = OpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=GOOGLE_API_KEY,
    )

    # Try to trick it into giving a thought signature?
    # Use a reasoning model check?
    model = "gemini-2.0-flash-exp"  # Trying a newer model just in case
    # Fallback to flash-latest if not available/working

    try:
        print(f"Calling model {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Solve this riddle: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? Explain your reasoning.",
                }
            ],
        )
    except Exception as e:
        print(f"Failed with {model}, trying gemini-1.5-flash...")
        model = "gemini-1.5-flash"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Explain your reasoning. What is 2+2?"}
            ],
        )

    print("\nResponse obtained.")

    print("\n--- Full Response Model Dump ---")
    print(json.dumps(response.model_dump(), default=str, indent=2))

    message = response.choices[0].message
    print("\n--- Message Extra ---")
    if message.model_extra:
        print(f"model_extra: {message.model_extra}")
    else:
        print("No model_extra found on message.")


if __name__ == "__main__":
    inspect_response()
