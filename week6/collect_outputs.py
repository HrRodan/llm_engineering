import json
from pathlib import Path


def collect_lite_outputs(items):
    """
    Collects outputs from /lite/output and adds them to the 'items' collection.
    Args:
        items: List of Item objects to update.
    """
    # Manual script to collect outputs from /lite/output and add them to the "items" collection
    # This is similar to Batch.apply_output but standalone

    output_dir = Path("lite/output")
    print(f"Checking for output files in {output_dir}...")

    processed_count = 0
    error_count = 0

    if output_dir.exists():
        # Iterate over all jsonl files in the directory
        for output_file in output_dir.glob("*.jsonl"):
            # print(f"Processing {output_file.name}...")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        summary = None
                        idx = None

                        # Groq / OpenAI Batch format
                        if "custom_id" in data and "response" in data:
                            idx = int(data["custom_id"])
                            if (
                                data["response"] is not None
                                and "body" in data["response"]
                            ):
                                body = data["response"]["body"]
                                if "choices" in body and len(body["choices"]) > 0:
                                    summary = body["choices"][0]["message"]["content"]

                        # Gemini format
                        elif "key" in data:
                            idx = int(data["key"])
                            if "response" in data and "candidates" in data["response"]:
                                candidates = data["response"]["candidates"]
                                if (
                                    candidates
                                    and "content" in candidates[0]
                                    and "parts" in candidates[0]["content"]
                                ):
                                    summary = candidates[0]["content"]["parts"][0][
                                        "text"
                                    ]

                        if idx is not None and summary:
                            if 0 <= idx < len(items):
                                items[idx].summary = summary
                                processed_count += 1
                        elif idx is not None:
                            # Optional: log missing summary
                            pass

                    except Exception as e:
                        # print(f"Error processing line in {output_file.name}: {e}")
                        error_count += 1

        print(f"Processed {processed_count} items. Errors: {error_count}")
    else:
        print(f"Directory {output_dir} does not exist.")


if __name__ == "__main__":
    print(
        "This script is intended to be run inside the notebook where 'items' is defined."
    )
    print(
        "Copy the 'collect_lite_outputs' function and call it with your 'items' list."
    )
