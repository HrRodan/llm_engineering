import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths
BASE_DIR = Path(__file__).parent
IDS_FILE = BASE_DIR / "lite" / "ids.txt"
OUTPUT_DIR = BASE_DIR / "lite" / "output"


def fetch_results():
    if not IDS_FILE.exists():
        print(f"Error: IDs file not found at {IDS_FILE}")
        return

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading IDs from {IDS_FILE}...")

    with open(IDS_FILE, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(", ")
        if len(parts) < 3:
            print(f"Skipping invalid line: {line.strip()}")
            continue

        # Format in ids.txt: file_id, None, batch_id
        # Example: file-1Ujmm9kE2cT7D277JAkbXF, None, batch_696e9f9b23888190a5a6616e49ffd6d9
        batch_id = parts[2]

        try:
            print(f"Checking status for batch: {batch_id}")
            batch = client.batches.retrieve(batch_id)

            if batch.status == "completed":
                if batch.output_file_id:
                    print(
                        f"  - Completed! Downloading output file: {batch.output_file_id}"
                    )
                    content = client.files.content(batch.output_file_id)

                    # Determine output filename (using batch_id for uniqueness)
                    output_file_path = OUTPUT_DIR / f"{batch_id}.jsonl"

                    with open(output_file_path, "wb") as out_f:
                        out_f.write(content.read())

                    print(f"  - Saved to {output_file_path}")
                else:
                    print(f"  - Completed but no output file_id found.")
            elif batch.status == "failed":
                print(f"  - Failed. Errors: {batch.errors}")
            else:
                print(f"  - Status: {batch.status}")

        except Exception as e:
            print(f"  - Error retrieving batch {batch_id}: {e}")


if __name__ == "__main__":
    fetch_results()
