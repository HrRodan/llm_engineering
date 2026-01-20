import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from tqdm.notebook import tqdm
from openai import OpenAI

load_dotenv(override=True)
# groq = ""  # Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-5-nano"
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.done = False
        self.status = None
        self.error = None
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "reasoning_effort": "low",
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:
            response = groq.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        response = groq.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        response = groq.batches.retrieve(self.batch_id)
        self.status = response.status
        if self.status == "completed":
            if response.output_file_id:
                self.output_file_id = response.output_file_id
                return True
            print(f"Batch {self.batch_id} completed but no output file found.")
            return False
        elif self.status == "failed":
            self.error = response.errors
            print(f"Batch {self.batch_id} failed. Errors: {response.errors}")
            return False
        else:
            print(f"Batch {self.batch_id} status: {self.status}")
            return False

    def fetch_output(self):
        output_file = str(self.output / self.filename)
        response = groq.files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                self.items[id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items, lite):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            if (batch.output / batch.filename).exists():
                batch.done = True
                continue
            if batch.batch_id:
                try:
                    batch.is_ready()
                    if batch.status != "failed" and batch.error is None:
                        continue
                except Exception as e:
                    print(f"Error checking batch status: {e}")
                    continue
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")


from google import genai
from google.genai import types


class BatchGemini:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_name = None  # Remote file name (URI)
        self.batch_name = None  # Batch job name
        self.output_file_name = None
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def make_jsonl(self, item):
        # Gemini Batch API expects:
        # {"key": "...", "request": {"contents": [...], "generation_config": ...}}
        request = {
            "contents": [{"role": "user", "parts": [{"text": item.full}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generation_config": {
                "response_mime_type": "text/plain",
            },
        }

        line = {"key": str(item.id), "request": request}
        return json.dumps(line)

    def make_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        batch_file = self.batches / self.filename
        # Upload using Files API
        response = self.client.files.upload(
            file=str(batch_file),
            config=types.UploadFileConfig(
                display_name=self.filename, mime_type="application/json"
            ),
        )
        self.file_name = response.name
        print(f"Uploaded file: {self.file_name}")

    def submit_batch(self):
        # Create batch job
        # Use gemini-2.5-flash-lite as requested
        batch_job = self.client.batches.create(
            model="gemini-2.5-flash-lite",
            src={"file_name": self.file_name},
            config=types.CreateBatchJobConfig(display_name=f"batch_{self.filename}"),
        )
        self.batch_name = batch_job.name
        print(f"Submitted batch job: {self.batch_name}")

    def is_ready(self):
        batch_job = self.client.batches.get(name=self.batch_name)
        state = batch_job.state.name
        if state == "JOB_STATE_SUCCEEDED":
            if batch_job.dest and batch_job.dest.file_name:
                self.output_file_name = batch_job.dest.file_name
            return True
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            print(f"Job failed/cancelled/expired: {state}")
            if batch_job.error:
                print(f"Error: {batch_job.error}")
            # Identify as done but maybe failed? For now just return False implying not ready-success
            # Or raise error? keeping it simple like original
            return False
        return False

    def fetch_output(self):
        if not self.output_file_name:
            print("No output file name available.")
            return

        output_file_path = self.output / self.filename
        print(f"Downloading results from {self.output_file_name} to {output_file_path}")

        # Download the results
        content = self.client.files.download(file=self.output_file_name)

        # Write bytes to file
        with open(output_file_path, "wb") as f:
            f.write(content)

    def apply_output(self):
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                # Gemini output format:
                # {"key": "...", "response": {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}}
                # or error

                key = int(json_line["key"])

                # Check for successful response
                if "response" in json_line and "candidates" in json_line["response"]:
                    candidates = json_line["response"]["candidates"]
                    if (
                        candidates
                        and "content" in candidates[0]
                        and "parts" in candidates[0]["content"]
                    ):
                        summary = candidates[0]["content"]["parts"][0]["text"]
                        self.items[key].summary = summary
                else:
                    print(
                        f"Error or no candidate for item {key}: {json_line.get('response', json_line)}"
                    )

        self.done = True

    @classmethod
    def create(cls, items, lite):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = BatchGemini(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
            # Pickle cannot serialize client objects usually, or we shouldn't save them
            # We'll remove client before pickling and recreate on load if needed or just let it be (re-init)
            if hasattr(batch, "client"):
                del batch.client

        with state.open("wb") as f:
            pickle.dump(cls.batches, f)

        # Restore items and client
        for batch in cls.batches:
            batch.items = items
            batch.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
            # Re-initialize client since we deleted it before save
            batch.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        print(f"Loaded {len(cls.batches)} batches")
