import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# 1. Define the Grader Function as a String
# This code runs in a sandboxed environment on OpenAI's servers.
grader_source = """
import re

def clean_price(price):
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        price = price.replace('$', '').replace(',', '')
        match = re.search(r"[-+]?\\d*\\.\\d+|\\d+", price)
        return float(match.group()) if match else 0.0
    return 0.0

def grade(sample, item):
    try:
        pred_text = sample.get('output_text', '')   
        pred_price = clean_price(pred_text)
        
        true_price = 0.0
        if 'messages' in item:
            true_price = clean_price(item['messages'][-1]['content'])
        elif 'price' in item:
            true_price = clean_price(item['price'])
            
        if true_price == 0:
            return 0.0
            
        error = abs(pred_price - true_price) / true_price
        return float(max(0.0, 1.0 - error))
    except Exception:
        return 0.0
"""


def upload_file(file_path):
    print(f"Uploading {file_path}...")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded file ID: {response.id}")
    return response.id


def create_job():
    # Paths to your local datasets
    train_path = r"c:\Users\Martin.Mueller\projects\llm_engineering\week6\jsonl\fine_tune_train.jsonl"
    valid_path = r"c:\Users\Martin.Mueller\projects\llm_engineering\week6\jsonl\fine_tune_validation.jsonl"

    print("Checking/Uploading files...")
    # Ideally, check if files exist or just upload. For simplicity, we upload fresh.
    train_file_id = upload_file(train_path)
    valid_file_id = upload_file(valid_path)

    print("Creating Reinforcement Fine-Tuning Job...")
    response = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=valid_file_id,
        model="o4-mini-2025-04-16",
        seed=42,
        suffix="pricer",
        method={
            "type": "reinforcement",
            "reinforcement": {
                "grader": {"type": "python", "source": grader_source},
                "hyperparameters": {"reasoning_effort": "medium"},
            },
        },
    )

    print(f"\nJob Created Successfully!")
    print(f"Job ID: {response.id}")
    print(f"Status: {response.status}")
    print(f"Model: {response.model}")


if __name__ == "__main__":
    create_job()
