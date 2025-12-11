from dotenv import load_dotenv
from openai import OpenAI
import os

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")

def get_gemini_client():
    client = OpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=google_api_key,
    )
    return client