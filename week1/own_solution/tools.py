import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv(override=True)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_DICT = {
    "gpt": {"gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2"},
    "ollama": {"llama3.2", "deepseek-r1:1.5b"},
    "gemini": {
        "gemini-3-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    },
}


def pretty_print_json(data):
    """
    Prints JSON data in a readable, indented format with syntax highlighting.
    Accepts a dictionary, list, or a JSON string.
    """
    try:
        # If input is a string, try to parse it as JSON first
        if isinstance(data, str):
            data = json.loads(data)

        # Convert back to string with indentation
        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)

        # Display using Markdown for syntax highlighting in the notebook
        display(Markdown(f"```json\n{pretty_json}\n```"))

    except json.JSONDecodeError:
        print("Invalid JSON string provided.")
    except Exception as e:
        print(f"Error prettifying JSON: {e}")


class LLMQuery:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(
        self,
        system_prompt: str,
        model="gemini-flash-lite-latest",
        stream=False,
        json_format=False,
    ):
        self.model = model
        self.stream = stream
        self.json_format = json_format
        self.client = self.get_client()
        self.system_prompt = system_prompt
        self.chat_history: List[Dict[str, str]] = []
        self.response = ""

    def get_client(self):
        if self.model in MODEL_DICT["gpt"]:
            client = OpenAI()
        elif self.model in MODEL_DICT["ollama"]:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        elif self.model in MODEL_DICT["gemini"]:
            client = OpenAI(
                base_url=GEMINI_BASE_URL,
                api_key=GOOGLE_API_KEY,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")
        return client

    def query(
        self,
        user_prompt,
        use_history=False,
        display_output=False,
        stream=None,
        json_format=None,
    ):
        if stream is None:
            stream = self.stream
        if json_format is None:
            json_format = self.json_format

        messages = [{"role": "system", "content": self.system_prompt}]
        if use_history:
            messages.extend(self.chat_history)

        # Handle user_prompt being a string or a list of message dicts
        if isinstance(user_prompt, list):
            messages.extend(user_prompt)
        else:
            messages.append({"role": "user", "content": user_prompt})

        kwargs = {"model": self.model, "messages": messages}
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream"] = True
        response = self.client.chat.completions.create(**kwargs)  # pyrefly: ignore

        # stream output
        if stream:
            output = ""
            if display_output:
                display_handle = display(Markdown(output), display_id=True)
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    output += content
                    if display_output:
                        display_handle.update(Markdown(output))  # pyrefly: ignore
            self.response = output
        else:
            # without stream
            self.response = response.choices[0].message.content
            if display_output:
                self.display_response()

        # Update chat history
        if isinstance(user_prompt, list):
            self.chat_history.extend(user_prompt)
        else:
            self.chat_history.append({"role": "user", "content": user_prompt})

        self.chat_history.append({"role": "assistant", "content": self.response})

    def display_response(self):
        if self.json_format:
            pretty_print_json(self.response)
        else:
            display(Markdown(self.response))

    def get_chat_history_as_string(self):
        history: List[str] = []
        for msg in self.chat_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            if role == "User":
                history.append(f"**User**: {content}")
            elif role == "Assistant":
                history.append(f"**Assistant**: {content}")
        return "\n\n".join(history)

    def display_chat_history(self):
        display(Markdown(self.get_chat_history_as_string()))
