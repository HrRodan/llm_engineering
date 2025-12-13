import json
import os
from typing import Dict, List, Literal, get_args, Union, Generator, Optional

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv(override=True)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

GPTModels = Literal[
    "gpt-4o-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-4.1-mini",
]

OllamaModels = Literal["llama3.2", "deepseek-r1:1.5b"]

GeminiModels = Literal[
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

OpenRouterModels = Literal[
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4",
]

ModelName = Union[GPTModels, OllamaModels, GeminiModels, OpenRouterModels]

MODEL_DICT = {
    "gpt": set(get_args(GPTModels)),
    "ollama": set(get_args(OllamaModels)),
    "gemini": set(get_args(GeminiModels)),
    "openrouter": set(get_args(OpenRouterModels)),
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
        system_prompt: str = "",
        model: ModelName = "gemini-flash-lite-latest",
        stream: bool = False,
        json_format: bool = False,
    ):
        """
        Initialize the LLMQuery instance.

        Args:
            system_prompt (str, optional): The system prompt to use. Defaults to "".
            model (ModelName, optional): The model to use. Defaults to "gemini-flash-lite-latest".
            stream (bool, optional): Whether to stream the response by default. Defaults to False.
            json_format (bool, optional): Whether to request JSON format by default. Defaults to False.
        """
        self.model = model
        self.stream = stream
        self.json_format = json_format
        self.client = self.get_client()
        self.system_prompt = system_prompt
        self.chat_history: List[Dict[str, str]] = []
        self.response = ""

    def get_client(self) -> OpenAI:
        """
        Get the OpenAI client for the configured model.

        Returns:
            OpenAI: The OpenAI client instance.

        Raises:
            ValueError: If the model is not supported.
        """
        if self.model in MODEL_DICT["gpt"]:
            client = OpenAI()
        elif self.model in MODEL_DICT["ollama"]:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        elif self.model in MODEL_DICT["gemini"]:
            client = OpenAI(
                base_url=GEMINI_BASE_URL,
                api_key=GOOGLE_API_KEY,
            )
        elif self.model in MODEL_DICT["openrouter"]:
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")
        return client

    def query(
        self,
        user_prompt: Union[str, List[Dict[str, str]]],
        use_history: bool = False,
        display_output: bool = False,
        stream: Optional[bool] = None,
        json_format: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        stream_chunks: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a query to the LLM and get the response.

        Args:
            user_prompt (Union[str, List[Dict[str, str]]]): The prompt to send to the LLM.
                Can be a string or a list of message dictionaries.
            use_history (bool, optional): Whether to include chat history in the context. Defaults to False.
            display_output (bool, optional): Whether to display the output using IPython display. Defaults to False.
            stream (bool, optional): Whether to stream the response. Overrides instance setting if provided. Models specific behavior.
            json_format (bool, optional): Whether to request JSON format response. Overrides instance setting.
            reasoning_effort (str, optional): Effort level for reasoning models (e.g. o1).
            stream_chunks (bool, optional): If True, yields chunks of the response as they arrive. Implies stream=True.

        Returns:
            Union[str, Generator[str, None, None]]: The response text, or a generator yielding response chunks if stream_chunks is True.
        """
        # --- Value Resolution ---
        if stream is None:
            stream = self.stream
        if json_format is None:
            json_format = self.json_format

        # Force stream=True if stream_chunks is requested to ensure generator is returned/used
        if stream_chunks:
            stream = True

        # --- Message Construction ---
        messages = [{"role": "system", "content": self.system_prompt}]
        if use_history:
            messages.extend(self.chat_history)

        # Handle user_prompt being a string or a list of message dicts
        # If list, it's appended directly (assumed to be correct format)
        if isinstance(user_prompt, list):
            messages.extend(user_prompt)
        else:
            messages.append({"role": "user", "content": user_prompt})

        # --- Request Preparation ---
        kwargs = {"model": self.model, "messages": messages}
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream"] = True
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        # Make the API call
        response = self.client.chat.completions.create(**kwargs)  # pyrefly: ignore

        # --- Generator Definition ---
        # Inner generator function to handle streaming and history update side-effects
        def stream_generator():
            output = ""
            display_handle = None
            if display_output:
                display_handle = display(Markdown(output), display_id=True)

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    output += content
                    if display_handle:
                        display_handle.update(Markdown(output))  # pyrefly: ignore
                    yield output

            self.response = output

            # Update chat history (only after full stream is consumed)
            if isinstance(user_prompt, list):
                self.chat_history.extend(user_prompt)
            else:
                self.chat_history.append({"role": "user", "content": user_prompt})

            self.chat_history.append({"role": "assistant", "content": self.response})

        # --- Execution Handling ---
        if stream:
            gen = stream_generator()
            if stream_chunks:
                # Return the generator directly for external consumption
                return gen
            else:
                # Consume generator fully to execute side effects (display + history)
                # This makes the method synchronous but simulated streaming display
                for _ in gen:
                    pass
                return self.response
        else:
            # --- Non-Streaming Handling ---
            self.response = response.choices[0].message.content
            if display_output:
                self.display_response()

            # Update chat history
            if isinstance(user_prompt, list):
                self.chat_history.extend(user_prompt)
            else:
                self.chat_history.append({"role": "user", "content": user_prompt})

            self.chat_history.append({"role": "assistant", "content": self.response})

            return self.response

    def display_response(self):
        """Display the response in the notebook using Markdown or JSON pretty print."""
        if self.json_format:
            pretty_print_json(self.response)
        else:
            display(Markdown(self.response))

    def get_chat_history_as_string(self) -> str:
        """
        Get the chat history as a formatted string.

        Returns:
            str: The formatted chat history.
        """
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
        """Display the chat history in the notebook."""
        display(Markdown(self.get_chat_history_as_string()))


if __name__ == "__main__":
    llm = LLMQuery(system_prompt="", model="gemini-2.5-flash")
