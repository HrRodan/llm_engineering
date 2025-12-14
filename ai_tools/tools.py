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
    "gpt-5.2-pro",
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
    "anthropic/claude-opus-4.5",
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
        self.system_prompt = system_prompt
        self.chat_history: List[Dict[str, str]] = []
        self.response = ""

    def _get_client_for_model(self, model: str) -> OpenAI:
        """
        Get the OpenAI client for the specified model.

        Args:
            model: The model name to get the client for.

        Returns:
            OpenAI: The OpenAI client instance.

        Raises:
            ValueError: If the model is not supported.
        """
        if model in MODEL_DICT["gpt"]:
            client = OpenAI()
        elif model in MODEL_DICT["ollama"]:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        elif model in MODEL_DICT["gemini"]:
            client = OpenAI(
                base_url=GEMINI_BASE_URL,
                api_key=GOOGLE_API_KEY,
            )
        elif model in MODEL_DICT["openrouter"]:
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
        else:
            raise ValueError(f"Model {model} not supported")
        return client

    @property
    def client(self) -> OpenAI:
        """
        Get the OpenAI client for the configured model.

        Returns:
            OpenAI: The OpenAI client instance.
        """
        return self._get_client_for_model(self.model)

    def _prepare_messages(
        self, user_prompt: Union[str, List[Dict[str, str]]], use_history: bool
    ) -> List[Dict[str, str]]:
        """
        Prepare the list of messages for the API call.

        Args:
            user_prompt: The user's input.
            use_history: Whether to include chat history.

        Returns:
            List of message dictionaries.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if use_history:
            messages.extend(self.chat_history)

        if isinstance(user_prompt, list):
            messages.extend(user_prompt)
        else:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    def _prepare_request_kwargs(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        json_format: bool,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Prepare the keyword arguments for the API call.
        """
        target_model = model if model else self.model
        request_kwargs = {"model": target_model, "messages": messages}
        if json_format:
            request_kwargs["response_format"] = {"type": "json_object"}
        if stream:
            request_kwargs["stream"] = True
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort

        # Include any additional kwargs
        request_kwargs.update(kwargs)

        return request_kwargs

    def _update_history(
        self, user_prompt: Union[str, List[Dict[str, str]]], response_content: str
    ):
        """
        Update the chat history with the user prompt and assistant response.
        """
        if isinstance(user_prompt, list):
            self.chat_history.extend(user_prompt)
        else:
            self.chat_history.append({"role": "user", "content": user_prompt})

        self.chat_history.append({"role": "assistant", "content": response_content})

    def query(
        self,
        user_prompt: Union[str, List[Dict[str, str]]],
        model: Optional[ModelName] = None,
        use_history: bool = False,
        display_output: bool = False,
        json_format: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a non-streaming query to the LLM.

        Args:
            user_prompt: The prompt to send.
            model: Optional model to use, overriding the default instance model.
            use_history: Whether to include chat history.
            display_output: Whether to display the output using IPython display.
            json_format: Whether to request JSON format (overrides instance default).
            reasoning_effort: Effort level for reasoning models.
            **kwargs: Additional arguments passed to the API call.

        Returns:
            The response text.
        """
        # Resolve defaults
        json_format = json_format if json_format is not None else self.json_format
        target_model = model if model else self.model
        client = self._get_client_for_model(target_model)

        messages = self._prepare_messages(user_prompt, use_history)
        request_kwargs = self._prepare_request_kwargs(
            messages,
            stream=False,
            json_format=json_format,
            model=target_model,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        response = client.chat.completions.create(**request_kwargs)  # pyrefly: ignore
        content = response.choices[0].message.content

        # Update state
        self.response = content
        self._update_history(user_prompt, content)

        if display_output:
            self.display_response()

        return content

    def query_stream(
        self,
        user_prompt: Union[str, List[Dict[str, str]]],
        model: Optional[ModelName] = None,
        use_history: bool = False,
        display_output: bool = False,
        json_format: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        return_generator: bool = True,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a streaming query to the LLM.

        Args:
            user_prompt: The prompt to send.
            model: Optional model to use, overriding the default instance model.
            use_history: Whether to include chat history.
            display_output: Whether to display the output incrementally using IPython display.
            json_format: Whether to request JSON format (overrides instance default).
            reasoning_effort: Effort level for reasoning models.
            return_generator: If True, returns a generator yielding chunks. If False, returns the full response string.
            **kwargs: Additional arguments passed to the API call.

        Yields:
            Chunks of the response text as they arrive (if return_generator=True).
        Returns:
            The full response string (if return_generator=False).
        """
        # Resolve defaults
        json_format = json_format if json_format is not None else self.json_format
        target_model = model if model else self.model
        client = self._get_client_for_model(target_model)

        messages = self._prepare_messages(user_prompt, use_history)
        request_kwargs = self._prepare_request_kwargs(
            messages,
            stream=True,
            json_format=json_format,
            model=target_model,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        response_stream = client.chat.completions.create(
            **request_kwargs
        )  # pyrefly: ignore

        def stream_generator():
            output = ""
            display_handle = None
            if display_output:
                display_handle = display(Markdown(output), display_id=True)

            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    output += content
                    if display_handle:
                        display_handle.update(Markdown(output))  # pyrefly: ignore
                    yield output

            # Update state after stream finishes
            self.response = output
            self._update_history(user_prompt, output)

        gen = stream_generator()

        if return_generator:
            return gen
        else:
            # Consume generator to ensure side effects run
            for _ in gen:
                pass
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
    llm = LLMQuery(system_prompt="", model="gemini-flash-lite-latest")
    a = llm.query(user_prompt="Hi", display_output=True)
    print(a)
