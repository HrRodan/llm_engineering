import json
import os
import base64
import io
from PIL import Image
from typing import (
    Dict,
    List,
    Literal,
    get_args,
    Union,
    Generator,
    Optional,
    Any,
    Callable,
)

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
    "gpt-image-1.5",
    "gpt-4o-mini-tts",
    "tts-1",
]

OllamaModels = Literal["llama3.2", "deepseek-r1:1.5b"]

GeminiModels = Literal[
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "models/imagen-4.0-generate-001",
    "gemini-2.5-pro-preview-tts",
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


def handle_tool_call(
    tool_calls: List[Dict[str, Any]], functions: List[Callable]
) -> List[Dict[str, Any]]:
    """
    Handle LLM tool calls by executing the corresponding functions.

    Iterates over a list of tool calls, looks up the function in the provided list,
    executes it with the provided arguments, and collects the results.

    Args:
        tool_calls (List[Dict]): A list of tool call dictionaries from the LLM response.
            Each dictionary should contain 'function' with 'name' and 'arguments', and an 'id'.
        functions (List[Callable]): A list of functions that can be called.

    Returns:
        List[Dict]: A list of tool response dictionaries containing 'tool_call_id', 'output', 'arguments', and 'name'.
    """
    tool_response = []
    function_map = {f.__name__: f for f in functions}

    for tool_call in tool_calls:
        # Extract the function name and parse arguments from the tool call
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        if function_name in function_map:
            function_to_call = function_map[function_name]
            result = function_to_call(**arguments)
            tool_response.append(
                {
                    "tool_call_id": tool_call["id"],
                    "output": result,
                    "arguments": arguments,
                    "name": function_name,
                }
            )
        else:
            print(f"Warning: Function {function_name} not found in provided functions")

    return tool_response


class LLMQuery:
    def __init__(
        self,
        system_prompt: str = "",
        model: ModelName = "gemini-flash-latest",
        stream: bool = False,
        json_format: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        functions: Optional[List[Callable]] = None,
        image_model: str = "models/imagen-4.0-generate-001",
        tts_model: str = "gpt-4o-mini-tts",
    ):
        """
        Initialize the LLMQuery instance.

        Args:
            system_prompt (str, optional): The system prompt to use. Defaults to "".
            model (ModelName, optional): The model to use. Defaults to "gemini-flash-lite-latest".
            stream (bool, optional): Whether to stream the response by default. Defaults to False.
            json_format (bool, optional): Whether to request JSON format by default. Defaults to False.
            tools (List[Dict], optional): List of tools to be available to the model. Defaults to None.
            tool_choice (Union[str, Dict], optional): Tool choice strategy. Defaults to None.
            functions (List[Callable], optional): List of functions to be available to the model. Defaults to None.
            image_model (str, optional): The image generation model to use. Defaults to "models/imagen-4.0-generate-001".
            tts_model (str, optional): The TTS model to use. Defaults to "tts-1".
        """
        self.model = model
        self.image_model = image_model
        self.tts_model = tts_model
        self.stream = stream
        self.json_format = json_format
        self.tools = tools
        if functions is None:
            self.functions = []
        else:
            self.functions = functions
        self.tool_choice = tool_choice
        self.system_prompt = system_prompt
        self.chat_history: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict] = []
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
        self,
        user_prompt: Union[str, List[Dict[str, str]], None],
        use_history: bool,
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

        if user_prompt is not None:
            if isinstance(user_prompt, list):
                messages.extend(user_prompt)
            else:
                messages.append({"role": "user", "content": user_prompt})

        # Ensure at least one message exists besides system prompt to satisfy APIs like Gemini
        if len(messages) == 1:
            messages.append({"role": "user", "content": ""})

        return messages

    def _prepare_request_kwargs(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        json_format: bool,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> Dict:
        """
        Prepare the keyword arguments for the API call.
        """
        target_model = model if model else self.model
        request_kwargs: Dict[str, Any] = {"model": target_model, "messages": messages}

        # Tools handling
        target_tools = tools if tools is not None else self.tools
        target_tool_choice = (
            tool_choice if tool_choice is not None else self.tool_choice
        )

        if target_tools:
            request_kwargs["tools"] = target_tools
        if target_tool_choice:
            request_kwargs["tool_choice"] = target_tool_choice

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
        self,
        user_prompt: Union[str, List[Dict[str, str]], None],
        response_content: Optional[str],
        tool_calls: Optional[List[Dict]] = None,
    ):
        """
        Update the chat history with the user prompt, assistant response and results from tool calls.
        """
        if user_prompt is not None:
            if isinstance(user_prompt, list):
                self.chat_history.extend(user_prompt)
            else:
                self.chat_history.append({"role": "user", "content": user_prompt})

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response_content,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        self.chat_history.append(assistant_msg)

    def query(
        self,
        user_prompt: Union[str, List[Dict[str, str]], None] = None,
        model: Optional[ModelName] = None,
        use_history: bool = True,
        display_output: bool = False,
        json_format: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
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
            tools: Optional list of tools to use.
            tool_choice: Optional tool choice strategy.
            **kwargs: Additional arguments passed to the API call.

        Returns:
            The response text.
        """
        # Resolve defaults
        json_format = json_format if json_format is not None else self.json_format
        target_model = model if model else self.model
        client = self._get_client_for_model(target_model)

        # Reset tool calls
        self.tool_calls = []

        messages = self._prepare_messages(user_prompt, use_history)
        request_kwargs = self._prepare_request_kwargs(
            messages,
            stream=False,
            json_format=json_format,
            model=target_model,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = message.content

        # Handle tool calls
        if message.tool_calls:
            self.tool_calls = [tc.model_dump() for tc in message.tool_calls]

        # Update state
        self.response = content if content is not None else ""
        self._update_history(
            user_prompt, content, self.tool_calls if self.tool_calls else None
        )

        if display_output:
            self.display_response()

        return self.response

    def query_stream(
        self,
        user_prompt: Union[str, List[Dict[str, str]], None] = None,
        model: Optional[ModelName] = None,
        use_history: bool = True,
        display_output: bool = False,
        json_format: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        return_generator: bool = True,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
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
            tools: Optional list of tools to use.
            tool_choice: Optional tool choice strategy.
            **kwargs: Additional arguments passed to the API call.

        Yields:
            Accumulated response text as it arrives (if return_generator=True).
        Returns:
            The full response string (if return_generator=False).
        """
        # Resolve defaults
        json_format = json_format if json_format is not None else self.json_format
        target_model = model if model else self.model
        client = self._get_client_for_model(target_model)

        # Reset tool calls
        self.tool_calls = []

        messages = self._prepare_messages(user_prompt, use_history)
        request_kwargs = self._prepare_request_kwargs(
            messages,
            stream=True,
            json_format=json_format,
            model=target_model,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        response_stream = client.chat.completions.create(**request_kwargs)

        def stream_generator():
            output = ""
            display_handle = None
            collected_tool_calls = {}

            if display_output:
                display_handle = display(Markdown(output), display_id=True)

            for chunk in response_stream:
                delta = chunk.choices[0].delta
                content = delta.content

                # Handle content
                if content:
                    output += content
                    if display_handle:
                        display_handle.update(Markdown(output))
                    yield output

                # Handle tool calls
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }

                        if tc_chunk.id:
                            collected_tool_calls[idx]["id"] += tc_chunk.id

                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                collected_tool_calls[idx]["function"]["name"] += (
                                    tc_chunk.function.name
                                )
                            if tc_chunk.function.arguments:
                                collected_tool_calls[idx]["function"]["arguments"] += (
                                    tc_chunk.function.arguments
                                )

            # Update state after stream finishes
            self.response = output
            if collected_tool_calls:
                self.tool_calls = list(collected_tool_calls.values())

            self._update_history(
                user_prompt,
                output if output else None,
                self.tool_calls if self.tool_calls else None,
            )

        gen = stream_generator()

        if return_generator:
            return gen
        else:
            # Consume generator to ensure side effects run
            for _ in gen:
                pass
            return self.response

    def append_tool_result(self, tool_outputs: List[Dict[str, Any]]):
        """
        Append the results of tool executions to the chat history.

        Args:
            tool_outputs: A list of dictionaries, where each dictionary contains:
                - tool_call_id: The ID of the tool call.
                - output: The output of the tool execution.
        """
        for tool_output in tool_outputs:
            output_content = tool_output["output"]
            if isinstance(output_content, Image.Image):
                output_content = "[Image created]"
            elif isinstance(output_content, bytes):
                output_content = "[Audio created]"
            elif not isinstance(output_content, str):
                try:
                    output_content = json.dumps(output_content)
                except (TypeError, ValueError):
                    output_content = f"[{type(output_content).__name__} object created]"

            self.chat_history.append(
                {
                    "role": "tool",
                    "content": output_content,
                    "tool_call_id": tool_output["tool_call_id"],
                }
            )

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
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        func_name = tool_call["function"]["name"]
                        args = tool_call["function"]["arguments"]
                        history.append(f"**Assistant Tool Call**: {func_name}({args})")
            elif role == "Tool":
                history.append(f"**Tool Output**: {content}")

        return "\n\n".join(history)

    @property
    def clean_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history as a list of dictionaries containing only role and content.

        Only includes messages from 'assistant' or 'user' roles that have non-empty content.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with 'role' and 'content' keys.
        """
        return [
            {"role": h["role"], "content": h["content"]}
            for h in self.chat_history
            if h["role"] in ("assistant", "user") and h["content"]
        ]

    def display_chat_history(self):
        """Display the chat history in the notebook."""
        display(Markdown(self.get_chat_history_as_string()))

    def get_tool_responses(
        self,
        max_iterations: int = 10,
    ) -> str:
        """
        Execute pending tool calls and continue the conversation until no more tool calls are made.

        Args:
            max_iterations: Maximum number of request-response cycles to prevent infinite loops.

        Returns:
            str: The final response from the assistant after all tool executions.
        """
        response = self.response
        iterations = 0

        while self.tool_calls and iterations < max_iterations:
            tool_response = handle_tool_call(self.tool_calls, functions=self.functions)
            self.append_tool_result(tool_response)
            query_response = self.query(tools=self.tools)

            if query_response:
                if response:
                    response += "\n\n" + query_response
                else:
                    response = query_response

            iterations += 1

        return response

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> Image.Image:
        """
        Generate an image using the specified model.

        Args:
            prompt: The prompt to generate the image for.
            model: Optional model to use, overriding the default instance image_model.
            size: The size of the image to generate. Defaults to "1024x1024".
            quality: The quality of the image to generate. Defaults to "standard".

        Returns:
            Image.Image: The generated image as a PIL Image object.
        """
        target_model = model if model else self.image_model
        client = self._get_client_for_model(target_model)
        response = client.images.generate(  # pyrefly: ignore
            model=target_model,
            prompt=prompt,
            size=size,
            quality=quality,
            response_format="b64_json",
        )

        if not response.data or not response.data[0].b64_json:
            raise ValueError("No image data returned from API")

        image_data = base64.b64decode(response.data[0].b64_json)
        return Image.open(io.BytesIO(image_data))

    def generate_tts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "onyx",
        speed: float = 1.0,
    ) -> bytes:
        """
        Generate speech from text using the specified model.

        Args:
            text: The text to generate speech for.
            model: Optional model to use, overriding the default instance tts_model.
            voice: The voice to use for generation. Defaults to "alloy".
            speed: The speed of the speech generation. Defaults to 1.0.

        Returns:
            bytes: The generated audio content.
        """
        target_model = model if model else self.tts_model
        client = self._get_client_for_model(target_model)
        response = client.audio.speech.create(
            model=target_model,
            input=text,
            voice=voice,
            speed=speed,
        )
        return response.content


if __name__ == "__main__":
    llm = LLMQuery(system_prompt="", model="gemini-flash-lite-latest")
    a = llm.query(user_prompt="Hi", display_output=True)
    print(a)
