import os
import sys
import tempfile
import gradio as gr
from dotenv import load_dotenv

# Ensure we can import from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# Imports from project
from ai_tools import LLMQuery

# Ensure we can import locally
if current_dir not in sys.path:
    sys.path.append(current_dir)

from pokemon import PokemonAPIClient, TOOLS

load_dotenv()

# Initialize Client
pokemon_client = PokemonAPIClient()
functions = [getattr(pokemon_client, tool["function"]["name"]) for tool in TOOLS]

# Initialize LLMQuery with a default model
client = LLMQuery(
    system_prompt=pokemon_client.get_system_prompt(),
    functions=functions,
    tools=TOOLS,
    model="gpt-5-nano",
)

# Initialize TTS Client
tts_client = LLMQuery(model="gpt-4o-mini-tts")

# Available models
model_names = [
    "gpt-4o-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-4.1-mini",
    "gpt-5.2-pro",
    "llama3.2",
    "deepseek-r1:1.5b",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4",
]


def chat(message, history):
    # Query the LLM
    response = client.query(message)
    # Handle any tool calls that occurred
    response = client.get_tool_responses()

    # Generate TTS
    try:
        audio_bytes = tts_client.generate_tts(response, voice="onyx")
        # Save to a temporary file for Gradio to play
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            fp.write(audio_bytes)
            return response, fp.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return response, None


def set_model(model_name):
    # Update the model in the LLMQuery client
    client.model = model_name


# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# Pokémon Chatbot")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=model_names,
            value=client.model,
            label="Select LLM Model",
            interactive=True,
        )
        audio_output = gr.Audio(label="Professor Eich's Voice", autoplay=True)

    # Using ChatInterface for the chat UI
    chat_interface = gr.ChatInterface(
        fn=chat, type="messages", additional_outputs=[audio_output]
    )

    model_dropdown.change(fn=set_model, inputs=model_dropdown, outputs=None)

if __name__ == "__main__":
    app.launch()
