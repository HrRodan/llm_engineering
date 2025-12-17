import json

nb_path = "c:/Users/Martin.Mueller/projects/llm_engineering/week2/solution/day5_solution.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define refactored code blocks
# Block 1: get_client and artist function
new_artist_block = """# Refactored client and artist function
def get_client():
    return LLMQuery(system_prompt=system_message, tools=tools, functions=[artist, get_ticket_price])

def artist(city):
    user_prompt = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style"
    # Temporary client for image generation
    temp_client = LLMQuery(system_prompt="")
    image = temp_client.generate_image(user_prompt)
    return image
"""

# Block 2: chat function, put_message_in_chatbot, and UI
new_chat_ui_block = """# Refactored Chat and UI with Client State
def chat(client, history):
    user_input = history[-1]['content']
    client.query(user_input)
    all_tool_responses = client.tool_calls
    voice = talker("Test")
    image = None
    for tr in all_tool_responses:
        if tr['name'] == 'artist':
            image = tr['output']
    return client.chat_history, voice, image

def put_message_in_chatbot(message, history, client):
    chatbot_text = history + [{"role":"user", "content":message}]
    print(chatbot_text)
    return "", chatbot_text

with gr.Blocks() as ui:
    client_state = gr.State(get_client)
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    message.submit(put_message_in_chatbot, inputs=[message, chatbot, client_state], outputs=[message, chatbot]).then(
        chat, inputs=[client_state, chatbot], outputs=[chatbot, audio_output, image_output])

ui.launch(inbrowser=True)
"""

# Apply modifications
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])

        # Replace 'a = LLMQuery...' block
        if (
            "a = LLMQuery(system_prompt = system_message)" in source
            and "def artist(city):" in source
        ):
            cell["source"] = new_artist_block.splitlines(keepends=True)
            print("Updated artist block")

        # Replace 'client = LLMQuery...' block and UI
        if (
            "client = LLMQuery(system_prompt=system_message" in source
            and "def chat(user_input):" in source
        ):
            cell["source"] = new_chat_ui_block.splitlines(keepends=True)
            print("Updated chat and UI block")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
