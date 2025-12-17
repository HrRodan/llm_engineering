import json
import os

file_path = r"c:/Users/Martin.Mueller/projects/llm_engineering/week2/solution/day5_solution.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# New content for Cell 24 (Definitions)
new_source_24 = [
    "from functools import partial\n",
    "\n",
    "def artist(city, client=None):\n",
    "    if client:\n",
    '        user_prompt = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style"\n',
    "        image = client.generate_image(user_prompt)\n",
    "        return image\n",
    '    return "Error: Client not initialized."\n',
    "\n",
    "def get_client():\n",
    "    return LLMQuery(system_prompt=system_message)",
]

# New content for Cell 15 (Chat & UI)
new_source_15 = [
    "def chat(message, client_state):\n",
    "    client = client_state\n",
    "    messages = client.query(user_prompt = message, display_output=False)\n",
    "\n",
    "    if client.tool_calls:\n",
    "        # Create a mapping of function names to functions, binding 'client' where needed\n",
    "        functions_map = {\n",
    '            "get_ticket_price": get_ticket_price,\n',
    '            "artist": partial(artist, client=client)\n',
    "        }\n",
    "        \n",
    "        for tool_call in client.tool_calls:\n",
    "            # Pass the function map to handle_tool_call\n",
    "            result = handle_tool_call(tool_call, functions_map)\n",
    "\n",
    "    if client.response:\n",
    "        return client.get_chat_history_as_string(), None, None\n",
    "    elif client.audio:\n",
    "        return client.get_chat_history_as_string(), client.audio, None\n",
    "    elif client.image:\n",
    "        # Show the image!\n",
    "        return client.get_chat_history_as_string(), None, client.image\n",
    "        \n",
    "    return client.get_chat_history_as_string(), None, None\n",
    "\n",
    "with gr.Blocks() as ui:\n",
    '    gr.Markdown("## FlightAI Support")\n',
    "    \n",
    "    # Initialize client state for each session\n",
    "    client_state = gr.State(get_client)\n",
    "    \n",
    "    chatbot = gr.Chatbot(height=500)\n",
    "    \n",
    '    message = gr.Textbox(placeholder="Ask me about flights, or for a picture of a city..", label="Your message:")\n',
    "    \n",
    "    with gr.Row():\n",
    '        audio_output = gr.Audio(label="Audio Response", autoplay=True)\n',
    '        image_output = gr.Image(label="Generated Image")\n',
    "\n",
    "    # Hooking up events to callbacks\n",
    "    # Pass client_state as input to chat\n",
    "    message.submit(chat, inputs=[message, client_state], outputs=[chatbot, audio_output, image_output])",
]

# Find and replace
for cell in nb["cells"]:
    if cell.get("id") == "121f2022":
        print("Found Cell 24. replacing...")
        cell["source"] = new_source_24
    elif cell.get("id") == "d877c453-e7fb-482a-88aa-1a03f976b9e9":
        print("Found Cell 15. replacing...")
        cell["source"] = new_source_15

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
