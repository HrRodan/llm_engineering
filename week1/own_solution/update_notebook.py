import json
import os

nb_path = r"c:\Users\Martin.Mueller\projects\llm_engineering\week1\own_solution\week1 EXERCISE.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell
target_cell = None
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class LLMQuery" in source:
            target_cell = cell
            found = True
            break

if not found:
    print("Cell with LLMQuery class not found!")
    exit(1)

# New source code
new_code = r"""from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_DICT = {
    'gpt': {'gpt-4o-mini', 'gpt-5-nano', 'gpt-5-mini', 'gpt-5.1'},
    'ollama': {'llama3.2', 'deepseek-r1:1.5b'},
    'gemini': {'gemini-3-pro-preview', 'gemini-flash-latest'}
}

def pretty_print_json(data):
    \"\"\"
    Prints JSON data in a readable, indented format with syntax highlighting.
    Accepts a dictionary, list, or a JSON string.
    \"\"\"
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
    def __init__(self, system_prompt, model='gemini-flash-latest', stream=False, json_format=False):
        self.model = model
        self.stream = stream
        self.json_format = json_format
        self.client = self.get_client()
        self.system_prompt = self.initiate_system_prompt(system_prompt)
        self.chat_history = []
        self.response = ''

    def get_client(self):
        if self.model in MODEL_DICT['gpt']:
            client = OpenAI()
        elif self.model in MODEL_DICT['ollama']:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
        elif self.model in MODEL_DICT['gemini']:
            client = OpenAI(
            base_url=GEMINI_BASE_URL,   
            api_key=GOOGLE_API_KEY,
        )
        else:
            raise ValueError(f"Model {self.model} not supported")
        return client
    
    def initiate_system_prompt(self, system_prompt):
        # Refactored to not rely on self.system_prompt before it is set
        if self.json_format and 'json' not in system_prompt.lower():
            return system_prompt + "\n\n" + "Return the result as strict JSON."
        else:
            return system_prompt
    
    def query(self, user_prompt, use_history=False, display_output=False, stream=None, json_format=None):
        if stream is None:
            stream = self.stream
        if json_format is None:
            json_format = self.json_format
            
        messages = [{"role": "system", "content": self.system_prompt}]
        if use_history:
            messages.extend(self.chat_history)
        messages.append({"role": "user", "content": user_prompt})

        kwargs = {
            "model": self.model,
            "messages": messages
        }
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream"] = True
            
        response = self.client.chat.completions.create(**kwargs)

        if stream:
            output = ""
            if display_output:
                display_handle = display(Markdown(output), display_id=True)
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    output += content
                    if display_output:
                        display_handle.update(Markdown(output))
            self.response = output
        else:
            self.response = response.choices[0].message.content
            if display_output:
                self.display_response()
        
        self.chat_history.append({"role": "user", "content": user_prompt})
        self.chat_history.append({"role": "assistant", "content": self.response})
    
    def display_response(self):
        if self.json_format:
            pretty_print_json(self.response)
        else:
            display(Markdown(self.response))
    
    def get_chat_history_as_string(self):
        history = []
        for msg in self.chat_history:
            role = msg['role'].capitalize()
            content = msg['content']
            if role == "User":
                history.append(f"**User**: {content}")
            elif role == "Assistant":
                history.append(f"**Assistant**: {content}")
        return "\n\n".join(history)
    
    def display_chat_history(self):
        display(Markdown(self.get_chat_history_as_string()))
"""

# Split into lines and ensure each ends with \n
source_lines = [line + '\n' for line in new_code.split('\n')]
# The split() might produce a trailing empty string if new_code ends with \n, let's fix that
if source_lines[-1] == '\n':
    source_lines.pop() 
# But wait, split('\n') on "a\n" gives ["a", ""]. "a\n" + "\n" -> "a\n" "\n".
# We want ["a\n"].
# Correct approach:
source_lines = []
lines = new_code.split('\n')
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        source_lines.append(line + '\n')
    else:
        # Last line. If original code ended with newline, we should too.
        # Usually notebook cells don't enforce a trailing newline in the last item, but having one is fine.
        if line:
            source_lines.append(line)

target_cell['source'] = source_lines

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
