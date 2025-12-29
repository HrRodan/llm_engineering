import os

file_path = "c:/Users/Martin.Mueller/projects/llm_engineering/week3/solution/data_generator_frontier.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = "df = pd.DataFrame(data)"
replacement = "df = pd.json_normalize(data)"

if target in content:
    new_content = content.replace(target, replacement)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Successfully patched {file_path}")
else:
    print(f"Target string '{target}' not found in {file_path}")
    # Check if it was already replaced
    if replacement in content:
        print("Replacement already present.")
    else:
        print("Could not find target or replacement. Please check the file content.")
