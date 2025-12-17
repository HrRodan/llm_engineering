import json

with open(
    "c:/Users/Martin.Mueller/projects/llm_engineering/week2/solution/day5_solution.ipynb",
    "r",
    encoding="utf-8",
) as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def chat" in source or "LLMQuery" in source or "put_message" in source:
            print(f"--- Cell {i} ---")
            print(source)
