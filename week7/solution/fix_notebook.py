import json
import os

input_path = "Week_7_Day_3_RL_TRAINING.ipynb"
output_path = "Week_7_Day_3_RL_TRAINING_FIXED.ipynb"

# Absolute paths
base_dir = r"c:\Users\Martin.Mueller\projects\llm_engineering\week7\solution"
input_full_path = os.path.join(base_dir, input_path)
output_full_path = os.path.join(base_dir, output_path)

print(f"Reading from {input_full_path}")
with open(input_full_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell to replace (ID: MDyR63OTNUJ6)
target_cell = None
found = False
for cell in nb["cells"]:
    if cell.get("metadata", {}).get("id") == "MDyR63OTNUJ6":
        target_cell = cell
        found = True
        break

if found:
    print("Found target cell. applying fix...")
    new_source = [
        "# Aggressively uninstall potentially conflicting libraries first\n",
        "!pip uninstall -y numpy torch torchvision torchaudio trl transformers peft accelerate bitsandbytes\n",
        "\n",
        "# Install numpy < 2.0 explicitly\n",
        '!pip install "numpy<2.0"\n',
        "\n",
        "# Install compatible versions of torch ecosystem and other libs for trl 0.9.6\n",
        "# We pin torch and transformers to versions compatible with numpy < 2.0 and trl 0.9.6\n",
        '!pip install "torch<2.5.0" "transformers<4.45.0" datasets peft "trl==0.9.6" bitsandbytes accelerate wandb\n',
        "\n",
        "import os\n",
        "import time\n",
        "\n",
        "# Automatically restart kernel to reload dependencies (Fixes ValueError: numpy.dtype size changed)\n",
        "try:\n",
        "    import numpy\n",
        '    print("Restarting runtime to apply package changes...")\n',
        "    time.sleep(1)\n",
        "    os.kill(os.getpid(), 9)\n",
        "except Exception as e:\n",
        '    print(f"Could not restart runtime automatically: {e}")',
    ]
    target_cell["source"] = new_source
    target_cell["outputs"] = []  # Clear outputs to be clean
    target_cell["execution_count"] = None

    with open(output_full_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)

    print(f"Successfully created {output_full_path}")
else:
    print("Could not find the target cell with id MDyR63OTNUJ6")
