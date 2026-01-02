import os
import sys
import subprocess
import time
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports
try:
    from ai_tools.tools import LLMQuery
    from system_info import retrieve_system_info
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    # Fallback if running relative to script location
    sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "..")))
    from ai_tools.tools import LLMQuery
    from system_info import retrieve_system_info
    from dotenv import load_dotenv

    load_dotenv(override=True)


# System Info
system_info = retrieve_system_info()

# Output Directory
output_dir = os.path.join(current_dir, "cpp_benchmark")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Prompts
system_prompt = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""


def user_prompt_for(python_code):
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called main.cpp and then compiled and executed; the compilation command is:
compile_command = ["g++", "main.cpp", "-o", "main.exe", "-O3", "-march=native", "-DNDEBUG"]
Respond only with C++ code.
Python code to port:

```python
{python_code}
```
"""


# Pi Code
pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""


# Helper Functions
def port(client, python_code):
    try:
        reply = client.query(
            user_prompt=user_prompt_for(python_code),
            display_output=False,
            use_history=False,
        )
        # Clean up response
        reply_clean = reply.replace("```cpp", "").replace("```", "")

        if reply_clean.strip() == "":
            raise Exception("Empty response")

        # Create unique filename based on model
        escaped_model = str(client.model).replace("/", "_").replace("-", "_")
        filename = os.path.join(output_dir, f"main_{escaped_model}.cpp")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(reply_clean)
        return filename
    except Exception as e:
        print(f"Error porting with {client.model}: {e}")
        return None


def run_benchmark_iterations(exe_name, runs=10):
    times = []
    results = []
    run_cmd = [exe_name]

    for i in range(runs):
        try:
            res = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
            output = res.stdout

            # Extract time using regex
            # Looking for "Execution Time: X.XXXXXX seconds"
            match_time = re.search(r"Execution Time:\s*([0-9.]+)\s*seconds", output)
            if match_time:
                times.append(float(match_time.group(1)))
            else:
                # Try fallback format if it's just the number
                try:
                    lines = output.strip().split("\n")
                    for line in lines:
                        if "seconds" in line:
                            times.append(float(line.split()[0]))
                            break
                except:
                    pass

            # Extract result using regex
            # Looking for "Result: X.XXXXXXXXXXXX"
            match_result = re.search(r"Result:\s*([0-9.]+)", output)
            if match_result:
                results.append(float(match_result.group(1)))

        except subprocess.CalledProcessError as e:
            print(f"Run {i + 1} failed for {exe_name}: {e}")
        except Exception as e:
            print(f"Error executing {exe_name}: {e}")

    if not times:
        return 0.0, 0.0, 0.0

    avg_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
    avg_result = statistics.mean(results) if results else 0.0

    return avg_time, stdev_time, avg_result


def process_client(client_data):
    name, client = client_data
    print(f"[{name}] Processing model: {client.model}")

    escaped_model = str(client.model).replace("/", "_").replace("-", "_")
    filename = os.path.join(output_dir, f"main_{escaped_model}.cpp")

    query_time = None

    # 1. Generate/Check File
    if os.path.exists(filename):
        print(
            f"[{name}] File {os.path.basename(filename)} already exists. Skipping generation."
        )
        query_time = 0.0  # Or None to indicate cached
    else:
        print(f"[{name}] Querying LLM to generate code...")
        start_q = time.time()
        filename = port(client, pi)
        end_q = time.time()
        query_time = end_q - start_q

        if not filename:
            return {"name": name, "model": client.model, "error": "Generation failed"}

    # 2. Compile
    exe_name = filename.replace(".cpp", ".exe")
    compile_cmd = ["g++", filename, "-o", exe_name, "-O3", "-march=native", "-DNDEBUG"]

    try:
        subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return {
            "name": name,
            "model": client.model,
            "error": f"Compilation failed: {e.stderr}",
        }
    except Exception as e:
        return {"name": name, "model": client.model, "error": f"Compilation error: {e}"}

    # 3. Benchmark
    print(f"[{name}] Running benchmark for 10 iterations...")
    avg_time, uncertainty, avg_result = run_benchmark_iterations(exe_name, runs=10)

    return {
        "name": name,
        "model": client.model,
        "filename": filename,
        "avg_time": avg_time,
        "uncertainty": uncertainty,
        "avg_result": avg_result,
        "query_time": query_time,
    }


# Main Execution
if __name__ == "__main__":
    # Define Clients
    clients = [
        (
            "OpenAI",
            LLMQuery(
                model="gpt-5.2", reasoning_effort="high", system_prompt=system_prompt
            ),
        ),
        (
            "Claude",
            LLMQuery(
                model="anthropic/claude-opus-4.5",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
        (
            "Grok",
            LLMQuery(
                model="x-ai/grok-4",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
        (
            "Gemini",
            LLMQuery(
                model="gemini-3-pro-preview",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
        (
            "Moonshot",
            LLMQuery(
                model="moonshotai/kimi-k2-thinking",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
        (
            "Z-AI",
            LLMQuery(
                model="z-ai/glm-4.7",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
        (
            "DeepSeek",
            LLMQuery(
                model="deepseek/deepseek-v3.2",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
        (
            "GPT-OSS",
            LLMQuery(
                model="openai/gpt-oss-120b",
                reasoning_effort="high",
                system_prompt=system_prompt,
            ),
        ),
    ]

    print(f"Starting benchmark for {len(clients)} models...")
    start_total = time.time()

    results = []

    # Run in parallel
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [executor.submit(process_client, c) for c in clients]
        for future in futures:
            results.append(future.result())

    end_total = time.time()

    print("\n" + "=" * 100)
    print(
        f"{'CLIENT':<15} | {'MODEL':<30} | {'AVG TIME (s)':<15} | {'UNCERTAINTY':<15} | {'QUERY TIME (s)':<15} | {'OUTPUT':<15}"
    )
    print("=" * 118)

    # Sort by time for better readability
    def sort_key(r):
        if "error" in r:
            return float("inf")
        return r["avg_time"]

    results.sort(key=sort_key)

    for res in results:
        name = res["name"]
        model = res["model"]
        if "error" in res:
            print(
                f"{name:<15} | {model:<30} | {'ERROR':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {res['error']}"
            )
        else:
            avg = f"{res['avg_time']:.6f}"
            unc = f"±{res['uncertainty']:.6f}"
            out = f"{res['avg_result']:.6f}"

            if res["query_time"] is None:
                qt = "N/A"
            elif res["query_time"] == 0.0:
                qt = "Cached"
            else:
                qt = f"{res['query_time']:.2f}"

            print(
                f"{name:<15} | {model:<30} | {avg:<15} | {unc:<15} | {qt:<15} | {out:<15}"
            )

    print("\n" + "=" * 100)
    print(f"Total Benchmark Duration: {end_total - start_total:.2f} seconds")
