import torch

try:
    print("Testing torch.set_float32_matmul_precision('high')...")
    torch.set_float32_matmul_precision("high")
    print("Function called successfully.")
except Exception as e:
    print(f"Exception: {e}")
