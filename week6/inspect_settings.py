import torch

print(f"PyTorch Version: {torch.__version__}")

try:
    print(
        f"torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}"
    )
except Exception as e:
    print(f"torch.backends.cuda.matmul.allow_tf32 Error: {e}")

try:
    print(f"torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
except Exception as e:
    print(f"torch.backends.cudnn.allow_tf32 Error: {e}")

try:
    print(
        f"torch.backends.cuda.matmul.fp32_precision: {torch.backends.cuda.matmul.fp32_precision}"
    )
except Exception as e:
    print(
        f"torch.backends.cuda.matmul.fp32_precision Error: {e}"
    )  # Expecting this to exist based on warning

try:
    print(
        f"torch.backends.cudnn.conv.fp32_precision: {torch.backends.cudnn.conv.fp32_precision}"
    )
except Exception as e:
    print(f"torch.backends.cudnn.conv.fp32_precision Error: {e}")
