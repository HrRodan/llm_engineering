import sys
import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


# Mock Item class
@dataclass
class MockItem:
    summary: str
    price: float


# Ensure we can import the project modules
sys.path.append(os.getcwd())
try:
    from week6.pricer.deep_neural_network import DeepNeuralNetworkRunner
except ImportError:
    # Try alternate path structure if run from root
    from pricer.deep_neural_network import DeepNeuralNetworkRunner


def main():
    print("Setting up verification...")

    # Create mock data
    train = [MockItem(summary=f"item {i}", price=10.0 + i) for i in range(100)]
    val = [MockItem(summary=f"val {i}", price=10.0 + i) for i in range(20)]

    # Instantiate runner
    print("Instantiating DeepNeuralNetworkRunner...")
    runner = DeepNeuralNetworkRunner(train, val)

    print("Running setup()...")
    runner.setup()

    # Create a test item
    test_item = MockItem(summary="test item", price=100.0)

    # Test inference in thread
    print("Testing inference in ThreadPoolExecutor...")
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            # Run a few concurrent inferences
            futures = [ex.submit(runner.inference, test_item) for _ in range(5)]
            results = [f.result() for f in futures]

        print("Inference results:", results)
        print("Verification SUCCESS: Inference ran in threads without crashing.")
    except Exception as e:
        print(f"Verification FAILED: Crashed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
