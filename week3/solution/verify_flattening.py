import pandas as pd
import json

# Sample nested data similar to what the notebook showed (taken from cell outputs)
data = [
    {
        "person_id": "d9f8a5b2-3c1e-4f6a-9b2e-7e4f1a2c9d3b",
        "first_name": "Anna",
        "address": {"street": "Prenzlauer Allee 45", "city": "Berlin"},
    },
    {
        "person_id": "a3c1b5f7-9d2e-4b6f-a1c8-2e7f3b9d4a5f",
        "first_name": "Maximilian",
        "address": {"street": "Sonnenstraße 12", "city": "München"},
    },
]

print("--- Original Behavior (pd.DataFrame) ---")
df_original = pd.DataFrame(data)
print(df_original)
print("\nColumns:", df_original.columns.tolist())

print("\n--- New Behavior (pd.json_normalize) ---")
df_new = pd.json_normalize(data)
print(df_new)
print("\nColumns:", df_new.columns.tolist())
