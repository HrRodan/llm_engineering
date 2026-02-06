# Resumable Vector DB Population using LLMQuery
import sys
import os
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path for imports to work if run directly
# Assuming script is in week8/solution/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from ai_tools.tools import LLMQuery
from agents.items import Item

# Load environment
load_dotenv(override=True)

# Configuration
DB_PATH = "products_vectorstore"


def populate_db():
    print("Initializing...")

    # Load dataset
    print("Loading dataset...")
    # Using the same dataset logic as notebook
    LITE_MODE = False
    username = "ed-donner"
    dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)
    print(f"Loaded {len(train):,} items.")

    # Initialize ChromaDB
    # We use relative path to match where the notebook runs (assuming script receives same CWD)
    # Or strict path:
    client = chromadb.PersistentClient(path=DB_PATH)

    # Initialize LLMQuery
    llm = LLMQuery()

    # Use a distinct collection name
    collection_name_llm = "products_llm"

    # Get or create collection
    collection_llm = client.get_or_create_collection(collection_name_llm)

    print(f"Populating {collection_name_llm} in {DB_PATH}...")

    batch_size = 1000
    total_items = len(train)

    for i in tqdm(range(0, total_items, batch_size)):
        # Define batch range
        end_index = min(i + batch_size, total_items)
        batch_ids = [f"doc_{j}" for j in range(i, end_index)]

        # Check if this batch is already fully processed
        existing = collection_llm.get(ids=batch_ids, include=[])

        if len(existing["ids"]) == len(batch_ids):
            # All items in batch exist
            # print(f"Skipping existing batch {i}") # Optional: comment out to reduce noise
            continue

        print(f"Processing batch {i} to {end_index}...")

        # Else, process the batch
        batch_items = train[i:end_index]
        documents = [item.summary for item in batch_items]

        try:
            # Generate embeddings
            vectors = llm.generate_embedding(documents)

            # Prepare metadata
            metadatas = [
                {"category": item.category, "price": item.price} for item in batch_items
            ]

            # Add to collection
            collection_llm.upsert(
                ids=batch_ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"Error processing batch {i} to {end_index}: {e}")
            break

    print("Population complete.")


if __name__ == "__main__":
    populate_db()
