# Multi-threaded Resumable Vector DB Population using LLMQuery
import sys
import os
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

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
COLLECTION_NAME = "products_llm"
BATCH_SIZE = 1000
MAX_WORKERS = 50

# Global dataset (loaded once)
train_dataset = []


def process_batch(batch_start_index):
    try:
        # Initialize thread-local resources
        # Each thread gets its own LLMQuery to ensure thread safety
        llm = LLMQuery()

        # We can share the client for read/write if the DB supports it,
        # but creating a client per thread is safer for some backends.
        # For ChromaDB with SQLite, concurrent writes might be locked,
        # so we rely on retry logic or just hope standard locking works.
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)

        total_items = len(train_dataset)
        end_index = min(batch_start_index + BATCH_SIZE, total_items)
        batch_ids = [f"doc_{j}" for j in range(batch_start_index, end_index)]

        # Check if this batch is already fully processed (Resumability)
        # We check existing IDs.
        existing = collection.get(ids=batch_ids, include=[])

        if len(existing["ids"]) == len(batch_ids):
            # Batch already exists
            return

        # Process the batch
        batch_items = train_dataset[batch_start_index:end_index]
        documents = [item.summary for item in batch_items]

        # Generate embeddings
        vectors = llm.generate_embedding(documents)

        # Prepare metadata
        metadatas = [
            {"category": item.category, "price": item.price} for item in batch_items
        ]

        # Add to collection (upsert to handle partials/retries safely)
        collection.upsert(
            ids=batch_ids, documents=documents, embeddings=vectors, metadatas=metadatas
        )

    except Exception as e:
        print(f"Error in batch {batch_start_index}: {e}")


def populate_db():
    global train_dataset
    print("Initializing...")

    # Load dataset
    print("Loading dataset...")
    # Using the same dataset logic as notebook
    LITE_MODE = False
    username = "ed-donner"
    dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)
    train_dataset = train
    print(f"Loaded {len(train):,} items.")

    # Initialize ChromaDB to ensure collection exists
    client = chromadb.PersistentClient(path=DB_PATH)
    client.get_or_create_collection(COLLECTION_NAME)

    print(f"Populating {COLLECTION_NAME} in {DB_PATH} with {MAX_WORKERS} workers...")

    total_items = len(train)
    # Generate start indices for batches
    batch_starts = range(0, total_items, BATCH_SIZE)

    # Use ThreadPoolExecutor for parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        # We wrap tqdm around as_completed to show progress
        futures = [executor.submit(process_batch, start) for start in batch_starts]

        for _ in tqdm(
            concurrent.futures.as_completed(futures), total=len(batch_starts)
        ):
            pass

    print("Population complete.")


if __name__ == "__main__":
    populate_db()
