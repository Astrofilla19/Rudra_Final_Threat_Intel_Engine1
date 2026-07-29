# find_eval_pipeline.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME


def get_qdrant_client() -> QdrantClient:
    """Forces local disk storage since we are not running a Qdrant server."""
    return QdrantClient(path=str(DB_DIR))


def search_for_ids(search_phrase: str, top_k: int = 3):
    print(f"\n🔍 Searching Qdrant database for: '{search_phrase}'")
    print("-" * 60)

    client = get_qdrant_client()

    # Ensure collection exists before querying
    if not client.collection_exists(COLLECTION_NAME):
        print("❌ Collection not found. Have you ingested videos yet?")
        return

    # Compute vector embedding for search query
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = model.encode(search_phrase, show_progress_bar=False).tolist()

    # Perform similarity search in Qdrant using the updated API
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        )
        search_results = response.points
    except Exception as e:
        print(f"❌ Could not search database: {e}")
        return

    if not search_results:
        print("No matching chunks found.")
        return

    # Parse and display Point IDs and payload metadata
    for point in search_results:
        payload = point.payload or {}
        chunk_id = point.id
        text = payload.get("text", "")
        video_title = payload.get("video_title", "Unknown Title")
        timestamp = payload.get("formatted_start", "00:00:00")

        print(f"📌 CHUNK ID:    {chunk_id}")
        print(f"📺 VIDEO:       {video_title} [{timestamp}]")
        print(f"📄 TEXT SNIPPET: {text[:150]}...\n")


if __name__ == "__main__":
    # Target phrase used to find chunk IDs when populating golden_dataset.json
    target_phrase = (
        "A former high-ranking Ukrainian intelligence officer, Dmytro Kozyura, "
        "has been sentenced to life in prison for spying for Russia. Once a "
        "counter-terrorism chief in Kyiv's security service, he was convicted of "
        "high treason for sharing state secrets with Moscow."
    )
    search_for_ids(target_phrase)