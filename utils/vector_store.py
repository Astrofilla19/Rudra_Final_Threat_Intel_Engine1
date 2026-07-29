# utils/vector_store.py
import json
import hashlib
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def ingest_to_vector_store(chunks_json_path: str, video_title: str):
    print("🗄️ Appending assets to persistent local Qdrant vector storage...")

    # Force strict local disk connection
    client = QdrantClient(path=str(DB_DIR))

    # Initialize model to compute vector dimensions and embeddings
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # [FIX 1]: Resolve the FutureWarning for newer versions of sentence-transformers
    try:
        vector_size = model.get_embedding_dimension()
    except AttributeError:
        vector_size = model.get_sentence_embedding_dimension()

    # Ensure collection exists with Cosine metric
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("⚠️ No chunks found to ingest.")
        client.close()
        return

    # Extract texts and compute embeddings in batches to protect VRAM
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    video_hash = hashlib.sha256(video_title.encode()).hexdigest()[:16]

    points = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_hash}_chunk_{index:04d}"))

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "video_title": video_title,
                    "start_time": chunk["metadata"]["start_time"],
                    "end_time": chunk["metadata"]["end_time"],
                    "formatted_start": chunk["metadata"]["formatted_start"],
                },
            )
        )

    # Upsert points directly to Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Qdrant database successfully updated with {len(points)} new chunks for '{video_title}'.")
    
    # [CRITICAL FIX 2]: Flush the data to the hard drive and release the SQLite lock!
    client.close()