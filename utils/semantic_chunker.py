# utils/semantic_chunker.py
import json
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import CHUNKS_DIR, EMBEDDING_MODEL_NAME

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def sanitize_filename(name: str) -> str:
    """Removes illegal characters from Windows file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def generate_semantic_chunks(transcript_json_path: str, threshold_percentile=65) -> str:
    print("🧩 Executing thematic splitting strategies...")
    
    # Cleaned up line: no extra brackets or syntax errors here
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        return ""

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    sentences = [seg["text"] for seg in segments]
    
    # Process embeddings in batches to prevent OOM errors on large transcripts
    embeddings = model.encode(sentences, batch_size=32, convert_to_tensor=True, show_progress_bar=False)
    embeddings = embeddings.cpu().numpy()

    distances = []
    for i in range(len(embeddings) - 1):
        vec1, vec2 = embeddings[i], embeddings[i + 1]
        cosine_dist = 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))
        distances.append(cosine_dist)

    breakpoint_threshold = np.percentile(distances, threshold_percentile) if distances else 0.5

    chunks = []
    current_chunk_text = []
    current_start = segments[0]["start"]

    for i, seg in enumerate(segments):
        current_chunk_text.append(seg["text"])
        current_end = seg["end"]

        if i < len(distances) and distances[i] > breakpoint_threshold:
            chunks.append({
                "text": " ".join(current_chunk_text),
                "metadata": {
                    "start_time": current_start,
                    "end_time": current_end,
                    "formatted_start": format_timestamp(current_start),
                },
            })
            if i + 1 < len(segments):
                current_start = segments[i + 1]["start"]
            current_chunk_text = []

    if current_chunk_text:
        chunks.append({
            "text": " ".join(current_chunk_text),
            "metadata": {
                "start_time": current_start,
                "end_time": current_end,
                "formatted_start": format_timestamp(current_start),
            },
        })

    video_stem = sanitize_filename(Path(transcript_json_path).stem)
    out_file = CHUNKS_DIR / f"{video_stem}_chunks.json"
    
    # Force create the directory
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

    print(f"✅ Generated {len(chunks)} isolated analytical context chunks for {video_stem}.")
    return str(out_file)