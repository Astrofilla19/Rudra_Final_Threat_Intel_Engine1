# ingest.py
import sys
from pathlib import Path

from utils.downloader import download_youtube_audio
from utils.transcriber import transcribe_audio
from utils.semantic_chunker import generate_semantic_chunks
from utils.vector_store import ingest_to_vector_store

def run_ingestion(txt_file_path: str):
    path = Path(txt_file_path)
    if not path.exists():
        print("❌ File not found. Terminating ingestion.")
        return

    with open(path, 'r', encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"\n📦 Loaded {len(urls)} targets for batch ingestion.")

    for index, video_url in enumerate(urls, 1):
        print(f"\n--- Processing Target {index}/{len(urls)}: {video_url} ---")
        try:
            # 1. Direct audio download via yt-dlp
            audio_file = download_youtube_audio(video_url)
            video_title = Path(audio_file).stem

            # 2. Whisper transcription
            transcript_json = transcribe_audio(audio_file)
            
            # 3. Dynamic semantic chunking (with VRAM protection)
            chunks_json = generate_semantic_chunks(transcript_json)

            # 4. Qdrant vector storage ingestion (with deterministic UUIDs)
            ingest_to_vector_store(chunks_json, video_title=video_title)
            print(f"🌟 Target '{video_title}' loaded successfully into Qdrant.")
        except Exception as e:
            print(f"⚠️ Error processing {video_url}. Details: {e}", file=sys.stderr)

    print("\n✅ Batch Ingestion Complete.")

if __name__ == "__main__":
    txt_file = input("Enter path to YouTube links (.txt file): ").strip()
    run_ingestion(txt_file)