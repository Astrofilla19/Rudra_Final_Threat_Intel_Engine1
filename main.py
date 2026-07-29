# main.py
import sys
from pathlib import Path

from utils.downloader import download_youtube_audio
from utils.transcriber import transcribe_audio
from utils.semantic_chunker import generate_semantic_chunks
from utils.vector_store import ingest_to_vector_store
from threat_intel_engine import run_intelligence_query

def run_pipeline():
    print("=" * 60)
    print("   VIDEO TRANSCRIPTION THREAT INTELLIGENCE SYSTEM FRAMEWORK")
    print("=" * 60)

    # Option to skip ingestion if the user only wants to query the existing Qdrant database
    do_ingest = input("\nDo you want to ingest new YouTube targets? (y/n): ").strip().lower()

    if do_ingest in ["y", "yes"]:
        txt_file = input("\nEnter path to YouTube links (.txt file): ").strip()
        if not Path(txt_file).exists():
            print("❌ File not found. Skipping ingestion phase.")
        else:
            with open(txt_file, 'r', encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]

            print(f"\n📦 Loaded {len(urls)} targets for batch ingestion.")

            for index, video_url in enumerate(urls, 1):
                print(f"\n--- Processing Target {index}/{len(urls)}: {video_url} ---")
                try:
                    audio_file = download_youtube_audio(video_url)
                    video_title = Path(audio_file).stem

                    transcript_json = transcribe_audio(audio_file)
                    chunks_json = generate_semantic_chunks(transcript_json)

                    ingest_to_vector_store(chunks_json, video_title=video_title)
                    print(f"🌟 Target '{video_title}' loaded successfully into Qdrant.")
                except Exception as e:
                    print(f"⚠️ Error processing {video_url}. Skipping to next. Details: {e}", file=sys.stderr)

            print("\n✅ Bulk Pipeline processing complete.")

    # Drop down into the interactive analyst query runtime loop
    print("\n🤖 Transitioning to Threat Intelligence Analyst CLI Interface...")
    while True:
        print("-" * 55)
        query = input("Ask a Threat Intelligence Query (or type 'exit' / 'quit'): ").strip()

        if query.lower() in ["exit", "quit"]:
            print("👋 Securing framework databases. Exiting execution runtime.")
            break
        if not query:
            continue

        try:
            run_intelligence_query(query)
        except Exception as e:
            print(f"❌ Query execution failed: {e}")

if __name__ == "__main__":
    run_pipeline()