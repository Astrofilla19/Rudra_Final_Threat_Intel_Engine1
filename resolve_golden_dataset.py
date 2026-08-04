# resolve_golden_dataset.py
#
# Run this LOCALLY, in your project folder (where DB_DIR / the Qdrant
# on-disk collection actually lives). It does NOT call an embedding model
# and does NOT re-run search — it simply scrolls every point already in
# your collection and matches it to the draft golden-dataset entries by
# exact text, so you get the *real* point IDs Qdrant assigned rather than
# a guessed/hand-typed chunk_id string.
#
# Why this matters: vector_store.py generates point IDs like
#   uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_hash}_chunk_{index:04d}")
# which produces a real UUID (e.g. "3fa85f64-5717-4562-b3fc-2c963f66afa6"),
# NOT a human-readable string like "39e959bb_chunk_0037". Your eval
# pipeline (run_eval_pipeline.py) falls back to str(point.id) -- the real
# UUID -- whenever payload["chunk_id"] is missing, which it always is
# with the current vector_store.py. So ground_truth_chunk_ids MUST be
# those real UUIDs or Hit_Rate/NDCG will silently be 0 for every query.
#
# Usage:
#   python resolve_golden_dataset.py
#
# Reads:  golden_dataset_draft.json  (has ground_truth_chunk_texts)
# Writes: data/golden_dataset.json   (has ground_truth_chunk_ids, ready
#                                      for run_eval_pipeline.py)

import json
from pathlib import Path
from qdrant_client import QdrantClient
from config import DB_DIR, COLLECTION_NAME


def normalize(s: str) -> str:
    """Collapse whitespace so minor formatting differences don't block a match."""
    return " ".join(s.split()).strip()


def load_all_points(client: QdrantClient):
    """Scroll the entire collection into memory: {normalized_text: [point_ids]}."""
    text_to_ids = {}
    next_page = None
    total = 0
    while True:
        points, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            total += 1
            text = normalize((p.payload or {}).get("text", ""))
            text_to_ids.setdefault(text, []).append(str(p.id))
        if next_page is None:
            break
    print(f"📦 Loaded {total} points from Qdrant collection '{COLLECTION_NAME}'")
    return text_to_ids


def resolve():
    draft_path = Path("golden_dataset_draft.json")
    if not draft_path.exists():
        print("❌ golden_dataset_draft.json not found in current directory.")
        return

    with open(draft_path, "r", encoding="utf-8") as f:
        draft = json.load(f)

    client = QdrantClient(path=str(DB_DIR))
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' not found at {DB_DIR}.")
        return

    text_to_ids = load_all_points(client)

    resolved = []
    unresolved_count = 0

    for i, item in enumerate(draft, 1):
        chunk_ids = []
        missing = []
        for chunk_text in item["ground_truth_chunk_texts"]:
            key = normalize(chunk_text)
            matches = text_to_ids.get(key)
            if matches:
                # Usually exactly one match; if duplicates exist, take all.
                chunk_ids.extend(matches)
            else:
                missing.append(chunk_text[:80])

        status = "✅" if not missing else "⚠️ "
        print(f"{status} [{i}/{len(draft)}] {item['query'][:60]}...")
        if missing:
            unresolved_count += 1
            for m in missing:
                print(f"     ❌ no exact text match found for: {m}...")

        resolved.append({
            "query": item["query"],
            "ground_truth_chunk_ids": chunk_ids,
            "ground_truth_answer": item["ground_truth_answer"],
            "video_source": item["video_source"],
        })

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Wrote {out_path} ({len(resolved)} entries)")
    if unresolved_count:
        print(f"⚠️  {unresolved_count} entries had at least one unmatched chunk.")
        print("   Likely cause: that video hasn't been ingested yet, or the")
        print("   semantic chunker split the text differently than the sample")
        print("   file used to draft the query. Fix by re-checking with")
        print("   find_chunk_ids.py against the actual chunk text in Qdrant.")
    else:
        print("🎉 All entries resolved cleanly against your live collection.")


if __name__ == "__main__":
    resolve()
