# run_eval_pipeline.py
import json
import ollama
from typing import TypedDict, List, Dict, Any, Generator
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langfuse import observe

# 1. IMPORT CONFIG FIRST (This injects the keys into os.environ)
from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
)

# 2. IMPORT LANGFUSE SECOND (Now it can find the keys!)
from eval_retrieval import evaluate_retrieval
from eval_generation import evaluate_generation


def get_qdrant_client() -> QdrantClient:
    """Forces local disk storage since we are not running a Qdrant server."""
    return QdrantClient(path=str(DB_DIR))


@observe(name="Run_Automated_RAG_Evaluation_Suite")
def run_automated_evaluation():
    print("=" * 60)
    print(" 🚀 INITIATING AUTOMATED THREAT INTEL RAG EVALUATION ")
    print("=" * 60)

    dataset_path = Path("data/golden_dataset.json")
    if not dataset_path.exists():
        print("❌ Error: data/golden_dataset.json not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # 1. Connect to Qdrant Vector Store
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ Error: Qdrant collection '{COLLECTION_NAME}' not found.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total_ndcg = 0.0
    total_hit_rate = 0.0
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_correctness = 0.0
    valid_samples = 0

    # 2. Iterate through Golden Dataset
    for i, scenario in enumerate(golden_data, 1):
        query = scenario["query"]
        print(f"\n[Test Case {i}/{len(golden_data)}]: {query[:60]}...")

        # --- A. RETRIEVAL PHASE (Qdrant) ---
        # [FIX]: Properly encode the query into a vector before searching
        query_vector = model.encode(query, show_progress_bar=False).tolist()
        
        # [FIX]: Use the new query_points API
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
        )
        search_results = response.points

        retrieved_ids = []
        retrieved_docs = []

        for point in search_results:
            payload = point.payload or {}
            chunk_identifier = payload.get("chunk_id", str(point.id))
            retrieved_ids.append(chunk_identifier)
            retrieved_docs.append(payload.get("text", ""))

        context_block = "\n---\n".join(retrieved_docs)

        # Grade Retrieval Metrics
        retrieval_metrics = evaluate_retrieval(
            retrieved_chunk_ids=retrieved_ids,
            ground_truth_ids=scenario["ground_truth_chunk_ids"],
            k=5,
        )
        print(f"   ↳ Retrieval NDCG: {retrieval_metrics['NDCG']} | Hit Rate: {retrieval_metrics['Hit_Rate']}")

        # --- B. GENERATION PHASE (Ollama) ---
        user_prompt = f"Context Data:\n{context_block}\n\nUser Intelligence Request: {query}"

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Threat Intelligence Analyst. Answer based ONLY on context."},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            generated_answer = response["message"]["content"]

            # Grade Generation using LLM-as-a-Judge
            gen_metrics = evaluate_generation(
                query=query,
                context=context_block,
                generated_answer=generated_answer,
                ground_truth=scenario["ground_truth_answer"],
            )

            faithfulness = gen_metrics.get("Faithfulness", 0)
            relevancy = gen_metrics.get("Answer_Relevancy", 0)
            correctness = gen_metrics.get("Answer_Correctness", 0)

            print(f"   ↳ Gen Faithfulness: {faithfulness}/5 | Relevancy: {relevancy}/5 | Correctness: {correctness}/5")

            total_ndcg += retrieval_metrics["NDCG"]
            total_hit_rate += retrieval_metrics["Hit_Rate"]
            total_faithfulness += faithfulness
            total_relevancy += relevancy
            total_correctness += correctness
            valid_samples += 1

        except Exception as e:
            print(f"❌ Generation failed for this scenario: {e}")

# 3. Aggregate Final Benchmark System Report
    if valid_samples > 0:
        print("\n" + "=" * 60)
        print(" 🏁 FINAL SYSTEM RAG METRICS REPORT ")
        print("=" * 60)
        print(f"Mean Hit Rate               : {total_hit_rate / valid_samples:.3f}")
        print(f"Mean NDCG (Retrieval)       : {total_ndcg / valid_samples:.3f}")
        print(f"Mean Faithfulness           : {total_faithfulness / valid_samples:.2f} / 5.0")
        print(f"Mean Answer Relevancy       : {total_relevancy / valid_samples:.2f} / 5.0")
        print(f"Mean Answer Correctness     : {total_correctness / valid_samples:.2f} / 5.0")
        print("=" * 60)
        
        # [NEW]: Return the metrics so Streamlit can build a UI dashboard
        return {
            "Hit_Rate": total_hit_rate / valid_samples,
            "NDCG": total_ndcg / valid_samples,
            "Faithfulness": total_faithfulness / valid_samples,
            "Relevancy": total_relevancy / valid_samples,
            "Correctness": total_correctness / valid_samples
        }
    return None

if __name__ == "__main__":
    run_automated_evaluation()