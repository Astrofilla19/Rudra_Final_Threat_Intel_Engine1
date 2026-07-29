# eval_retrieval.py
import math
from typing import List, Dict


def evaluate_retrieval(
    retrieved_chunk_ids: List[str], ground_truth_ids: List[str], k: int = 5
) -> Dict[str, float]:
    """
    Calculates deterministic RAG retrieval metrics at cutoff K.
    """
    # Truncate to top-K results
    retrieved_k = retrieved_chunk_ids[:k]

    # Binary relevance array: 1 if retrieved chunk is in ground truth, else 0
    relevance = [1 if chunk_id in ground_truth_ids else 0 for chunk_id in retrieved_k]

    # 1. Hit Rate (Did we retrieve at least one ground-truth chunk?)
    hit_rate = 1 if sum(relevance) > 0 else 0

    # 2. Precision@K
    precision_at_k = sum(relevance) / k if k > 0 else 0.0

    # 3. Recall@K
    recall_at_k = (
        sum(relevance) / len(ground_truth_ids) if ground_truth_ids else 0.0
    )

    # 4. Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    # 5. Normalized Discounted Cumulative Gain (NDCG@K)
    dcg = sum(
        rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1)
    )

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevance, start=1)
    )

    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "Hit_Rate": hit_rate,
        "Precision": round(precision_at_k, 3),
        "Recall": round(recall_at_k, 3),
        "MRR": round(mrr, 3),
        "NDCG": round(ndcg_at_k, 3),
    }


if __name__ == "__main__":
    retrieved = ["chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"]
    ground_truth = ["chunk_A", "chunk_D", "chunk_Z"]

    metrics = evaluate_retrieval(retrieved, ground_truth, k=5)
    print(f"📊 Retrieval Metrics: {metrics}")