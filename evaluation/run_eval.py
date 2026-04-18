from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retriever import Retriever


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def dcg(retrieved_ids, relevant_ids, k):
    gain = 0.0
    for i, r in enumerate(retrieved_ids[:k]):
        if r in relevant_ids:
            gain += 1.0 / np.log2(i + 2)
    return gain


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    ideal = dcg(list(relevant_ids), relevant_ids, k)
    return 0.0 if ideal == 0 else dcg(retrieved_ids, relevant_ids, k) / ideal


def citation_to_doc_ids(citation_str, all_pages):
    matched = set()
    try:
        parts = citation_str.split(", p. ")
        pdf_name = parts[0].strip()
        page_num = int(parts[1].strip())
        for p in all_pages:
            if p.pdf_name == pdf_name and p.page_number == page_num:
                matched.add(p.doc_id)
    except Exception:
        pass
    return matched


def run_evaluation(
    retriever,
    benchmark_path="evaluation/benchmark_queries.json",
    top_k=5,
    output_path="evaluation/results.json",
):
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    all_pages = retriever._pages
    results = []
    latencies = []
    modality_scores = {}

    print(f"\nRunning evaluation: {len(benchmark)} queries | top_k={top_k}")
    print("=" * 70)

    for q in benchmark:
        query = q["query"]
        relevant_citations = q.get("relevant_pages", [])
        modality = q.get("modality", "unknown")

        relevant_ids = set()
        for cit in relevant_citations:
            relevant_ids |= citation_to_doc_ids(cit, all_pages)

        t0 = time.time()
        retrieved = retriever.retrieve(query, top_k=top_k, load_images=False)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [r.record.doc_id for r in retrieved]
        retrieved_citations = [r.citation for r in retrieved]

        r1   = recall_at_k(retrieved_ids, relevant_ids, 1)
        r5   = recall_at_k(retrieved_ids, relevant_ids, top_k)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, top_k)

        row = {
            "query": query,
            "modality": modality,
            "retrieved": retrieved_citations,
            "relevant": relevant_citations,
            "latency_ms": round(latency_ms, 1),
            "recall@1": round(r1, 3),
            f"recall@{top_k}": round(r5, 3),
            f"ndcg@{top_k}": round(ndcg, 3),
        }
        results.append(row)

        modality_scores.setdefault(modality, []).append(ndcg)

        hit = "✓" if r1 > 0 else "✗"
        print(f"  {hit} [{modality:15s}] {latency_ms:5.0f}ms | "
              f"R@1={r1:.2f} nDCG@{top_k}={ndcg:.2f} | {query[:48]}...")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Queries evaluated : {len(results)}")
    print(f"Mean latency      : {np.mean(latencies):.0f}ms ± {np.std(latencies):.0f}ms")
    print(f"Mean Recall@1     : {np.mean([r['recall@1'] for r in results]):.3f}")
    print(f"Mean Recall@{top_k}    : {np.mean([r[f'recall@{top_k}'] for r in results]):.3f}")
    print(f"Mean nDCG@{top_k}     : {np.mean([r[f'ndcg@{top_k}'] for r in results]):.3f}")
    print(f"\nBy modality (nDCG@{top_k}):")
    for mod, scores in sorted(modality_scores.items()):
        print(f"  {mod:18s}: {np.mean(scores):.3f}  (n={len(scores)})")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "num_queries": len(results),
                "mean_latency_ms": round(float(np.mean(latencies)), 1),
                "mean_recall_at_1": round(float(np.mean([r["recall@1"] for r in results])), 3),
                f"mean_recall_at_{top_k}": round(float(np.mean([r[f"recall@{top_k}"] for r in results])), 3),
                f"mean_ndcg_at_{top_k}": round(float(np.mean([r[f"ndcg@{top_k}"] for r in results])), 3),
                "by_modality": {mod: round(float(np.mean(s)), 3) for mod, s in modality_scores.items()},
            },
            "queries": results,
        }, f, indent=2)

    print(f"\nResults saved → {output_path}")
    return results


if __name__ == "__main__":
    retriever = Retriever(index_dir="data/index", pdf_dir="data/pdfs", top_k=5)
    retriever.load()
    run_evaluation(retriever, top_k=5)