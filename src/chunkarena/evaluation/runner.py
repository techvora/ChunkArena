"""Evaluation runner.

Orchestrates the full benchmark:

    1. Load the golden dataset.
    2. Load every Qdrant collection and build its BM25 index.
    3. Compute per-collection chunk stats.
    4. For every (method, technique, question) triple, run retrieval and
       score every metric.
    5. Aggregate to a per-(method, technique) summary, add a composite
       score and per-metric verdicts.
    6. Run pairwise paired t-tests per technique on nDCG.
    7. Build the 8-sheet Excel report.
    8. Write raw_results.csv, chunk_stats.csv, summary.csv.

Logic is unchanged from the original single-file evaluate.py; the only
difference is that metrics, retrieval, chunk stats and the Excel builder
are imported from their respective packages.
"""

import time
import warnings
import pandas as pd
from tqdm import tqdm

from chunkarena.config import (
    CHUNK_METHODS, FINAL_K, RETRIEVAL_K, SOFT_THRESHOLD,
    COMPOSITE_WEIGHTS, METHOD_PARAMS,
    RAW_RESULTS_CSV, CHUNK_STATS_CSV, SUMMARY_CSV,
)
from chunkarena.metrics import (
    get_embedding, hit_at_k, mrr_score, precision_at_k, ndcg_at_k,
    recall_at_k, avg_rank_score, redundancy_score, boundary_score,
    token_cost, context_relevance, faithfulness, answer_correctness,
    threshold_verdict,
)
from .data_loader import load_questions
from .chunk_store import build_all_stores
from .chunk_stats import chunk_stats
from .retrieval import dense_search, hybrid_search, rerank
from chunkarena.reporting import build_workbook


warnings.filterwarnings("ignore")


def run():
    """Execute the end-to-end benchmark and write every output artifact.

    Loads the golden dataset and every Qdrant collection, computes
    per-collection chunk stats, then iterates over every
    ``(method, technique, question)`` triple: runs dense and hybrid
    retrieval, reranks each, and scores the result with the full metric
    suite (hit, MRR, precision, nDCG, recall, avg rank, redundancy,
    boundary, token cost, context relevance, faithfulness, answer
    correctness). Aggregates to a per-``(method, technique)`` summary,
    adds a composite score and threshold verdicts, writes raw/summary/
    stats CSVs, and finally hands everything to the Excel report
    builder. Latency is accounted honestly: hybrid latency adds the
    dense portion back in because the dense call is served from cache
    inside ``hybrid_search``.
    """
    print(f"SOFT_THRESHOLD   = {SOFT_THRESHOLD}")
    print(f"COMPOSITE_WEIGHTS = {COMPOSITE_WEIGHTS}")

    questions_data = load_questions()

    print("Loading models...")
    print("  Models ready")

    build_all_stores()

    # ============================================================
    # CHUNK QUALITY STATS
    # ============================================================
    print("\nComputing chunk quality stats...")
    stats_rows = []
    for method in CHUNK_METHODS:
        print(f"  {method}...")
        stats_rows.append(chunk_stats(method))
    stats_df = pd.DataFrame(stats_rows)

    # ============================================================
    # MAIN EVALUATION LOOP
    # ============================================================
    print("\nRunning evaluation...")
    raw_results = []

    for method in CHUNK_METHODS:
        print(f"\n  Method: {method}")
        for q in tqdm(questions_data, desc=f"    {method}"):
            query      = q["question"]
            gold_spans = q["gold_spans"]
            q_emb      = get_embedding(query)

            t0 = time.perf_counter()
            dense_ids, dense_texts = dense_search(query, q_emb, method, RETRIEVAL_K)
            t_dense_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            hybrid_ids, hybrid_texts = hybrid_search(query, q_emb, method, RETRIEVAL_K)
            t_hybrid_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            dr_ids, dr_texts = rerank(query, dense_ids, dense_texts, FINAL_K, tag="dense")
            t_dense_rr_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            hr_ids, hr_texts = rerank(query, hybrid_ids, hybrid_texts, FINAL_K, tag="hybrid")
            t_hybrid_rr_ms = (time.perf_counter() - t0) * 1000.0

            # Honest hybrid latency: hybrid_search internally calls
            # dense_search which is cache-served, so t_hybrid_ms alone
            # captures only the BM25+RRF portion. The real hybrid cost
            # is dense retrieval + BM25 + RRF, so we add t_dense_ms.
            t_hybrid_full_ms = t_dense_ms + t_hybrid_ms

            techniques = {
                "dense"         : ((dense_ids[:FINAL_K],  dense_texts[:FINAL_K]),  t_dense_ms),
                "hybrid"        : ((hybrid_ids[:FINAL_K], hybrid_texts[:FINAL_K]), t_hybrid_full_ms),
                "dense_rerank"  : ((dr_ids, dr_texts),                              t_dense_ms + t_dense_rr_ms),
                "hybrid_rerank" : ((hr_ids, hr_texts),                              t_hybrid_full_ms + t_hybrid_rr_ms),
            }

            gold_answer = q["gold_answer"]
            for tech, ((_ids, texts), latency_ms) in techniques.items():
                h   = hit_at_k(texts, gold_spans, FINAL_K)
                m   = mrr_score(texts, gold_spans)
                p   = precision_at_k(texts, gold_spans, FINAL_K)
                n   = ndcg_at_k(texts, gold_spans, FINAL_K)
                rc  = recall_at_k(texts, gold_spans, FINAL_K)
                ar  = avg_rank_score(texts, gold_spans)
                rd  = redundancy_score(texts)
                b   = boundary_score(texts)
                tc  = token_cost(texts)
                cr  = context_relevance(query, texts)
                fa  = faithfulness(texts, gold_answer)
                ac  = answer_correctness(texts, gold_spans)
                miss = 1 if pd.isna(ar) else 0

                raw_results.append({
                    "method"            : method,
                    "technique"         : tech,
                    "question_id"       : q["id"],
                    "question"          : query,
                    "hit@k"             : h,
                    "mrr"               : round(m,  4),
                    "precision@k"       : round(p,  4),
                    "ndcg@k"            : round(n,  4),
                    "recall@k"          : round(rc, 4),
                    "avg_rank"          : ar,
                    "miss"              : miss,
                    "redundancy"        : rd,
                    "diversity"         : round(1 - rd, 4),
                    "boundary"          : b,
                    "token_cost"        : tc,
                    "latency_ms"        : round(latency_ms, 3),
                    "context_relevance" : cr,
                    "faithfulness"      : fa,
                    "answer_correctness": ac,
                })

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(RAW_RESULTS_CSV, index=False)
    stats_df.to_csv(CHUNK_STATS_CSV, index=False)
    print("\nRaw results saved.")

    # ============================================================
    # SUMMARY - aggregate per (method, technique)
    # ============================================================
    summary_df = raw_df.groupby(["method", "technique"]).agg(
        hit_at_k        = ("hit@k",       "mean"),
        mrr             = ("mrr",         "mean"),
        precision_at_k  = ("precision@k", "mean"),
        ndcg_at_k       = ("ndcg@k",      "mean"),
        recall_at_k     = ("recall@k",    "mean"),
        avg_rank        = ("avg_rank",    "mean"),
        miss_rate       = ("miss",        "mean"),
        redundancy      = ("redundancy",  "mean"),
        diversity       = ("diversity",   "mean"),
        boundary        = ("boundary",    "mean"),
        avg_token_cost  = ("token_cost",  "mean"),
        avg_latency_ms  = ("latency_ms",  "mean"),
        context_relevance  = ("context_relevance",  "mean"),
        faithfulness       = ("faithfulness",       "mean"),
        answer_correctness = ("answer_correctness", "mean"),
        n_questions     = ("question_id", "count"),
    ).round(4).reset_index()

    summary_df["answer_rate"] = (1 - summary_df["miss_rate"]).round(4)

    summary_df["composite_score"] = sum(
        summary_df[m] * w for m, w in COMPOSITE_WEIGHTS.items()
    ).round(4)

    summary_df = summary_df.sort_values(
        "composite_score", ascending=False
    ).reset_index(drop=True)
    summary_df["rank"] = summary_df.index + 1

    for metric in ["hit_at_k", "mrr", "ndcg_at_k", "recall_at_k",
                   "redundancy", "composite_score"]:
        summary_df[f"{metric}_verdict"] = summary_df[metric].apply(
            lambda v: threshold_verdict(metric, v)
        )

    summary_df.to_csv(SUMMARY_CSV, index=False)

    overall_best = summary_df.iloc[0]
    print(f"\n{'='*60}")
    print(f"OVERALL BEST: {overall_best['method']} + {overall_best['technique']}")
    print(f"  Composite   : {overall_best['composite_score']}")
    print(f"  nDCG@{FINAL_K}      : {overall_best['ndcg_at_k']}")
    print(f"  MRR         : {overall_best['mrr']}")
    print(f"  Hit@{FINAL_K}       : {overall_best['hit_at_k']}")
    print(f"  Miss rate   : {overall_best['miss_rate']:.1%}")
    print(f"  Redundancy  : {overall_best['redundancy']} (mean pairwise similarity)")
    print(f"{'='*60}")

    # ============================================================
    # EXCEL WORKBOOK
    # ============================================================
    build_workbook(
        summary_df=summary_df,
        raw_df=raw_df,
        final_k=FINAL_K,
        method_params=METHOD_PARAMS,
    )

    print("\nAll outputs:")
    print(f"   {RAW_RESULTS_CSV}       per-question scores (with miss column)")
    print(f"   {CHUNK_STATS_CSV}       chunk quality per method")
    print(f"   {SUMMARY_CSV}           aggregated scores + verdicts ranked")
    print("   benchmark_report.xlsx  8-sheet formatted report\n")
    print("Key design choices:")
    print(f"  - SOFT_THRESHOLD      : {SOFT_THRESHOLD} (semantic relevance)")
    print(f"  - Redundancy          : mean pairwise cosine similarity (standard)")
    print("  - Rerank cache key    : includes technique tag (no collision)")
