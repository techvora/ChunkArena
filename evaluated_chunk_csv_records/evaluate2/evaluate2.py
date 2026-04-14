# evaluate_chunking_strategies_fixed.py
import json
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ========== CONFIGURATION ==========
CSV_PATH = "/home/root473/Documents/POC/ChunkArena/Golden_dataset/Banking_system.csv"
SIMILARITY_THRESHOLD = 0.75   # for fact coverage
FINAL_EVAL_K = 5              # all metrics computed on top-K chunks
RETRIEVAL_K_NON_RERANK = FINAL_EVAL_K      # non-rerank retrieve exactly K
RETRIEVAL_K_RERANK = 50                     # rerank retrieve larger pool
RERANK_OUTPUT_K = FINAL_EVAL_K              # reranker outputs exactly K
# ====================================

# ------------------------------
# 1. Load golden CSV and extract atomic facts
# ------------------------------
df = pd.read_csv(CSV_PATH)

questions_data = []
for _, row in df.iterrows():
    question = row["Question"]
    answer = row["Answer"] if pd.notna(row["Answer"]) else ""
    facts_str = row["Facts"] if pd.notna(row["Facts"]) else ""
    atomic_facts = [fact.strip() for fact in facts_str.split(",") if fact.strip()]
    if not atomic_facts and answer:
        atomic_facts = [answer]
    questions_data.append({
        "id": len(questions_data),
        "question": question,
        "answer": answer,
        "atomic_facts": atomic_facts
    })

# ------------------------------
# 2. Initialize models and Qdrant client
# ------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")
embedder = SentenceTransformer('BAAI/bge-m3', device=device)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

client = QdrantClient(host="localhost", port=6333)

CHUNK_METHODS = ["fixed_size", "overlapping", "sentence", "paragraph", "recursive", "header", "semantic"]

# ------------------------------
# 3. Preload all chunks and BM25 models
# ------------------------------
def get_all_chunks(collection_name):
    scroll = client.scroll(collection_name=collection_name, limit=10000, with_payload=True)
    points = scroll[0]
    ids = [p.id for p in points]
    texts = [p.payload["text"] for p in points]
    return ids, texts

all_chunks_cache = {}
all_chunks_text_dict = {}
bm25_models = {}

print("Loading collections and building BM25 models...")
for method in CHUNK_METHODS:
    try:
        ids, texts = get_all_chunks(method)
        all_chunks_cache[method] = (ids, texts)
        all_chunks_text_dict[method] = {cid: text for cid, text in zip(ids, texts)}
        tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in texts]
        bm25_models[method] = BM25Okapi(tokenized_corpus)
        print(f"  {method}: {len(ids)} chunks loaded")
    except Exception as e:
        print(f"  Failed to load {method}: {e}")
        all_chunks_cache[method] = ([], [])
        bm25_models[method] = None

# ------------------------------
# 4. Embed all questions once
# ------------------------------
questions = [q["question"] for q in questions_data]
question_embs = embedder.encode(questions, show_progress_bar=True, convert_to_numpy=True)

# ------------------------------
# 5. Retrieval functions
# ------------------------------
def dense_search(query_emb, collection_name, top_k):
    results = client.query_points(
        collection_name=collection_name,
        query=query_emb.tolist(),
        limit=top_k,
        with_payload=False
    )
    ids = [hit.id for hit in results.points]
    texts = [all_chunks_text_dict[collection_name][cid] for cid in ids]
    return ids, texts

def reciprocal_rank_fusion(dense_results, bm25_results, k, rrf_k=60):
    scores = defaultdict(float)
    for rank, (cid, _) in enumerate(dense_results):
        scores[cid] += 1.0 / (rrf_k + rank + 1)
    for rank, (cid, _) in enumerate(bm25_results):
        scores[cid] += 1.0 / (rrf_k + rank + 1)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids[:k]

def hybrid_search(query_text, query_emb, collection_name, top_k):
    dense_resp = client.query_points(
        collection_name=collection_name,
        query=query_emb.tolist(),
        limit=100,
        with_payload=False
    )
    dense_results = [(hit.id, hit.score) for hit in dense_resp.points]
    
    bm25 = bm25_models[collection_name]
    if bm25 is None:
        return dense_search(query_emb, collection_name, top_k)
    
    tokenized_query = re.findall(r'\w+', query_text.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    all_ids, _ = all_chunks_cache[collection_name]
    bm25_results = [(all_ids[i], bm25_scores[i]) for i in range(len(all_ids))]
    bm25_results.sort(key=lambda x: x[1], reverse=True)
    bm25_results = bm25_results[:100]
    
    fused_ids = reciprocal_rank_fusion(dense_results, bm25_results, k=top_k)
    texts = [all_chunks_text_dict[collection_name][cid] for cid in fused_ids]
    return fused_ids, texts

def rerank(query, retrieved_ids, retrieved_texts, top_k):
    if not retrieved_texts:
        return [], []
    pairs = [[query, text] for text in retrieved_texts]
    scores = cross_encoder.predict(pairs)
    sorted_idx = np.argsort(scores)[::-1][:top_k]
    reranked_ids = [retrieved_ids[i] for i in sorted_idx]
    reranked_texts = [retrieved_texts[i] for i in sorted_idx]
    return reranked_ids, reranked_texts

# ------------------------------
# 6. Metrics using atomic facts
# ------------------------------
def semantic_similarity(text1, text2):
    emb1 = embedder.encode([text1], convert_to_numpy=True)
    emb2 = embedder.encode([text2], convert_to_numpy=True)
    return cosine_similarity(emb1, emb2)[0][0]

def is_fact_covered(chunk_text, fact, threshold=SIMILARITY_THRESHOLD):
    return semantic_similarity(chunk_text, fact) >= threshold

def fact_coverage_at_k(retrieved_texts, atomic_facts, k, threshold=SIMILARITY_THRESHOLD):
    covered = set()
    for chunk in retrieved_texts[:k]:
        for fact in atomic_facts:
            if is_fact_covered(chunk, fact, threshold):
                covered.add(fact)
    return len(covered) / len(atomic_facts) if atomic_facts else 1.0

def mrr_fact(retrieved_texts, atomic_facts, threshold=SIMILARITY_THRESHOLD):
    for rank, chunk in enumerate(retrieved_texts, 1):
        if any(is_fact_covered(chunk, fact, threshold) for fact in atomic_facts):
            return 1.0 / rank
    return 0.0

def precision_at_k(retrieved_texts, atomic_facts, k, threshold=SIMILARITY_THRESHOLD):
    top_k = retrieved_texts[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for chunk in top_k if any(is_fact_covered(chunk, fact, threshold) for fact in atomic_facts))
    return relevant / k

# FIX: Correct nDCG with proper ideal graded relevance
def ndcg_graded_at_k(retrieved_texts, atomic_facts, k, threshold=SIMILARITY_THRESHOLD):
    """Graded nDCG where each chunk gets a grade = number of NEW facts it covers.
       Ideal DCG: first chunk covers ALL facts, subsequent chunks cover 0."""
    grades = []
    covered_so_far = set()
    for chunk in retrieved_texts[:k]:
        new_facts = 0
        for fact in atomic_facts:
            if fact not in covered_so_far and is_fact_covered(chunk, fact, threshold):
                new_facts += 1
                covered_so_far.add(fact)
        grades.append(new_facts)
    
    total_facts = len(atomic_facts)
    # Ideal: first chunk covers all facts, rest 0
    ideal_grades = [total_facts] + [0] * (k - 1)
    
    dcg = sum(g / np.log2(idx+2) for idx, g in enumerate(grades))
    idcg = sum(ig / np.log2(idx+2) for idx, ig in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else 0.0

def semantic_redundancy(retrieved_texts, k):
    """Average pairwise cosine similarity among top-k chunks."""
    top = retrieved_texts[:k]
    if len(top) < 2:
        return 0.0
    embs = embedder.encode(top, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embs)
    n = len(top)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += sim_matrix[i, j]
            count += 1
    return total / count if count > 0 else 0.0

def boundary_accuracy(retrieved_texts, k):
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    punct_end = sum(1 for c in top if c.strip() and c.strip()[-1] in '.!?')
    return punct_end / len(top)

# ------------------------------
# 7. Evaluate one chunking method with standard ratios
# ------------------------------
def evaluate_method(method_name, questions_data, k=FINAL_EVAL_K):
    if method_name not in all_chunks_cache or len(all_chunks_cache[method_name][0]) == 0:
        print(f"Skipping {method_name}: no chunks loaded")
        return None
    
    techniques = {
        "Dense": lambda q, q_emb: dense_search(q_emb, method_name, top_k=RETRIEVAL_K_NON_RERANK),
        "Hybrid": lambda q, q_emb: hybrid_search(q, q_emb, method_name, top_k=RETRIEVAL_K_NON_RERANK),
        "Dense+Rerank": lambda q, q_emb: rerank(q, *dense_search(q_emb, method_name, top_k=RETRIEVAL_K_RERANK), top_k=RERANK_OUTPUT_K),
        "Hybrid+Rerank": lambda q, q_emb: rerank(q, *hybrid_search(q, q_emb, method_name, top_k=RETRIEVAL_K_RERANK), top_k=RERANK_OUTPUT_K),
    }
    
    results_tech = {}
    
    for tech_name, search_func in tqdm(techniques.items(), desc=f"  {method_name}"):
        per_question_records = []
        
        for q_idx, q_data in enumerate(questions_data):
            question = q_data["question"]
            atomic_facts = q_data["atomic_facts"]
            if not atomic_facts:
                continue
            
            q_emb = question_embs[q_idx]
            retrieved_ids, retrieved_texts = search_func(question, q_emb)
            
            coverage = fact_coverage_at_k(retrieved_texts, atomic_facts, k)
            mrr = mrr_fact(retrieved_texts, atomic_facts)
            precision = precision_at_k(retrieved_texts, atomic_facts, k)
            ndcg = ndcg_graded_at_k(retrieved_texts, atomic_facts, k)
            redundancy = semantic_redundancy(retrieved_texts, k)
            boundary = boundary_accuracy(retrieved_texts, k)
            
            top_chunks = retrieved_texts[:3] if retrieved_texts else []
            
            per_question_records.append({
                "question_id": q_data["id"],
                "question": question,
                "fact_coverage": coverage,
                "mrr": mrr,
                "precision": precision,
                "ndcg": ndcg,
                "redundancy": redundancy,
                "boundary_accuracy": boundary,
                "retrieved_chunks_sample": " ||| ".join(top_chunks[:3])
            })
        
        avg_metrics = {
            "fact_coverage": np.mean([r["fact_coverage"] for r in per_question_records]),
            "mrr": np.mean([r["mrr"] for r in per_question_records]),
            "precision": np.mean([r["precision"] for r in per_question_records]),
            "ndcg": np.mean([r["ndcg"] for r in per_question_records]),
            "redundancy": np.mean([r["redundancy"] for r in per_question_records]),
            "boundary_accuracy": np.mean([r["boundary_accuracy"] for r in per_question_records]),
        }
        
        results_tech[tech_name] = {
            "averages": avg_metrics,
            "per_question": per_question_records
        }
    
    return results_tech

# ------------------------------
# 8. Run evaluation
# ------------------------------
all_results = {}
for method in CHUNK_METHODS:
    print(f"\nEvaluating {method}...")
    res = evaluate_method(method, questions_data, k=FINAL_EVAL_K)
    if res:
        all_results[method] = res

# ------------------------------
# 9. Generate CSVs
# ------------------------------
summary_rows = []
for method, tech_dict in all_results.items():
    for tech_name, data in tech_dict.items():
        avg = data["averages"]
        for metric, value in avg.items():
            summary_rows.append({
                "chunking_method": method,
                "retrieval_technique": tech_name,
                "metric": metric,
                "value": value
            })
df_summary = pd.DataFrame(summary_rows)
df_summary_pivot = df_summary.pivot_table(index=["chunking_method", "retrieval_technique"], 
                                          columns="metric", values="value").reset_index()
df_summary_pivot.to_csv("chunking_comparison_summary_fixed.csv", index=False)
print("\nSaved summary to chunking_comparison_summary_fixed.csv")

detail_rows = []
for method, tech_dict in all_results.items():
    for tech_name, data in tech_dict.items():
        for qrec in data["per_question"]:
            detail_rows.append({
                "chunking_method": method,
                "retrieval_technique": tech_name,
                "question_id": qrec["question_id"],
                "question": qrec["question"],
                "fact_coverage": qrec["fact_coverage"],
                "mrr": qrec["mrr"],
                "precision": qrec["precision"],
                "ndcg": qrec["ndcg"],
                "redundancy": qrec["redundancy"],
                "boundary_accuracy": qrec["boundary_accuracy"],
                "retrieved_chunks_sample": qrec["retrieved_chunks_sample"]
            })
df_details = pd.DataFrame(detail_rows)
df_details.to_csv("per_question_details_fixed.csv", index=False)
print("Saved per-question details to per_question_details_fixed.csv")

print("\n✅ Evaluation complete with fixed nDCG and standard top‑k ratios.")