# evaluate_with_golden_table.py
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_db_faiss import FAISSVectorDB
from vector_db_qdrant import QdrantVectorDB
from vector_db_milvus import MilvusVectorDB

# ---------------------------
# 1. Load golden dataset from CSV
# ---------------------------
def load_golden(csv_path):
    df = pd.read_csv(csv_path)
    golden = []
    for _, row in df.iterrows():
        # Build list of relevant text snippets
        relevant_snippets = []
        # Add the answer
        if pd.notna(row['Answer']):
            relevant_snippets.append(row['Answer'].strip())
        # Add each fact (split by semicolon)
        if pd.notna(row['Facts']):
            facts = [f.strip() for f in row['Facts'].split(';')]
            relevant_snippets.extend(facts)
        # Optional: add keywords as well (if you want higher recall)
        # if pd.notna(row['keywords']):
        #     keywords = [k.strip() for k in row['keywords'].split(',')]
        #     relevant_snippets.extend(keywords)
        
        golden.append({
            'query': row['Question'],
            'relevant_snippets': relevant_snippets,
            'expected_section': row['document Section'] if pd.notna(row['document Section']) else None,
            'answer': row['Answer']
        })
    return golden

# ---------------------------
# 2. Relevance check
# ---------------------------
def is_chunk_relevant(chunk_text, relevant_snippets):
    """Return True if chunk contains any of the relevant snippets (case‑insensitive)."""
    chunk_lower = chunk_text.lower()
    for snippet in relevant_snippets:
        if snippet.lower() in chunk_lower:
            return True
    return False

# ---------------------------
# 3. Metrics
# ---------------------------
def recall_at_k(retrieved_chunks, relevant_snippets, k=5):
    """Binary recall@k: 1 if at least one relevant chunk in top‑k, else 0."""
    for chunk in retrieved_chunks[:k]:
        if is_chunk_relevant(chunk['text'], relevant_snippets):
            return 1.0
    return 0.0

def mrr(retrieved_chunks, relevant_snippets):
    """Mean Reciprocal Rank of the first relevant chunk."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_chunk_relevant(chunk['text'], relevant_snippets):
            return 1.0 / rank
    return 0.0

def hit_rate(retrieved_chunks, relevant_snippets, k=5):
    """Alias for recall@k (same thing)."""
    return recall_at_k(retrieved_chunks, relevant_snippets, k)

# ---------------------------
# 4. Main evaluation loop
# ---------------------------
def main():
    # Configuration
    chunk_methods = ["fixed_size", "sentence", "paragraph", "semantic"]
    vector_dbs = {
        "FAISS": FAISSVectorDB,
        "Qdrant": QdrantVectorDB,
        "Milvus": MilvusVectorDB
    }
    
    # Load golden data
    golden = load_golden("golden.csv")
    print(f"Loaded {len(golden)} queries from golden dataset.")
    
    # Load embedding model once
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_texts = [g['query'] for g in golden]
    query_embeddings = embedder.encode(query_texts, show_progress_bar=True)
    
    results = {}
    
    for method in chunk_methods:
        # Load chunks produced earlier for this method
        with open(f"chunks_{method}.json", "r") as f:
            chunks = json.load(f)
        
        for db_name, DBClass in vector_dbs.items():
            print(f"\n=== Testing {method} with {db_name} ===")
            # Initialize DB
            if db_name == "FAISS":
                db = DBClass(dimension=384)
            else:
                db = DBClass()  # assumes localhost with default ports
            
            # Create a separate index for this (method, db) pair
            index_name = f"{method}_{db_name}"
            db.create_index(chunks, index_name)
            
            recalls = []
            mrrs = []
            for i, query_data in enumerate(golden):
                q_emb = query_embeddings[i]
                retrieved = db.search(q_emb, top_k=10)  # get top 10 for MRR
                rel_snippets = query_data['relevant_snippets']
                # Skip if no relevant snippets (should not happen)
                if not rel_snippets:
                    continue
                rec = recall_at_k(retrieved, rel_snippets, k=5)
                mrr_val = mrr(retrieved, rel_snippets)
                recalls.append(rec)
                mrrs.append(mrr_val)
            
            avg_recall = np.mean(recalls)
            avg_mrr = np.mean(mrrs)
            results[f"{method}_{db_name}"] = {
                "recall@5": avg_recall,
                "mrr": avg_mrr
            }
            print(f"  Recall@5: {avg_recall:.3f}")
            print(f"  MRR: {avg_mrr:.3f}")
    
    # Display final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS (averaged over all queries)")
    print("="*60)
    for combo, metrics in results.items():
        print(f"{combo:30} | Recall@5: {metrics['recall@5']:.3f} | MRR: {metrics['mrr']:.3f}")
    
    best = max(results, key=lambda x: results[x]['recall@5'])
    print(f"\n🏆 Best combination: {best} with Recall@5 = {results[best]['recall@5']:.3f}")

if __name__ == "__main__":
    main()