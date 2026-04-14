# evaluate.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_db_faiss import FAISSVectorDB
from vector_db_qdrant import QdrantVectorDB
from vector_db_milvus import MilvusVectorDB

# Load golden dataset
# Format: [{"query": "text", "relevant_sentences": ["sentence1", ...]}]
with open("golden.json", "r") as f:
    golden = json.load(f)

# Pre‑compute query embeddings once
embedder = SentenceTransformer('all-MiniLM-L6-v2')
queries = [item["query"] for item in golden]
query_embs = embedder.encode(queries, show_progress_bar=True)

# Define metrics
def recall_at_k(retrieved_chunks, relevant_sentences, k=5):
    # retrieved_chunks: list of top‑k chunk texts
    # relevant_sentences: list of ground‑truth sentences
    # A chunk is relevant if it contains at least one relevant sentence
    retrieved_text = " ".join(retrieved_chunks[:k]).lower()
    for sent in relevant_sentences:
        if sent.lower() in retrieved_text:
            return 1.0
    return 0.0

def mrr(retrieved_chunks, relevant_sentences):
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        for sent in relevant_sentences:
            if sent.lower() in chunk["text"].lower():
                return 1.0 / rank
    return 0.0

# Main evaluation loop
chunk_methods = ["fixed_size", "sentence", "paragraph", "semantic"]
vector_dbs = {
    "FAISS": FAISSVectorDB,
    "Qdrant": QdrantVectorDB,
    "Milvus": MilvusVectorDB
}

results = {}

for method in chunk_methods:
    # Load chunks for this method
    with open(f"chunks_{method}.json", "r") as f:
        chunks = json.load(f)
    
    for db_name, DBClass in vector_dbs.items():
        print(f"\nEvaluating {method} with {db_name}")
        # Initialize DB
        if db_name == "FAISS":
            db = DBClass(dimension=384)
        else:
            db = DBClass()  # uses defaults localhost
        # Create index (each method gets a unique index name)
        index_name = f"{method}_{db_name}"
        db.create_index(chunks, index_name)
        
        # Evaluate each query
        recalls = []
        mrrs = []
        for i, q in enumerate(golden):
            q_emb = query_embs[i]
            retrieved = db.search(q_emb, top_k=10)  # get 10 for MRR@10
            # Use only top 5 for recall@5
            rec = recall_at_k([r["text"] for r in retrieved], q["relevant_sentences"], k=5)
            mrr_score = mrr(retrieved, q["relevant_sentences"])
            recalls.append(rec)
            mrrs.append(mrr_score)
        
        results[f"{method}_{db_name}"] = {
            "recall@5": np.mean(recalls),
            "mrr": np.mean(mrrs)
        }
        print(f"  recall@5: {results[f'{method}_{db_name}']['recall@5']:.3f}")
        print(f"  MRR: {results[f'{method}_{db_name}']['mrr']:.3f}")

# Save and display best combination
print("\n=== FINAL RESULTS ===")
for combo, metrics in results.items():
    print(f"{combo}: Recall@5={metrics['recall@5']:.3f}, MRR={metrics['mrr']:.3f}")

best = max(results, key=lambda x: results[x]['recall@5'])
print(f"\nBest combination: {best} with recall@5={results[best]['recall@5']:.3f}")