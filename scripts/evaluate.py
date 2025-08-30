import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the sys.path to allow importing rag_qwen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qwen.config import Config
from rag_qwen.io import load_jsonl
from rag_qwen.eval import calculate_pass_at_k
from scripts.search import RAGSystem # Import RAGSystem from search.py
from rag_qwen.hybrid import reciprocal_rank_fusion # Import reciprocal_rank_fusion

def evaluate_rag(config_path: str = "configs/rag_qwen.yaml", eval_data_path: str = "data/evaluation_set.jsonl", k_values: list[int] = [5, 10, 20]):
    config = Config(config_path)
    rag_system = RAGSystem(config_path) # Initialize the RAG system

    print(f"Loading evaluation data from {eval_data_path}...")
    evaluation_set = load_jsonl(eval_data_path)
    print(f"Loaded {len(evaluation_set)} evaluation samples.")

    all_retrieved_docs = []
    all_relevant_docs = []

    print("Running evaluation queries...")
    for sample in tqdm(evaluation_set, desc="Evaluating queries"):
        query = sample['query']
        relevant_docs_for_query = [(doc['doc_id'], doc['chunk_index']) for doc in sample['relevant_chunks']]
        
        # --- TEMPORARY WORKAROUND START ---
        # This part needs to be aligned with the actual RAGSystem.search return value.
        # For now, I will call the internal components of RAGSystem to get retrieved chunks.
        
        # 1. Semantic Search (FAISS)
        query_embedding = rag_system.embedder.embed_documents([query]).cpu().numpy()
        D, I = rag_system.faiss_index.search(query_embedding, rag_system.config.rrf['k_sem'])
        semantic_results = []
        for i, score in zip(I[0], D[0]):
            if i != -1:
                chunk = rag_system.processed_chunks[i]
                semantic_results.append({
                    "doc_id": chunk['doc_id'],
                    "chunk_index": chunk['chunk_index'],
                    "score": float(score),
                    "content": chunk['content']
                })

        # 2. Contextual BM25 Search
        bm25_results = rag_system.bm25_search.search(query, top_k=rag_system.config.rrf['k_bm25'])
        bm25_results_with_content = []
        for res in bm25_results:
            found_chunk = next((c for c in rag_system.processed_chunks if c['doc_id'] == res['doc_id'] and c['chunk_index'] == res['chunk_index']), None)
            if found_chunk:
                res['content'] = found_chunk['content']
                bm25_results_with_content.append(res)

        # 3. RRF Merging
        fused_ranking_keys = reciprocal_rank_fusion(
            [semantic_results, bm25_results_with_content],
            k=rag_system.config.rrf['pool'],
            top_n=rag_system.config.rrf['pool']
        )
        
        fused_chunks = []
        for doc_id, chunk_index in fused_ranking_keys:
            found_chunk = next((c for c in rag_system.processed_chunks if c['doc_id'] == doc_id and c['chunk_index'] == chunk_index), None)
            if found_chunk:
                fused_chunks.append(found_chunk)

        # 4. Cross-Encoder Re-rank
        passages_for_rerank = [chunk['content'] for chunk in fused_chunks]
        if passages_for_rerank:
            rerank_scores = rag_system.reranker.rerank(query, passages_for_rerank)
            reranked_chunks_with_scores = sorted(
                zip(fused_chunks, rerank_scores),
                key=lambda x: x[1], reverse=True
            )
            retrieved_chunks_for_query = [(chunk['doc_id'], chunk['chunk_index']) for chunk, score in reranked_chunks_with_scores]
        else:
            retrieved_chunks_for_query = []

        # --- TEMPORARY WORKAROUND END ---

        all_retrieved_docs.append(retrieved_chunks_for_query)
        all_relevant_docs.append(relevant_docs_for_query)

    print("Calculating Pass@K scores...")
    for k in k_values:
        pass_at_k = calculate_pass_at_k(all_retrieved_docs, all_relevant_docs, k)
        print(f"Pass@{k}: {pass_at_k:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system.")
    parser.add_argument("--config", type=str, default="configs/rag_qwen.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--k", type=int, nargs='+', default=[5, 10, 20],
                        help="List of K values for Pass@K calculation (e.g., 5 10 20).")
    args = parser.parse_args()

    try:
        evaluate_rag(args.config, k_values=args.k)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have ingested data by running `python scripts/ingest.py` first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure all models are prefetched and available locally, or check your configuration.")
