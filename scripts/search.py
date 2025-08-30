import os
import sys
from pathlib import Path
import json
import faiss
import numpy as np

# Add the parent directory to the sys.path to allow importing rag_qwen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qwen.config import Config
from rag_qwen.io import load_json
from rag_qwen.embeddings import Embeddings
from rag_qwen.bm25 import BM25Search
from rag_qwen.hybrid import reciprocal_rank_fusion
from rag_qwen.rerank import Reranker
from rag_qwen.answer import AnswerSynthesizer
from rag_qwen.cache import Cache
from rag_qwen.utils import set_offline_mode

class RAGSystem:
    def __init__(self, config_path: str = "configs/rag_qwen.yaml"):
        self.config = Config(config_path)
        set_offline_mode(self.config.offline)

        self.cache_dir_path = Path(self.config.cache_dir) # Convert to Path object
        self.cache = Cache(str(self.cache_dir_path)) # Cache expects string, so convert back for Cache init
        self.embedder = Embeddings(self.config)
        self.bm25_search = BM25Search()
        self.reranker = Reranker(self.config)
        self.answer_synthesizer = AnswerSynthesizer(self.config)

        self.faiss_index = None
        self.processed_chunks = []

        self._load_indexed_data()

    def _load_indexed_data(self):
        print("Loading indexed data...")
        processed_chunks_path = self.cache_dir_path / "processed_chunks.json"
        embeddings_path = self.cache_dir_path / "embeddings.json"

        if not processed_chunks_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Indexed data not found. Please run `python scripts/ingest.py` first. "
                f"Expected files: {processed_chunks_path}, {embeddings_path}"
            )

        self.processed_chunks = load_json(processed_chunks_path)
        all_embeddings = np.array(load_json(embeddings_path), dtype=np.float32)

        # Initialize FAISS index
        embedding_dim = all_embeddings.shape[1]
        metric_type = self.config.faiss['metric']
        if metric_type == "cosine":
            # For cosine similarity, normalize vectors and use IP (Inner Product)
            # Faiss IP search on L2 normalized vectors is equivalent to cosine similarity
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            faiss.normalize_L2(all_embeddings)
        elif metric_type == "L2":
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported FAISS metric: {metric_type}")

        if self.config.faiss['use_pca']:
            pca_dim = self.config.faiss['pca_dim']
            if pca_dim >= embedding_dim:
                print(f"Warning: PCA dimension ({pca_dim}) is >= embedding dimension ({embedding_dim}). Skipping PCA.")
                self.faiss_index = faiss.IndexFlatIP(embedding_dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(embedding_dim)
            else:
                print(f"Applying PCA to {pca_dim} dimensions...")
                mat = faiss.PCAMatrix(embedding_dim, pca_dim)
                mat.train(all_embeddings)
                all_embeddings = mat.apply_py(all_embeddings)
                self.faiss_index = faiss.IndexFlatIP(pca_dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(pca_dim)
        else:
            self.faiss_index = faiss.IndexFlatIP(embedding_dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(embedding_dim)

        self.faiss_index.add(all_embeddings)
        print(f"FAISS index loaded with {self.faiss_index.ntotal} vectors.")

        print("Indexing documents with BM25...")
        self.bm25_search.index_documents(self.processed_chunks)
        print("BM25 index ready.")

    def search(self, query: str, k: int = 10, print_citations: bool = False) -> str:
        # 1. Semantic Search (FAISS)
        query_embedding = self.embedder.embed_documents([query]).cpu().numpy()
        if self.config.faiss['use_pca'] and self.config.faiss['pca_dim'] < query_embedding.shape[1]:
            print("Warning: PCA was used during indexing. Query embedding should also be transformed. "
                  "This implementation assumes PCA is handled by the FAISS index type (e.g., IndexPQ). "
                  "For IndexFlat, PCA needs to be applied manually to the query.")

        D, I = self.faiss_index.search(query_embedding, self.config.rrf['k_sem'])
        semantic_results = []
        for i, score in zip(I[0], D[0]):
            if i != -1: # Ensure valid index
                chunk = self.processed_chunks[i]
                semantic_results.append({
                    "doc_id": chunk['doc_id'],
                    "chunk_index": chunk['chunk_index'],
                    "score": float(score), # Convert numpy float to Python float
                    "content": chunk['content'] # Include content for reranking/answer synthesis
                })

        # 2. Contextual BM25 Search
        bm25_results = self.bm25_search.search(query, top_k=self.config.rrf['k_bm25'])
        # Add content to bm25_results for reranking/answer synthesis
        bm25_results_with_content = []
        for res in bm25_results:
            # Find the corresponding chunk to get its content
            found_chunk = next((c for c in self.processed_chunks if c['doc_id'] == res['doc_id'] and c['chunk_index'] == res['chunk_index']), None)
            if found_chunk:
                res['content'] = found_chunk['content']
                bm25_results_with_content.append(res)

        # 3. RRF Merging
        # RRF expects a list of lists of items, where each item has doc_id and chunk_index
        # The score is not directly used in RRF, only the rank.
        fused_ranking_keys = reciprocal_rank_fusion(
            [semantic_results, bm25_results_with_content],
            k=self.config.rrf['pool'], # Using pool size as k for RRF
            top_n=self.config.rrf['pool']
        )
        
        # Retrieve the actual chunks based on fused_ranking_keys
        fused_chunks = []
        for doc_id, chunk_index in fused_ranking_keys:
            found_chunk = next((c for c in self.processed_chunks if c['doc_id'] == doc_id and c['chunk_index'] == chunk_index), None)
            if found_chunk:
                fused_chunks.append(found_chunk)

        # Handle case where fused_chunks might be empty
        if not fused_chunks:
            return "No relevant information found after hybrid retrieval."

        # 4. Cross-Encoder Re-rank
        passages_for_rerank = [chunk['content'] for chunk in fused_chunks]
        if passages_for_rerank:
            rerank_scores = self.reranker.rerank(query, passages_for_rerank)
            
            # Pair chunks with their rerank scores and sort
            reranked_chunks_with_scores = sorted(
                zip(fused_chunks, rerank_scores),
                key=lambda x: x[1], reverse=True
            )
            
            # Select top_k chunks after reranking
            top_k_chunks = [chunk for chunk, score in reranked_chunks_with_scores[:k]]
        else:
            top_k_chunks = []

        # 5. Answer Synthesis
        if not top_k_chunks:
            return "No relevant information found to answer the query after reranking."

        answer = self.answer_synthesizer.generate_answer(query, top_k_chunks)
        
        if print_citations:
            # Citations are already part of the answer string from AnswerSynthesizer
            return answer
        else:
            # Remove citations block if not requested
            if "CITATIONS:" in answer:
                return answer.split("CITATIONS:")[0].strip()
            return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search RAG system.")
    parser.add_argument("--config", type=str, default="configs/rag_qwen.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--query", type=str,
                        help="The query to search for. If not provided, enters interactive chat mode.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of top results to consider after reranking.")
    parser.add_argument("--print-citations", action="store_true",
                        help="Print citations along with the answer.")
    args = parser.parse_args()

    try:
        rag_system = RAGSystem(args.config)
        
        if args.query:
            # Single query mode
            answer = rag_system.search(args.query, args.k, args.print_citations)
            print(answer)
        else:
            # Interactive chat mode
            print("\nEntering chat mode. Type 'exit' or 'quit' to end the session.")
            while True:
                query = input("\nYour query: ")
                if query.lower() in ["exit", "quit"]:
                    print("Exiting chat mode.")
                    break
                
                if not query.strip():
                    print("Query cannot be empty. Please enter a query.")
                    continue

                answer = rag_system.search(query, args.k, args.print_citations)
                print(answer)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have ingested data by running `python scripts/ingest.py` first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure all models are prefetched and available locally, or check your configuration.")
