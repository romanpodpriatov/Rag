import os
import sys
from pathlib import Path
import json
from tqdm import tqdm

# Add the parent directory to the sys.path to allow importing rag_qwen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qwen.config import Config
from rag_qwen.io import load_json, save_json
from rag_qwen.chunking import Chunker
from rag_qwen.embeddings import Embeddings
from rag_qwen.bm25 import BM25Search
from rag_qwen.cache import Cache
from rag_qwen.utils import set_offline_mode

def ingest_data(config_path: str = "configs/rag_qwen.yaml", data_path: str = "data/codebase_chunks.json"):
    config = Config(config_path)
    set_offline_mode(config.offline)

    cache_dir_path = Path(config.cache_dir) # Convert to Path object
    cache = Cache(str(cache_dir_path)) # Cache expects string, so convert back for Cache init
    chunker = Chunker(config)
    embedder = Embeddings(config)
    bm25_search = BM25Search()

    print(f"Loading data from {data_path}...")
    documents = load_json(data_path)
    print(f"Loaded {len(documents)} documents.")

    processed_chunks = []
    all_embeddings = []

    print("Processing documents and generating contextual embeddings...")
    for doc_data in tqdm(documents, desc="Processing documents"):
        doc_id = doc_data["doc_id"]
        full_document_content = doc_data["content"] # This is the full document text

        for chunk in doc_data["chunks"]:
            chunk_text = chunk["content"]
            chunk_index = chunk["original_index"] # Assuming original_index is the chunk_index

            # Cache key for situating context
            sit_ctx_key = f"sit_ctx_{doc_id}_{chunk_index}"
            
            # Get or generate situating context
            situating_context = cache.get_or_set(
                sit_ctx_key,
                lambda: chunker.generate_situating_context(full_document_content, chunk_text)
            )
            
            chunk["situating_context"] = situating_context
            chunk["contextualized_content"] = situating_context + "\n\n" + chunk_text

            # Cache key for embedding
            embedding_key = f"embedding_{doc_id}_{chunk_index}"
            
            # Get or generate embedding
            embedding = cache.get_or_set(
                embedding_key,
                lambda: embedder.embed_documents([chunk['contextualized_content']]).tolist()[0] # Convert to list for JSON serialization
            )
            all_embeddings.append(embedding)
            processed_chunks.append(chunk)

    print("Indexing documents with BM25...")
    bm25_search.index_documents(processed_chunks)
    # BM25 index is built in memory, so no need to save it explicitly for now
    # If persistence is needed, it would require serialization (e.g., pickle)

    # Save processed chunks and embeddings (for FAISS indexing later)
    # This is a simplified approach. In a real system, FAISS index would be saved separately.
    save_json(processed_chunks, cache_dir_path / "processed_chunks.json")
    save_json(all_embeddings, cache_dir_path / "embeddings.json")
    
    print("Data ingestion complete. Processed chunks and embeddings saved to cache directory.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest data for RAG system.")
    parser.add_argument("--config", type=str, default="configs/rag_qwen.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--data", type=str, default="data/codebase_chunks.json",
                        help="Path to the codebase chunks JSON file.")
    args = parser.parse_args()

    ingest_data(args.config, args.data)