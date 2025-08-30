# RAG with Local Qwen Models

This project implements an advanced Retrieval-Augmented Generation (RAG) system using exclusively local Qwen models, designed for offline operation.

## mermaid 

[Корпус] --> [Чанкинг] --> [Situating context (Qwen3-0.6B)]
                                |--> [КЭШ контекстов]
                └─(склейка original+context)─┐
                                            v
                                 [Эмбеддинг (Qwen3-Embedding-0.6B)]
                                            |--> [КЭШ эмбеддингов]
                                            v
                                      (FAISS INDEX)
                        └───────────────┐
[склейка original+context] --tokenize--> (BM25 INDEX по двум полям)

====================  ОНЛАЙН-ЗАПРОС  ====================

[User Query] -> [нормализовать]
         |--> [FAISS Top N_sem]
         |--> [BM25  Top N_bm25]
                 \            /
                  \          /
                   [RRF: fusion (w_sem,w_bm25) → pool N]
                               |
                               v
                [Re-rank (Qwen3-Reranker-0.6B, batch)]
                               |--> [КЭШ реранк-скоров]
                               v
                         [Top-K чанков]
                               |
                               v
     [Qwen3-0.6B: генерация ответа + строгие CITATIONS]
                               |
                               v
                      [Ответ пользователю]

(Оценка: scripts/evaluate.py считает Pass@5/10/20)

## Features

- **Offline Mode**: Supports `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `local_files_only=True` for model loading.
- **Local Models**: Utilizes Qwen/Qwen3-Embedding-0.6B for embeddings, Qwen/Qwen3-Reranker-0.6B for reranking, and Qwen/Qwen3-0.6B for answer generation.
- **Contextual Embeddings**: Generates "situating context" for each chunk using Qwen3-0.6B.
- **Hybrid Retrieval**: Combines FAISS (semantic search) and BM25 (keyword search) using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-sorts retrieved documents using Qwen3-Reranker-0.6B.
- **Answer Synthesis**: Generates answers with citations from Qwen3-0.6B.
- **Evaluation**: Supports Pass@K metric calculation.
- **Caching**: Caches situating contexts, embeddings, and rerank scores for performance.

## Project Structure

```
rag_qwen/
  __init__.py
  config.py
  io.py
  chunking.py
  prompts.py
  embeddings.py
  bm25.py
  hybrid.py
  rerank.py
  answer.py
  eval.py
  cache.py
  utils.py
scripts/
  ingest.py
  search.py
  evaluate.py
  prefetch.py
configs/
  rag_qwen.yaml
README.md
requirements.txt
```

## Setup

1.  **Clone the repository** (if not already done).
2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prefetch Models (Optional but Recommended for Offline Use)**:

    If `local_files_only` is set to `false` in `configs/rag_qwen.yaml`, you can prefetch the models:

    ```bash
    python scripts/prefetch.py --config configs/rag_qwen.yaml
    ```

    If `local_files_only` is `true`, ensure the models are already in your Hugging Face cache.

## Usage

### 1. Ingest Data

Process your codebase chunks and generate embeddings. This will create necessary index files and caches.

```bash
python scripts/ingest.py --config configs/rag_qwen.yaml --data data/codebase_chunks.json
```

### 2. Search

Query the RAG system to get answers with optional citations. `scripts/search.py` now supports both single-query mode and an interactive chat mode.

**Single-Query Mode**

To run a single query and get an immediate answer:

```bash
python scripts/search.py --config configs/rag_qwen.yaml --query "Your query here" --k 10 --print-citations
```

**Interactive Chat Mode**

To enter an interactive chat session with the RAG system, simply run `scripts/search.py` without the `--query` argument:

```bash
python scripts/search.py --config configs/rag_qwen.yaml
```

Once in chat mode, you will be prompted to enter your queries. Type `exit` or `quit` to end the session.

Example interactive session:

```
$ python scripts/search.py --config configs/rag_qwen.yaml
Entering chat mode. Type 'exit' or 'quit' to end the session.

Your query: What is RAG?
Answer: RAG stands for Retrieval-Augmented Generation. It's a technique that enhances the capabilities of large language models by retrieving relevant information from an external knowledge base before generating a response. This helps to provide more accurate, up-to-date, and contextually relevant answers.

Your query: How does it work?
Answer: RAG typically involves two main phases: retrieval and generation. In the retrieval phase, given a user query, the system searches a knowledge base (like a collection of documents or a database) to find relevant pieces of information. In the generation phase, these retrieved documents are then fed along with the original query to a large language model, which uses this combined context to formulate a comprehensive and informed answer.

Your query: exit
Exiting chat mode.
```

### 3. Evaluate

Evaluate the RAG system's performance using Pass@K metric.

```bash
python scripts/evaluate.py --config configs/rag_qwen.yaml --k 5 10 20
```

## Configuration

Adjust the RAG parameters in `configs/rag_qwen.yaml`.

```yaml
models:
  embed: "Qwen/Qwen3-Embedding-0.6B"
  reranker: "Qwen/Qwen3-Reranker-0.6B"
  generator: "Qwen/Qwen3-0.6B"
  local_files_only: true
  trust_remote_code: true
context_window_tokens: 4096
chunk_size_chars: 1200
chunk_overlap_chars: 200
rrf: { w_sem: 0.8, w_bm25: 0.2, k_sem: 60, k_bm25: 60, pool: 150 }
rerank: { batch_size: 8, top_k: 10 }
faiss: { use_pca: true, pca_dim: 384, metric: "cosine" }
quantization: { bnb_4bit: true }
generation: { temperature: 0.2, top_p: 0.9, max_new_tokens: 512 }
offline: { hf_hub_offline: true, transformers_offline: true }
cache_dir: "./cache"
```
