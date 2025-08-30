from rank_bm25 import BM25Okapi
import numpy as np

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []

    def index_documents(self, documents: list[dict]):
        # Documents are expected to have 'content' and 'contextualized_content' fields
        # We will create a combined corpus for BM25
        self.corpus = []
        self.doc_ids = []
        for doc in documents:
            # Combine content and contextualized_content for BM25 indexing
            combined_text = f"{doc['content']} {doc['contextualized_content']}"
            self.corpus.append(combined_text.split())
            self.doc_ids.append((doc['doc_id'], doc['chunk_index']))
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> list[tuple]:
        if not self.bm25:
            return []

        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices based on scores
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.doc_ids[idx][0],
                "chunk_index": self.doc_ids[idx][1],
                "score": doc_scores[idx]
            })
        return results

# Example usage (for testing purposes)
if __name__ == "__main__":
    bm25_search = BM25Search()

    documents = [
        {"doc_id": "doc1", "chunk_index": 0, "content": "The quick brown fox", "contextualized_content": "This chunk introduces a quick fox."},
        {"doc_id": "doc1", "chunk_index": 1, "content": "jumps over the lazy dog", "contextualized_content": "This chunk describes the fox's action."},
        {"doc_id": "doc2", "chunk_index": 0, "content": "Hello world example", "contextualized_content": "A basic programming example."},
    ]

    bm25_search.index_documents(documents)

    query = "quick fox"
    results = bm25_search.search(query, top_k=2)
    print(f"BM25 search results for '{query}':")
    for res in results:
        print(res)

    query = "programming example"
    results = bm25_search.search(query, top_k=1)
    print(f"\nBM25 search results for '{query}':")
    for res in results:
        print(res)
