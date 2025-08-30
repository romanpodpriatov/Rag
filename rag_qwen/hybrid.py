def reciprocal_rank_fusion(ranked_lists: list[list[tuple]], k=60, top_n=150) -> list[tuple]:
    """
    Performs Reciprocal Rank Fusion (RRF) on a list of ranked lists.

    Args:
        ranked_lists: A list of ranked lists. Each inner list contains tuples of (doc_id, chunk_index, score).
                      The score is not directly used in RRF, but the rank is.
        k: A constant that dampens the influence of very high ranks.
        top_n: The number of top results to return after fusion.

    Returns:
        A list of tuples (doc_id, chunk_index) representing the fused ranking.
    """
    fused_scores = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            doc_id, chunk_index = item['doc_id'], item['chunk_index']
            # Ranks are 1-based, so add 1 to the 0-based index
            rrf_score = 1.0 / (k + rank + 1)
            
            key = (doc_id, chunk_index)
            if key not in fused_scores:
                fused_scores[key] = 0.0
            fused_scores[key] += rrf_score

    # Sort by fused score in descending order
    sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Return only the doc_id and chunk_index, up to top_n
    return [(item[0], item[1]) for (item, score) in sorted_results[:top_n]]

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example ranked lists from semantic search and BM25
    semantic_results = [
        {"doc_id": "docA", "chunk_index": 0, "score": 0.9},
        {"doc_id": "docB", "chunk_index": 1, "score": 0.8},
        {"doc_id": "docC", "chunk_index": 0, "score": 0.75},
        {"doc_id": "docA", "chunk_index": 1, "score": 0.7},
    ]

    bm25_results = [
        {"doc_id": "docB", "chunk_index": 1, "score": 10.5},
        {"doc_id": "docA", "chunk_index": 0, "score": 9.2},
        {"doc_id": "docD", "chunk_index": 0, "score": 8.1},
        {"doc_id": "docC", "chunk_index": 0, "score": 7.8},
    ]

    # RRF expects a list of lists of items, where each item has doc_id and chunk_index
    # The score is not directly used in RRF, only the rank.
    fused_ranking = reciprocal_rank_fusion([semantic_results, bm25_results], k=60, top_n=3)
    print("Fused Ranking:", fused_ranking)

    # Expected output (order might vary slightly based on tie-breaking, but top items should be consistent):
    # Fused Ranking: [('docA', 0), ('docB', 1), ('docC', 0)]