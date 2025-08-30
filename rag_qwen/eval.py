def calculate_pass_at_k(retrieved_docs: list[list[tuple]], relevant_docs: list[list[tuple]], k: int) -> float:
    """
    Calculates Pass@K metric.

    Args:
        retrieved_docs: A list of lists, where each inner list contains tuples of (doc_id, chunk_index)
                        representing the retrieved documents for a query.
        relevant_docs: A list of lists, where each inner list contains tuples of (doc_id, chunk_index)
                       representing the relevant documents for each query.
        k: The K for Pass@K.

    Returns:
        The Pass@K score.
    """
    if not retrieved_docs or not relevant_docs or len(retrieved_docs) != len(relevant_docs):
        raise ValueError("retrieved_docs and relevant_docs must be non-empty and have the same length.")

    num_queries = len(retrieved_docs)
    passed_queries = 0

    for i in range(num_queries):
        retrieved_k = set(retrieved_docs[i][:k])
        relevant_set = set(relevant_docs[i])

        # Check if any of the relevant documents are in the top K retrieved documents
        if any(doc in retrieved_k for doc in relevant_set):
            passed_queries += 1

    return passed_queries / num_queries

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example 1: Perfect match
    retrieved1 = [[("doc1", 0), ("doc2", 1)], [("doc3", 0)]]
    relevant1 = [[("doc1", 0)], [("doc3", 0)]]
    pass_at_1_1 = calculate_pass_at_k(retrieved1, relevant1, k=1)
    print(f"Pass@1 (Example 1): {pass_at_1_1}") # Expected: 1.0

    # Example 2: Partial match
    retrieved2 = [[("doc1", 0), ("doc2", 1)], [("doc3", 0)]]
    relevant2 = [[("doc4", 0)], [("doc3", 0)]]
    pass_at_1_2 = calculate_pass_at_k(retrieved2, relevant2, k=1)
    print(f"Pass@1 (Example 2): {pass_at_1_2}") # Expected: 0.5

    # Example 3: No match within K
    retrieved3 = [[("doc1", 0), ("doc2", 1)], [("doc3", 0)]]
    relevant3 = [[("doc1", 1)], [("doc4", 0)]]
    pass_at_1_3 = calculate_pass_at_k(retrieved3, relevant3, k=1)
    print(f"Pass@1 (Example 3): {pass_at_1_3}") # Expected: 0.0

    # Example 4: K larger than retrieved
    retrieved4 = [[("doc1", 0)], [("doc3", 0)]]
    relevant4 = [[("doc1", 0)], [("doc3", 0)]]
    pass_at_5_4 = calculate_pass_at_k(retrieved4, relevant4, k=5)
    print(f"Pass@5 (Example 4): {pass_at_5_4}") # Expected: 1.0