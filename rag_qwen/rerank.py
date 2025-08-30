from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rag_qwen.config import Config
import torch

class Reranker:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def _load_reranker_model(self):
        if self.tokenizer is None or self.model is None:
            model_name = self.config.models['reranker']
            local_files_only = self.config.models['local_files_only']
            trust_remote_code = self.config.models['trust_remote_code']

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
            )
            # Ensure padding token is set for the tokenizer object itself
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    # Fallback if no EOS token, though unlikely for Qwen
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
                    self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
            ).eval()

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self._load_reranker_model()
        
        # Prepare pairs for the cross-encoder
        pairs = [[query, passage] for passage in passages]
        
        # Tokenize in batches
        batch_size = self.config.rerank['batch_size']
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            inputs = self.tokenizer(
                batch_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                pad_to_multiple_of=8, # Optional: for performance on some hardware
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get scores from logits.squeeze(-1)
                batch_scores = outputs.logits.squeeze(-1).tolist()
                scores.extend(batch_scores)
                
        return scores

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Create a dummy config file for testing
    dummy_config_content = """
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
"""
    import os
    from pathlib import Path
    Path("configs").mkdir(exist_ok=True)
    with open("configs/rag_qwen.yaml", "w") as f:
        f.write(dummy_config_content)

    # Set offline environment variables for testing
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    config = Config()
    reranker = Reranker(config)

    query = "What is the capital of France?"
    passages = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is in Paris.",
        "Berlin is the capital of Germany."
    ]

    try:
        scores = reranker.rerank(query, passages)
        print(f"Rerank scores: {scores}")
        
        # Sort passages by scores
        ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        print("\nRanked Passages:")
        for passage, score in ranked_passages:
            print(f"Score: {score:.4f}, Passage: {passage}")

    except Exception as e:
        print(f"Error during reranking: {e}")
        print("Please ensure Qwen-Reranker models are prefetched or set local_files_only to False for online testing.")