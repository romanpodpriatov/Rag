from transformers import AutoTokenizer, AutoModel
from rag_qwen.config import Config
import torch
import torch.nn.functional as F

class Embeddings:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def _load_embedding_model(self):
        if self.tokenizer is None or self.model is None:
            model_name = self.config.models['embed']
            local_files_only = self.config.models['local_files_only']
            trust_remote_code = self.config.models['trust_remote_code']

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
            ).eval()

    def embed_documents(self, texts: list[str]) -> torch.Tensor:
        self._load_embedding_model()
        
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform mean pooling across the token embeddings
        # Take the last hidden state and attention mask
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        
        # Expand attention_mask to match token_embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum the embeddings weighted by the attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the attention mask to get the number of tokens (for averaging)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
        
        # Mean pooling
        mean_embeddings = sum_embeddings / sum_mask
        
        # L2 normalization
        embeddings = F.normalize(mean_embeddings, p=2, dim=1)
        
        return embeddings


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
    embedder = Embeddings(config)

    texts_to_embed = [
        "This is a test sentence.",
        "Another sentence for embedding."
    ]

    try:
        embeddings = embedder.embed_documents(texts_to_embed)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"First embedding (first 5 elements): {embeddings[0][:5]}")
    except Exception as e:
        print(f"Error during embedding: {e}")
        print("Please ensure Qwen-Embedding models are prefetched or set local_files_only to False for online testing.")
