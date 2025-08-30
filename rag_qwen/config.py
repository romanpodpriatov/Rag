import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="configs/rag_qwen.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def models(self):
        return self.get('models')

    @property
    def context_window_tokens(self):
        return self.get('context_window_tokens')

    @property
    def chunk_size_chars(self):
        return self.get('chunk_size_chars')

    @property
    def chunk_overlap_chars(self):
        return self.get('chunk_overlap_chars')

    @property
    def rrf(self):
        return self.get('rrf')

    @property
    def rerank(self):
        return self.get('rerank')

    @property
    def faiss(self):
        return self.get('faiss')

    @property
    def quantization(self):
        return self.get('quantization')

    @property
    def generation(self):
        return self.get('generation')

    @property
    def offline(self):
        return self.get('offline')

    @property
    def cache_dir(self):
        return self.get('cache_dir')

# Example usage (for testing purposes, remove in final version if not needed)
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
    Path("configs").mkdir(exist_ok=True)
    with open("configs/rag_qwen.yaml", "w") as f:
        f.write(dummy_config_content)

    cfg = Config()
    print(f"Embed model: {cfg.models['embed']}")
    print(f"Cache dir: {cfg.cache_dir}")
    print(f"RRF w_sem: {cfg.rrf['w_sem']}")
    print(f"Offline HF_HUB_OFFLINE: {cfg.offline['hf_hub_offline']}")