from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_qwen.config import Config
from rag_qwen.prompts import ANSWER_PROMPT_WITH_CITATIONS
import torch

class AnswerSynthesizer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def _load_generator_model(self):
        if self.tokenizer is None or self.model is None:
            model_name = self.config.models['generator']
            local_files_only = self.config.models['local_files_only']
            trust_remote_code = self.config.models['trust_remote_code']

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
            ).eval()

    def generate_answer(self, query: str, top_chunks: list[dict]) -> str:
        self._load_generator_model()

        enumerated_top_chunks_with_doc_id_and_index = ""
        citations = []
        for i, chunk in enumerate(top_chunks):
            enumerated_top_chunks_with_doc_id_and_index += f"Snippet {i+1}: {chunk['content']}\n"
            citations.append(f"{chunk['doc_id']}:{chunk['chunk_index']}")

        prompt = ANSWER_PROMPT_WITH_CITATIONS.format(
            query=query,
            enumerated_top_chunks_with_doc_id_and_index=enumerated_top_chunks_with_doc_id_and_index
        )

        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{'role': 'user', 'content': prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        generation_config = self.config.generation

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=generation_config['max_new_tokens'],
                temperature=generation_config['temperature'],
                top_p=generation_config['top_p'],
                do_sample=True, # Enable sampling for temperature/top_p
                pad_token_id=self.tokenizer.eos_token_id # Avoid warning
            )

        generated_text = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Always append citations block programmatically
        citations_block = f"\nCITATIONS: {','.join(citations)}"
        return generated_text.strip() + citations_block

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
    synthesizer = AnswerSynthesizer(config)

    query = "What is the capital of France?"
    top_chunks = [
        {"doc_id": "doc1", "chunk_index": 0, "content": "Paris is the capital and most populous city of France."},
        {"doc_id": "doc2", "chunk_index": 1, "content": "The Eiffel Tower is in Paris."},
        {"doc_id": "doc3", "chunk_index": 0, "content": "Berlin is the capital of Germany."}
    ]

    try:
        answer = synthesizer.generate_answer(query, top_chunks)
        print(f"Generated Answer: {answer}")
    except Exception as e:
        print(f"Error during answer generation: {e}")
        print("Please ensure Qwen models are prefetched or set local_files_only to False for online testing.")
