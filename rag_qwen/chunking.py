from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_qwen.prompts import SITUATE_PROMPT
from rag_qwen.config import Config
import torch

class Chunker:
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

    def generate_situating_context(self, full_document: str, chunk_text: str) -> str:
        self._load_generator_model()

        # Truncate document and chunk if they are too long for the prompt
        # A rough estimation: 4096 tokens context window, prompt takes some tokens
        # Let's say 3000 tokens for document and 500 for chunk
        max_doc_tokens = 3000
        max_chunk_tokens = 500

        full_document_truncated = self.tokenizer.decode(
            self.tokenizer.encode(full_document, max_length=max_doc_tokens, truncation=True)
        )
        chunk_text_truncated = self.tokenizer.decode(
            self.tokenizer.encode(chunk_text, max_length=max_chunk_tokens, truncation=True)
        )

        prompt = SITUATE_PROMPT.format(
            full_document_truncated=full_document_truncated,
            chunk_text_truncated=chunk_text_truncated
        )

        # Apply chat template if available, otherwise just encode
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{'role': 'user', 'content': prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=80, # As per requirement, 2-4 sentences, <=80 words
                temperature=self.config.generation['temperature'],
                top_p=self.config.generation['top_p'],
                do_sample=True, # Enable sampling for temperature/top_p
                pad_token_id=self.tokenizer.eos_token_id # Avoid warning
            )

        # Decode the generated text, skipping the input prompt part
        generated_text = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()

    def chunk_document(self, doc_id: str, document_content: str, metadata: dict = None) -> list:
        # This is a placeholder for actual chunking logic. 
        # For now, it will create a single chunk and generate situating context.
        # In a real scenario, this would split the document into smaller chunks
        # based on chunk_size_chars and chunk_overlap_chars.
        
        # For demonstration, let's assume one chunk for now.
        # The actual chunking logic needs to be implemented based on token limits.
        
        chunks = []
        # Simple chunking for now, assuming the document is small enough or we take the first part
        # In a real implementation, this would iterate and split the document
        
        chunk_text = document_content[:self.config.chunk_size_chars] # Simple char-based chunking
        
        # Generate situating context for this chunk
        situating_context = self.generate_situating_context(document_content, chunk_text)
        
        # Combine situating context with the original chunk text
        contextualized_content = f"Context: {situating_context}\n\n{chunk_text}"
        
        chunk_info = {
            "doc_id": doc_id,
            "chunk_index": 0, # Placeholder, needs proper indexing for multiple chunks
            "content": chunk_text,
            "contextualized_content": contextualized_content,
            "metadata": metadata if metadata else {}
        }
        chunks.append(chunk_info)
        
        return chunks


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
    chunker = Chunker(config)

    doc = """The quick brown fox jumps over the lazy dog. This is a test document to demonstrate chunking and situating context generation. It contains several sentences to simulate a real document. The fox is very quick and the dog is very lazy. This document is about animals and their characteristics. We are testing the Qwen model's ability to understand context and generate relevant summaries. The quick brown fox jumps over the lazy dog. This is a test document to demonstrate chunking and situating context generation. It contains several sentences to simulate a real document. The fox is very quick and the dog is very lazy. This document is about animals and their characteristics. We are testing the Qwen model's ability to understand context and generate relevant summaries.
    """
    
    # This part will fail if models are not prefetched and local_files_only is True
    # For actual testing, models need to be available locally.
    try:
        chunks = chunker.chunk_document("doc1", doc)
        for chunk in chunks:
            print(f"Doc ID: {chunk['doc_id']}, Chunk Index: {chunk['chunk_index']}")
            print(f"Content: {chunk['content']}")
            print(f"Contextualized Content: {chunk['contextualized_content']}")
            print("---")
    except Exception as e:
        print(f"Error during chunking: {e}")
        print("Please ensure Qwen models are prefetched or set local_files_only to False for online testing.")