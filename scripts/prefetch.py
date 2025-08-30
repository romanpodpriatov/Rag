import os
import sys
from pathlib import Path

# Add the parent directory to the sys.path to allow importing rag_qwen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qwen.config import Config
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

def prefetch_models(config_path: str = "configs/rag_qwen.yaml"):
    config = Config(config_path)

    local_files_only = config.models['local_files_only']
    trust_remote_code = config.models['trust_remote_code']

    if local_files_only:
        print("'local_files_only' is set to True in config. Skipping prefetching.")
        print("Ensure models are already available in your Hugging Face cache.")
        return

    print("Prefetching models...")

    # Embedder
    embed_model_name = config.models['embed']
    print(f"Prefetching embedder: {embed_model_name}")
    try:
        AutoTokenizer.from_pretrained(embed_model_name, trust_remote_code=trust_remote_code)
        AutoModel.from_pretrained(embed_model_name, trust_remote_code=trust_remote_code)
        print(f"Successfully prefetched {embed_model_name}")
    except Exception as e:
        print(f"Error prefetching {embed_model_name}: {e}")

    # Reranker
    reranker_model_name = config.models['reranker']
    print(f"Prefetching reranker: {reranker_model_name}")
    try:
        AutoTokenizer.from_pretrained(reranker_model_name, trust_remote_code=trust_remote_code)
        AutoModelForSequenceClassification.from_pretrained(reranker_model_name, trust_remote_code=trust_remote_code)
        print(f"Successfully prefetched {reranker_model_name}")
    except Exception as e:
        print(f"Error prefetching {reranker_model_name}: {e}")

    # Generator
    generator_model_name = config.models['generator']
    print(f"Prefetching generator: {generator_model_name}")
    try:
        AutoTokenizer.from_pretrained(generator_model_name, trust_remote_code=trust_remote_code)
        AutoModelForCausalLM.from_pretrained(generator_model_name, trust_remote_code=trust_remote_code)
        print(f"Successfully prefetched {generator_model_name}")
    except Exception as e:
        print(f"Error prefetching {generator_model_name}: {e}")

    print("Prefetching complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prefetch Qwen models for offline RAG.")
    parser.add_argument("--config", type=str, default="configs/rag_qwen.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    prefetch_models(args.config)