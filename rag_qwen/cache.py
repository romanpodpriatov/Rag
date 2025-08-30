import os
import json
import hashlib
from pathlib import Path

class Cache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        # Use SHA256 hash of the key to create a filename
        hashed_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{hashed_key}.json"

    def get(self, key: str):
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def set(self, key: str, value):
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=4)

    def get_or_set(self, key: str, func, *args, **kwargs):
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        value = func(*args, **kwargs)
        self.set(key, value)
        return value
