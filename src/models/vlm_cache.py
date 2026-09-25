"""
Disk-backed JSON Cache for Gemini 2.5 Flash VLM figure analysis outputs.
"""

import os
import json
from typing import Dict, Any, Optional


class VLMCache:
    """Cache manager preventing duplicate VLM API calls for identical extracted figures."""

    def __init__(self, cache_file: str = "data/cache/vlm_cache.json"):
        """
        Initialize VLM cache manager.

        Args:
            cache_file: Path to JSON file persisting cached VLM analysis output.
        """
        self.cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache entries from JSON file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def save(self) -> None:
        """Persist current cache entries to JSON file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save VLM cache: {e}")

    def make_key(self, doc_id: str, figure_id: str, prompt_version: str = "v1") -> str:
        """Construct deterministic cache key."""
        return f"{doc_id}_{figure_id}_{prompt_version}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached VLM analysis dictionary if present."""
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store VLM analysis dictionary in cache and persist to disk."""
        self._cache[key] = value
        self.save()

    def clear(self) -> None:
        """Clear cache entries."""
        self._cache.clear()
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                pass
