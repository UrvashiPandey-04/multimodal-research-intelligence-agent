"""
LLM Client module using Google Gemini API (gemini-2.5-flash) via google-genai SDK.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.config import get_gemini_api_key

load_dotenv()


class MissingAPIKeyError(Exception):
    """Raised when GEMINI_API_KEY environment variable is not configured."""
    pass


class GeminiAPIError(Exception):
    """Raised when Google Gemini API call fails."""
    pass


class GeminiLLM:
    """Wrapper for Google Gemini API (gemini-2.5-flash) generation calls."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Gemini LLM client.

        Args:
            api_key: Gemini API Key. If None, resolves from GEMINI_API_KEY env var or st.secrets.
            model_name: Gemini model ID (default: gemini-2.5-flash).
        """
        key = api_key or get_gemini_api_key()
        self.api_key = key
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if key:
            self.client = genai.Client(api_key=key)
        else:
            self.client = None


    def is_configured(self) -> bool:
        """Check if Gemini API key is configured and client is initialized."""
        return self.client is not None and len(self.api_key or "") > 0

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate grounded text using Gemini 2.5 Flash.

        Args:
            prompt: User prompt string with context.
            system_instruction: System prompt guiding model behavior.
            temperature: Sampling temperature (default: 0.2 for factual deterministic outputs).

        Returns:
            Generated text string response.

        Raises:
            MissingAPIKeyError: If API key is missing.
            GeminiAPIError: If API call fails.
        """
        if not self.is_configured():
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file."
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise GeminiAPIError(f"Gemini API call failed: {e}") from e

