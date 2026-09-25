"""
Text Normalizer module for cleaning PDF extracted text conservatively.
"""

import re


class TextNormalizer:
    """Conservative text normalizer preserving scientific notation, equations, numbers, and citations."""

    def __init__(self):
        # Regex patterns for normalization
        self._crlf_pattern = re.compile(r"\r\n|\r")
        self._multi_space_pattern = re.compile(r"[ \t]+")
        self._multi_newline_pattern = re.compile(r"\n{3,}")
        self._trailing_line_spaces = re.compile(r"[ \t]+\n")
        self._leading_line_spaces = re.compile(r"\n[ \t]+")

    def normalize(self, text: str) -> str:
        """
        Normalize text conservatively without modifying mathematical symbols, numbers, or citations.

        Args:
            text: Raw input text.

        Returns:
            Normalized clean string.
        """
        if not text:
            return ""

        # 1. Standardize CRLF to LF
        cleaned = self._crlf_pattern.sub("\n", text)

        # 2. Remove space before/after linebreaks
        cleaned = self._trailing_line_spaces.sub("\n", cleaned)
        cleaned = self._leading_line_spaces.sub("\n", cleaned)

        # 3. Collapse multiple horizontal spaces/tabs into a single space
        cleaned = self._multi_space_pattern.sub(" ", cleaned)

        # 4. Collapse 3+ consecutive newlines into double newlines (paragraph preservation)
        cleaned = self._multi_newline_pattern.sub("\n\n", cleaned)

        # 5. Strip leading and trailing whitespace
        return cleaned.strip()


def normalize_text(text: str) -> str:
    """Helper function to normalize text conservatively."""
    normalizer = TextNormalizer()
    return normalizer.normalize(text)
