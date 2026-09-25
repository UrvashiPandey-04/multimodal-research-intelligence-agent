"""
Text Quality Assessor module for validating extracted PDF page text usability.
"""

import string
from typing import List
from src.ingestion.schema import QualityMetrics


class TextQualityAssessor:
    """Deterministic text quality evaluator for extracted page text."""

    def __init__(
        self,
        min_words: int = 15,
        min_printable_ratio: float = 0.85,
        min_avg_word_length: float = 1.5,
        max_avg_word_length: float = 25.0,
    ):
        """
        Initialize quality thresholds.

        Args:
            min_words: Minimum word count threshold.
            min_printable_ratio: Minimum ratio of printable characters (including whitespace/newlines).
            min_avg_word_length: Minimum average word length (detects garbage char streams).
            max_avg_word_length: Maximum average word length (detects unbroken binary strings).
        """
        self.min_words = min_words
        self.min_printable_ratio = min_printable_ratio
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length

    def evaluate(self, text: str) -> QualityMetrics:
        """
        Evaluate quality metrics for a given text block.

        Args:
            text: Extracted raw text string.

        Returns:
            QualityMetrics object detailing char/word counts, printable ratio, and usability flag.
        """
        if not text:
            return QualityMetrics(
                char_count=0,
                word_count=0,
                printable_ratio=0.0,
                avg_word_length=0.0,
                is_usable=False,
                failure_reasons=["empty_text"],
            )

        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # Count printable characters including standard whitespace (\n, \t, \r, ' ')
        allowed_whitespace = {'\n', '\r', '\t', ' '}
        printable_chars = sum(1 for c in text if c in string.printable or c in allowed_whitespace)
        printable_ratio = printable_chars / char_count if char_count > 0 else 0.0

        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

        failure_reasons: List[str] = []

        if word_count < self.min_words:
            failure_reasons.append(f"insufficient_word_count ({word_count} < {self.min_words})")

        if printable_ratio < self.min_printable_ratio:
            failure_reasons.append(f"low_printable_ratio ({printable_ratio:.2f} < {self.min_printable_ratio})")

        if word_count > 0:
            if avg_word_length < self.min_avg_word_length:
                failure_reasons.append(f"suspiciously_short_words (avg: {avg_word_length:.2f})")
            elif avg_word_length > self.max_avg_word_length:
                failure_reasons.append(f"suspiciously_long_words (avg: {avg_word_length:.2f})")

        is_usable = len(failure_reasons) == 0

        return QualityMetrics(
            char_count=char_count,
            word_count=word_count,
            printable_ratio=printable_ratio,
            avg_word_length=avg_word_length,
            is_usable=is_usable,
            failure_reasons=failure_reasons,
        )


def assess_text_quality(
    text: str,
    min_words: int = 15,
    min_printable_ratio: float = 0.85,
) -> QualityMetrics:
    """Helper function for quick text quality evaluation."""
    assessor = TextQualityAssessor(
        min_words=min_words,
        min_printable_ratio=min_printable_ratio,
    )
    return assessor.evaluate(text)
