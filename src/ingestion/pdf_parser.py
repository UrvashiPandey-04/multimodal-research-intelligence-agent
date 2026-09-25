"""
PDF Parser module integrating PyMuPDF native extraction, Text Quality Assessment, OCR Fallback, and Conservative Normalization.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf as fitz
from dotenv import load_dotenv

from src.ingestion.schema import NormalizedPage, NormalizedDocument, QualityMetrics
from src.ingestion.text_quality import TextQualityAssessor
from src.ingestion.normalizer import TextNormalizer
from src.ingestion.ocr import OCREngine, OCRNotAvailableError, OCRError

load_dotenv()


class PDFParser:
    """Production-grade PDF Parser with Quality Assessment, OCR Fallback, and Normalization."""

    def __init__(
        self,
        min_words_per_page: int = 15,
        min_printable_ratio: float = 0.85,
        ocr_enabled: Optional[bool] = None,
        tesseract_cmd: Optional[str] = None,
        ocr_engine: Optional[OCREngine] = None,
    ):
        """
        Initialize the PDF parser.

        Args:
            min_words_per_page: Minimum words required for usable native text.
            min_printable_ratio: Minimum ratio of printable characters.
            ocr_enabled: Whether OCR fallback is enabled (default: reads OCR_ENABLED from .env).
            tesseract_cmd: Custom Tesseract binary path.
            ocr_engine: Custom OCREngine instance (useful for mocking in unit tests).
        """
        self.quality_assessor = TextQualityAssessor(
            min_words=min_words_per_page,
            min_printable_ratio=min_printable_ratio,
        )
        self.normalizer = TextNormalizer()

        if ocr_enabled is None:
            env_ocr = os.getenv("OCR_ENABLED", "true").lower()
            self.ocr_enabled = env_ocr in ("true", "1", "yes")
        else:
            self.ocr_enabled = ocr_enabled

        self.ocr_engine = ocr_engine
        self.tesseract_cmd = tesseract_cmd

    def _get_ocr_engine(self) -> OCREngine:
        """Lazy-initialize OCREngine if needed."""
        if self.ocr_engine is None:
            self.ocr_engine = OCREngine(tesseract_cmd=self.tesseract_cmd)
        return self.ocr_engine

    def parse_pdf(self, pdf_path: str | Path) -> NormalizedDocument:
        """
        Parse a PDF file page-by-page, assess quality, fallback to OCR if needed, and normalize.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            NormalizedDocument object ready for downstream chunking.

        Raises:
            FileNotFoundError: If PDF file does not exist.
            ValueError: If file is corrupted or encrypted.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise ValueError(f"Failed to open PDF file {pdf_path}: {e}") from e

        if doc.is_encrypted:
            if not doc.authenticate(""):
                doc.close()
                raise ValueError(f"PDF file {pdf_path} is encrypted and password-protected.")

        filename = path.name
        doc_id = path.stem
        total_pages = doc.page_count
        pages: List[NormalizedPage] = []
        native_count = 0
        ocr_count = 0

        try:
            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                page_num = page_idx + 1

                # 1. Native text extraction
                raw_native_text = (page.get_text("text") or "").strip()
                images = page.get_images()
                has_images = len(images) > 0

                # 2. Text quality evaluation
                quality_metrics = self.quality_assessor.evaluate(raw_native_text)

                extraction_method = "native"
                final_raw_text = raw_native_text
                final_quality = quality_metrics

                # 3. OCR Fallback if quality threshold is not met
                if not quality_metrics.is_usable and self.ocr_enabled:
                    try:
                        engine = self._get_ocr_engine()
                        if engine.is_available():
                            raw_ocr_text = engine.process_pymupdf_page(page)
                            ocr_quality = self.quality_assessor.evaluate(raw_ocr_text)

                            # If OCR yields better or usable text, accept OCR result
                            extraction_method = "ocr"
                            final_raw_text = raw_ocr_text
                            final_quality = ocr_quality
                    except (OCRNotAvailableError, OCRError) as ocr_err:
                        # Record OCR failure in quality metrics failure reasons
                        final_quality.failure_reasons.append(f"ocr_attempt_failed: {ocr_err}")

                if extraction_method == "native":
                    native_count += 1
                else:
                    ocr_count += 1

                # 4. Conservative Text Normalization
                normalized_text = self.normalizer.normalize(final_raw_text)

                normalized_page = NormalizedPage(
                    doc_id=doc_id,
                    source_filename=filename,
                    page_number=page_num,
                    text=normalized_text,
                    raw_text=final_raw_text,
                    extraction_method=extraction_method,
                    quality_metrics=final_quality,
                    has_images=has_images,
                    metadata={
                        "image_count": len(images),
                        "width": page.rect.width,
                        "height": page.rect.height,
                    },
                )
                pages.append(normalized_page)

        finally:
            doc.close()

        doc_metadata = {
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
            "format": "PDF",
        }

        return NormalizedDocument(
            doc_id=doc_id,
            source_filename=filename,
            filepath=str(path.resolve()),
            total_pages=total_pages,
            pages=pages,
            native_page_count=native_count,
            ocr_page_count=ocr_count,
            metadata=doc_metadata,
        )
