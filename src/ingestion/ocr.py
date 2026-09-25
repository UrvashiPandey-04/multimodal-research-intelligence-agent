"""
OCR Engine module integrating Tesseract OCR fallback for scanned or low-text PDF pages.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Optional
from PIL import Image
import pytesseract
import pymupdf as fitz
from dotenv import load_dotenv

load_dotenv()


class OCRNotAvailableError(Exception):
    """Raised when Tesseract OCR engine is not installed or unavailable."""
    pass


class OCRError(Exception):
    """Raised when an error occurs during OCR image processing."""
    pass


class OCREngine:
    """Tesseract OCR Engine for extracting text from rendered page images."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize Tesseract OCR engine.

        Args:
            tesseract_cmd: Custom path to tesseract.exe binary. If None, checks env var TESSERACT_CMD
                           or default system installation path on Windows.
        """
        # Resolve tesseract executable path
        cmd = (
            tesseract_cmd
            or os.getenv("TESSERACT_CMD")
            or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        if cmd and Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = str(cmd)
        elif tesseract_cmd:
            raise OCRNotAvailableError(f"Specified Tesseract binary not found at: {tesseract_cmd}")

    def is_available(self) -> bool:
        """Check whether Tesseract engine is installed and functional."""
        try:
            version = pytesseract.get_tesseract_version()
            return version is not None
        except Exception:
            return False

    def process_image(self, image: Image.Image, lang: str = "eng") -> str:
        """
        Run Tesseract OCR on a PIL Image object.

        Args:
            image: PIL Image instance.
            lang: OCR language model code (default: "eng").

        Returns:
            Extracted text string.

        Raises:
            OCRNotAvailableError: If Tesseract engine is unavailable.
            OCRError: If OCR execution fails.
        """
        if not self.is_available():
            raise OCRNotAvailableError(
                "Tesseract OCR is not available. Please install Tesseract-OCR and set TESSERACT_CMD."
            )

        try:
            text = pytesseract.image_to_string(image, lang=lang)
            return text or ""
        except Exception as e:
            raise OCRError(f"OCR processing failed: {e}") from e

    def process_pymupdf_page(self, page: fitz.Page, dpi: int = 300, lang: str = "eng") -> str:
        """
        Render PyMuPDF Page to a high-DPI pixmap image and execute OCR.

        Args:
            page: PyMuPDF Page object.
            dpi: Resolution DPI for rendering page image (default: 300).
            lang: Language model code (default: "eng").

        Returns:
            Extracted OCR text string.
        """
        try:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(BytesIO(img_bytes))
            return self.process_image(pil_img, lang=lang)
        except (OCRNotAvailableError, OCRError):
            raise
        except Exception as e:
            raise OCRError(f"Failed to render PDF page for OCR: {e}") from e
