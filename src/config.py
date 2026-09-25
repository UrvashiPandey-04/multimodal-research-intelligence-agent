"""
Centralized Configuration Module for Multimodal Research Intelligence Agent.
Manages environment variables, default model identifiers, and relative file system paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API Keys & Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LLM & Vision Model Identifiers
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Data & Persistence Directories (Relative to project root)
PAPERS_DIR = os.getenv("PAPERS_DIR", str(PROJECT_ROOT / "data" / "papers"))
FIGURES_DIR = os.getenv("FIGURES_DIR", str(PROJECT_ROOT / "data" / "figures"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", str(PROJECT_ROOT / "data" / "chroma"))
EVALUATION_DATA_DIR = os.getenv("EVALUATION_DATA_DIR", str(PROJECT_ROOT / "data" / "evaluation"))
REPORTS_DIR = os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "reports"))

# OCR Engine Configuration
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in ("true", "1", "yes")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def get_config_summary() -> dict:
    """Return a sanitized summary of current configuration settings."""
    return {
        "gemini_api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"),
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chroma_persist_dir": CHROMA_PERSIST_DIR,
        "papers_dir": PAPERS_DIR,
        "figures_dir": FIGURES_DIR,
        "evaluation_data_dir": EVALUATION_DATA_DIR,
        "ocr_enabled": OCR_ENABLED,
    }
