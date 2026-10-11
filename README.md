# Multimodal Research Intelligence Agent

A production-grade AI-powered research assistant designed to parse, retrieve, reason, and cite complex multimodal scientific research papers (PDFs).

---

## 1. Project Overview
Scientific research papers contain rich information distributed across multiple modalities: unstructured narrative text, structured tables, visual figures, diagrams, and numerical result benchmarks. Standard text-only Retrieval-Augmented Generation (RAG) pipelines fail to capture critical facts stored inside tables and visual charts, often leading to hallucinations or incomplete answers.

The **Multimodal Research Intelligence Agent** solves this challenge by ingesting PDFs into distinct modal components (Text, Tables, Figures/Captions), indexing them into a unified vector store with fine-grained page provenance, routing queries via an explicit LangGraph tool-calling agent, performing safe numerical calculations, and generating evidence-grounded responses with strict page-level citations.

---

## 2. Problem Statement
Traditional "Chat with PDF" applications treat research papers solely as plain text streams. This leads to severe limitations:
- **Table Failure:** Flattened text representation destroys column-row alignment and structural relationships.
- **Figure Blindness:** Visual charts, architectural diagrams, and flowcharts are ignored or degraded to text captions.
- **Lack of Evidence Provenance:** Generated answers lack verifiable citations pointing to specific papers, pages, tables, or figures.
- **Calculation Hallucinations:** LLMs routinely fail at exact numerical calculations (e.g., percentage change between benchmark metrics across papers).

---

## 3. Existing Solution
Standard RAG frameworks (e.g., basic LangChain/LlamaIndex starters) split PDF text into fixed-size chunks, store them in a vector database, and perform top-k similarity search over text blocks.

---

## 4. Existing Solution Limitations
1. **Modal Degradation:** Tables and figures lose structural metadata.
2. **Coarse-Grained Citations:** Answers refer vaguely to the document name without exact page/table/figure location.
3. **Single-Paper Tunnel Vision:** Comparative queries across multiple documents struggle without explicit query decomposition.
4. **Unreliable Arithmetic:** LLMs miscalculate numerical differences or performance improvements.

---

## 5. Proposed Solution
Our architecture implements:
- **Multimodal Decomposition:** Specialized ingestion parsers for PyMuPDF (text/figures), pdfplumber (structured tables), and Tesseract OCR (scanned fallback).
- **Metadata Provenance Tracking:** Every extracted chunk, table, and figure carries explicit metadata: `paper`, `page`, `modality`, and `source_id`.
- **Explicit LangGraph Tool-Calling Agent:** Controlled routing to dedicated tools (`search_text`, `search_tables`, `search_figures`, `calculate`).
- **Multimodal Reasoning:** Gemini 2.5 Flash for visual interpretation of retrieved figure images and evidence synthesis.
- **Safe Calculator Tool:** Isolated python arithmetic handler for exact calculations without unrestricted `eval()`.
- **Strict Citation Validation:** Automated cross-validation verifying that every generated citation exists in retrieved context before displaying.

---

## 6. Architecture

```
                    Research PDFs (data/papers/)
                         |
                         v
                Document Ingestion (PyMuPDF / OCR)
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        Text           Tables         Figures
          |              |              |
          v              v              v
    Text Chunks     Structured       Captions /
    + Metadata        Tables         Figure Images
          |              |              |
          +--------------+--------------+
                         |
                         v
                Embedding / Indexing (BGE Small v1.5)
                         |
                         v
                    ChromaDB Vector Store
                         |
                         v
                    User Query
                         |
                         v
                  LangGraph Router / Agent
                         |
          +--------------+--------------+--------------+
          |              |              |              |
          v              v              v              v
     Text Search    Table Search   Figure Search   Calculator
          |              |              |              |
          +--------------+--------------+--------------+
                         |
                         v
                  Context Fusion & Grounding
                         |
                         v
               Gemini 2.5 Flash Response
                         |
                         v
            Answer + Evidence + Provenance Citations
```

---

## 7. Tech Stack
- **Language:** Python 3.11+
- **LLM & Vision:** Google Gemini 2.5 Flash (`google-genai` SDK)
- **Embeddings:** `sentence-transformers` (`BAAI/bge-small-en-v1.5` - run locally)
- **Vector Database:** ChromaDB
- **PDF Extraction:** PyMuPDF (`fitz`), `pdfplumber`
- **OCR Fallback:** Tesseract OCR (`pytesseract`, `Pillow`)
- **Orchestration:** LangGraph & LangChain Core
- **Backend/UI:** FastAPI, Streamlit
- **Testing & Env:** `pytest`, `python-dotenv`

---

## 8. Canonical Dataset / Papers
The system is tested against three canonical hydro-meteorological deep learning papers:
1. **ConvLSTM:** *"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"* (arXiv:1506.04214) -> Expected file: `data/papers/convlstm_precipitation_nowcasting.pdf`
2. **Deep DCNN Flood:** *"A deep convolutional neural network model for rapid prediction of fluvial flood inundation"* (arXiv:2006.11555) -> Expected file: `data/papers/dcnn_fluvial_flood_inundation.pdf`
3. **Remote Sensing Flood:** *"Deep Learning for Remote Sensing to Improve Flood Inundation Mapping"* (arXiv:2606.02310) -> Expected file: `data/papers/remote_sensing_flood_mapping.pdf`

*Note: PDF files must be placed in `data/papers/` directory.*

---

## 9. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11/3.14)
- Tesseract OCR engine installed (Optional fallback: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

### Setup Instructions
1. Clone the repository and navigate to root:
   ```bash
   git clone https://github.com/UrvashiPandey-04/multimodal-research-intelligence-agent.git
   cd multimodal-research-intelligence-agent
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # Linux/macOS
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Set your `GEMINI_API_KEY` in `.env`.

---

## 10. Command Reference & API Usage

### 1. Launch Streamlit Web Application (Phase 11 Frontend)
```bash
streamlit run streamlit_app.py
```
*Opens interactive research dashboard in your browser at `http://localhost:8501`.*

### 2. Run Complete Unit & Integration Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Run Single Query API Entrypoint (Python API)
```python
from src import answer_query, ResearchResponse

response: ResearchResponse = answer_query("Which model achieves the lowest Rainfall-MSE in Table 1?")
print("Answer:", response.answer)
print("Validation Status:", response.validation_status)
print("Citations:", response.citations)
```

### 4. Run Benchmark Evaluation Suite (Phase 9 Evaluation)
```bash
python scripts/run_evaluation.py
```
*Generates machine-readable `reports/evaluation_results.json` and human-readable `reports/evaluation_summary.md`.*

### 5. Run End-to-End System Smoke Test
```bash
python scripts/smoke_test.py
```

### 6. Example Research Queries
- **Text Question:** *"What main research problem does the ConvLSTM paper address?"*
- **Table Question:** *"Which model achieves the lowest Rainfall-MSE score in Table 1?"*
- **Figure Question:** *"What does Figure 1 illustrate regarding 2D image transformations over time?"*
- **Cross-Paper Question:** *"Compare the purpose of canonical_sample.pdf vs convlstm_precipitation_nowcasting.pdf."*
- **Numerical Calculation:** *"What is the percentage improvement from Rainfall-MSE 0.120 (FC-LSTM) to 0.089 (ConvLSTM 5x5)?"*
- **Insufficient Evidence Question:** *"What color is the author's cat in the ConvLSTM paper?"*

---

## 11. Streamlit Web Interface Architecture

The Streamlit UI (`app/streamlit_app.py` / `streamlit_app.py`) serves as a pure presentation layer for the underlying agentic backend.

### Architectural Flow:
```
User (Browser)
  ↓
Streamlit UI (app/streamlit_app.py)
  ↓
answer_query(query, agent=cached_agent)  [Canonical Backend Entrypoint]
  ↓
LangGraph Research Agent
  ├── Text Retriever (BGE Dense Vectors in ChromaDB)
  ├── Table Retriever (pdfplumber Markdown & DataFrames)
  ├── Figure Retriever + Gemini 2.5 VLM Analysis
  └── AST SafeCalculator
  ↓
Context Fusion & Grounding
  ↓
Answer Generation & Citation Validation
  ↓
ResearchResponse Object
  ↓
Streamlit UI Render (Answer + Citations + Grouped Evidence + Safe Trace)
```

### Key UI Features:
- **Resource Caching:** `@st.cache_resource` caches the canonical paper vector indices once per application runtime (`get_cached_agent()`), avoiding expensive PDF re-ingestion on Streamlit reruns.
- **Custom PDF Upload:** Upload and ingest custom research papers (≤ 20 MB, ≤ 50 pages) dynamically into the active vector index with real-time `IngestionReport` metrics.
- **Target Document Filtering:** Select specific papers or query globally across "All Research Papers" with multi-modal provenance preservation.
- **Resilient Quota Handling:** Gracefully handles Gemini API HTTP 429 quota exhaustion (`RESOURCE_EXHAUSTED`) with clean warnings while preserving and displaying all retrieved multimodal evidence (text, tables, figures).
- **Answer Presentation:** Clean separation between final answer, insufficient evidence warnings, and detailed provenance.
- **Multimodal Evidence:** Grouped by source paper to prevent provenance ambiguity; tabular evidence is rendered via interactive `st.dataframe()` with Markdown fallback; figure evidence displays extracted PNG cropped images alongside captions and VLM findings.
- **Safe Metadata Trace:** Expandable execution details showing query analysis, tool selection, citation validation status, and evidence counts without exposing internal chain-of-thought or model secrets.
- **Sidebar Integration:** Displays available PDF papers in `data/papers/`, supported modalities, and non-secret system status.

---

## 12. Known Limitations & Scope
- **Document Scale:** Vector index is pre-configured for PDF files placed in `data/papers/`. Adding new PDFs via UI requires triggering ingestion (<20MB, <50 pages).
- **OCR Engine:** Scanned fallback OCR relies on local Tesseract installation (`tesseract.exe`). If uninstalled, PyMuPDF native text extraction remains active.
- **VLM API:** Figure visual analysis requires `GEMINI_API_KEY`. If unconfigured, system gracefully falls back to text captions and heuristic descriptions.

---

## 13. Failure-Mode & Reliability Matrix

| Failure Mode | Detection Mechanism | Internal System Behavior | User-Facing Result |
|---|---|---|---|
| **Empty / Whitespace Query** | Input validation in `answer_query()` | Rejects execution before LangGraph invocation | `validation_status="insufficient_evidence"`, asks for non-empty query |
| **No Relevant Evidence Found** | `fused_context.evidence_count == 0` | Triggers explicit LLM grounding guardrail | `validation_status="insufficient_evidence"`, *"I don't have enough evidence..."* |
| **Missing Document / Vector Store** | `FileNotFoundError` in retriever | Logs error event, sets `errors` array | Returns graceful error response without stack trace leakage |
| **Gemini API Failure / Missing Key** | API exception or missing `GEMINI_API_KEY` | Triggers deterministic evidence summary fallback | `validation_status="valid"` with evidence-grounded summary fallback |
| **Gemini API Quota (HTTP 429)** | `is_quota_error()` / `GeminiQuotaExceededError` | Fails fast without redundant retries; preserves retrieved chunks | `validation_status="quota_exceeded"`, clean warning displayed, multimodal evidence preserved |
| **VLM Execution Failure** | Exception in `analyze_figure()` | Falls back to PyMuPDF crop + figure caption heuristics | Returns answer based on text caption without failing pipeline |
| **Calculator Syntax / Div-by-Zero** | `SafeCalculator` AST evaluation check | Sets `calc_res.success = False`, records error | `validation_status="valid"` relying on text evidence without crash |
| **LLM Invented Citation Tag** | `CitationValidator` registry cross-check | Rejects hallucinated `[E99]` citation tags | Formats answer preserving only verified provenance citations |

---

## 14. Data Directory & Persistence Commit Policy

- **Committed to Git:**
  - `data/papers/` (Canonical paper PDFs)
  - `data/evaluation/evaluation_dataset.json` (30 ground-truth evaluation items)
  - `reports/evaluation_summary.md` (Summary markdown benchmark reports)
- **Ignored from Git (`.gitignore`):**
  - `.env` (API keys and credentials)
  - `data/chroma/` (Generated ChromaDB vector storage)
  - `data/figures/` (Cropped PNG figure images)
  - `data/cache/` (Disk-cached VLM responses)
  - `__pycache__/`, `.pytest_cache/`

---

## 15. System Architecture & Component Mapping

- **PDF Ingestion & OCR:** `src/ingestion/pdf_parser.py`, `src/ingestion/ocr.py`, `src/ingestion/normalizer.py`
- **Chunking & Vector Storage:** `src/ingestion/chunker.py`, `src/models/embeddings.py`, `src/retrieval/vector_store.py`
- **Table Extraction & Search:** `src/ingestion/table_parser.py`, `src/retrieval/table_repository.py`, `src/retrieval/table_retriever.py`
- **Figure Extraction & VLM:** `src/ingestion/figure_parser.py`, `src/retrieval/figure_repository.py`, `src/models/vlm_analyzer.py`, `src/retrieval/figure_retriever.py`
- **LangGraph Agent Orchestration:** `src/agent/graph.py`, `src/agent/nodes.py`, `src/agent/query_analyzer.py`, `src/agent/calculator.py`, `src/agent/context_fuser.py`
- **Citation Provenance:** `src/citations/evidence_registry.py`, `src/citations/citation_validator.py`
- **Single Entry Point API:** `src/api.py` (`answer_query`)
- **Evaluation & Runner:** `src/evaluation/schema.py`, `src/evaluation/dataset.py`, `src/evaluation/metrics.py`, `src/evaluation/runner.py`, `scripts/run_evaluation.py`

