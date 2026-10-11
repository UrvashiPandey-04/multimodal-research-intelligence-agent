# Multimodal Research Intelligence Agent

An AI-powered research assistant that extracts, retrieves, and analyzes information from scientific research papers in PDF format, including text, tables, and figures, with evidence-grounded answers and page-level citations.

## 1. Project Overview

Scientific research papers contain information across multiple modalities, including narrative text, structured tables, figures, diagrams, and numerical results. Traditional text-based Retrieval-Augmented Generation (RAG) systems may fail to preserve table structures or interpret visual information, leading to incomplete or poorly grounded answers.

The **Multimodal Research Intelligence Agent** addresses these challenges through specialized PDF processing, vector-based retrieval, LangGraph agent orchestration, numerical calculations, and citation validation.

## 2. Problem Statement

Traditional PDF question-answering systems have several limitations:

- **Table Limitations:** Flattening tables into plain text can lose row-column relationships.
- **Figure Limitations:** Charts and diagrams may not be adequately represented by text alone.
- **Citation Limitations:** Answers may lack precise references to their source pages.
- **Numerical Errors:** Language models can produce incorrect calculations when comparing research metrics.
- **Cross-Paper Analysis:** Comparing evidence across multiple documents requires reliable retrieval and source tracking.

## 3. Existing Solution

Basic RAG systems generally divide PDF text into chunks, generate embeddings, store them in a vector database, and retrieve relevant chunks through similarity search.

Although effective for many text-based questions, this approach may not adequately handle structured tables, visual figures, precise provenance, or numerical comparisons.

## 4. Proposed Solution

The project combines multimodal document processing with evidence-grounded retrieval and agent-based reasoning.

- **Multimodal PDF Processing:** Extracts text, tables, figures, and captions using PyMuPDF and pdfplumber, with OCR support through Tesseract.
- **Metadata Tracking:** Associates extracted evidence with metadata such as paper, page, modality, and source identifier.
- **Vector Search:** Uses BGE embeddings and ChromaDB to retrieve relevant research content.
- **Agentic Workflow:** Uses LangGraph to coordinate text search, table search, figure search, and calculation tools.
- **Figure Analysis:** Uses Gemini 2.5 Flash for visual interpretation and evidence synthesis.
- **Safe Calculations:** Performs numerical operations through a restricted Python-based calculator rather than unrestricted `eval()`.
- **Citation Validation:** Checks generated citation references against retrieved evidence before displaying answers.
- **Evidence-Grounded Responses:** Presents answers alongside supporting evidence and source provenance.

## 5. System Architecture

```text
                  Research Paper PDFs
                          |
                          v
                 PDF Ingestion Pipeline
                          |
             +------------+------------+
             |            |            |
             v            v            v
           Text         Tables       Figures
             |            |            |
             v            v            v
          Chunking    Extraction   Image/Captions
             |            |            |
             +------------+------------+
                          |
                          v
                Embedding and Indexing
                  (BGE Small v1.5)
                          |
                          v
                       ChromaDB
                          |
                          v
                     User Query
                          |
                          v
                  LangGraph Agent
                          |
             +------------+------------+
             |            |            |
             v            v            v
        Text Search   Table Search  Figure Search
             |            |            |
             +------------+------------+
                          |
                    Calculator Tool
                          |
                          v
                Evidence Fusion and
                  Citation Validation
                          |
                          v
                 Gemini 2.5 Flash
                          |
                          v
               Answer + Supporting
                Evidence + Citations
                          |
                          v
                    Streamlit UI
```

## 6. Tech Stack

- **Programming Language:** Python
- **LLM and Vision:** Google Gemini 2.5 Flash, `google-genai`
- **Embeddings:** `sentence-transformers`, `BAAI/bge-small-en-v1.5`
- **Vector Database:** ChromaDB
- **PDF Processing:** PyMuPDF, pdfplumber
- **OCR:** Tesseract OCR, pytesseract, Pillow
- **Agent Orchestration:** LangGraph, LangChain Core
- **Backend and Interface:** FastAPI, Streamlit
- **Testing and Configuration:** pytest, python-dotenv

## 7. Research Papers and Dataset

The system is designed to work with scientific research PDFs. The current documented evaluation set includes the following papers:

1. **ConvLSTM:** *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting* — [arXiv:1506.04214](https://arxiv.org/abs/1506.04214)
2. **Deep DCNN Flood:** *A deep convolutional neural network model for rapid prediction of fluvial flood inundation* — [arXiv:2006.11555](https://arxiv.org/abs/2006.11555)
3. **Remote Sensing Flood:** *Deep Learning for Remote Sensing to Improve Flood Inundation Mapping* — [arXiv:2606.02310](https://arxiv.org/abs/2606.02310)

Place the corresponding PDF files in `data/papers/` using the filenames expected by the application.

## 8. Installation and Setup

### Prerequisites

- Python version supported by the project's dependencies
- Git
- A Google Gemini API key for LLM and visual-analysis functionality
- Tesseract OCR installed if OCR fallback is required

### Step 1: Clone the Repository

```bash
git clone https://github.com/UrvashiPandey-04/multimodal-research-intelligence-agent.git
cd multimodal-research-intelligence-agent
```

### Step 2: Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root and add your API key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Keep `.env` out of version control. Never commit API keys or other credentials.

If your repository provides a `.env.example` file, you can copy it and populate the required values instead.

## 9. Running the Application

### Launch the Streamlit Interface

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

### Run the Test Suite

```bash
python -m pytest tests/ -v
```

### Run the Evaluation Suite

```bash
python scripts/run_evaluation.py
```

If supported by the current repository version, this generates evaluation reports in `reports/`.

### Run the Smoke Test

```bash
python scripts/smoke_test.py
```

Run these commands only if the corresponding scripts are present in your checkout.

## 10. Key Application Features

- **PDF Upload and Ingestion:** Upload research PDFs within the application's documented limits of 20 MB and 50 pages.
- **Paper Selection:** Query a selected research paper or search across all available papers.
- **Text Retrieval:** Retrieve relevant passages from indexed research documents.
- **Table Retrieval:** Search extracted tables while retaining their source information.
- **Figure Retrieval:** Access extracted figure images and associated captions, with visual analysis where available.
- **Numerical Calculations:** Use a dedicated calculator tool for supported arithmetic operations.
- **Evidence Presentation:** Display retrieved evidence grouped by source paper to maintain provenance.
- **Citation Validation:** Validate citation references against the available evidence.
- **Quota Handling:** Handle Gemini quota errors with a user-facing warning while preserving retrieved evidence where supported.
- **Execution Trace:** Display selected execution details, tool usage, evidence counts, and citation-validation status.

## 11. Architecture and Code Organization

The application separates document processing, retrieval, agent orchestration, citation validation, and the user interface.

| Component | Responsibility |
|---|---|
| `src/ingestion/` | PDF parsing, OCR, chunking, tables, and figures |
| `src/models/` | Embedding and visual-analysis components |
| `src/retrieval/` | Vector, table, and figure retrieval |
| `src/agent/` | LangGraph workflow, query analysis, calculator, and context fusion |
| `src/citations/` | Evidence registry and citation validation |
| `app/streamlit_app.py` | Streamlit interface implementation |
| `streamlit_app.py` | Root application entry point |

The exact files and responsibilities should be checked against the current repository structure.

## 12. Reliability and Limitations

- **API Availability:** Gemini-dependent functionality requires valid API credentials and available quota.
- **OCR Dependency:** Scanned-document OCR requires a working Tesseract installation.
- **Extraction Quality:** PDF layouts, scanned pages, complex tables, and figure quality can affect extraction results.
- **Evidence Availability:** Answers depend on the quality and relevance of retrieved evidence.
- **Local Storage:** Generated vector indexes and extracted assets require local storage and may need to be regenerated when documents change.
- **Citation Scope:** Citation validation checks references against retrieved evidence; it does not independently establish that every generated claim is scientifically correct.

When the LLM service is unavailable or quota is exhausted, the application may provide a fallback response based on retrieved evidence rather than a model-generated answer.

## 13. Testing and Evaluation

The project includes tests for its ingestion, retrieval, agent, and citation-related functionality, along with evaluation and smoke-test scripts where available.

Run the test suite with:

```bash
python -m pytest tests/ -v
```

Use the evaluation script to assess the system against its configured research questions and expected results. Refer to the generated reports for the metrics actually measured by the current implementation.

## 14. Data and Version Control

Keep configuration secrets and generated artifacts separate from source code.

- **Configuration:** Store API keys in `.env`; exclude it from Git.
- **Research Papers:** Keep the required PDFs in `data/papers/`.
- **Generated Vector Store:** Store ChromaDB artifacts locally.
- **Extracted Figures and Cache:** Keep generated images and cached responses out of version control unless explicitly required.
- **Evaluation Artifacts:** Commit only the reports and datasets intended to be tracked by the project.

Check `.gitignore` and the repository's current tracking rules before adding or removing files.

## 15. Future Improvements

- Expand evaluation coverage across additional research domains.
- Improve extraction and retrieval for complex tables and scanned documents.
- Strengthen multimodal retrieval and cross-paper comparison.
- Add more systematic evaluation of citation correctness and answer grounding.
- Improve observability and performance for larger document collections.

## Author

**Urvashi Pandey**

B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)

GitHub: [UrvashiPandey-04](https://github.com/UrvashiPandey-04)

**Repository:** [Multimodal Research Intelligence Agent](https://github.com/UrvashiPandey-04/multimodal-research-intelligence-agent)

