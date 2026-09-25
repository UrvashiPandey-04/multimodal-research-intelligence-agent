# Multimodal Research Intelligence Agent — Comprehensive Interview Guide

This guide provides technical explanations for candidates or developers presenting the Multimodal Research Intelligence Agent in technical interviews or code reviews. All answers reflect the actual Phase 1–11 codebase implementation.

---

## A. Project Overview
The **Multimodal Research Intelligence Agent** is an evidence-grounded AI research assistant designed to answer complex scientific questions over PDF research papers. Unlike generic text-only chatbots, it handles multimodal elements (text narratives, structured tables, visual figures/diagrams, and exact numerical calculations) with deterministic page-level provenance citations.

---

## B. Why RAG is Needed
Large Language Models (LLMs) suffer from context window limits, lack of domain-specific/recent paper access, and a tendency to hallucinate plausible-sounding technical claims. Retrieval-Augmented Generation (RAG) grounds the LLM by fetching authoritative text, table, and figure snippets directly from local paper indices before generation.

---

## C. Why ChromaDB is Used
`ChromaDB` is a lightweight, open-source vector database that supports local persistence without requiring external cloud database subscriptions. It seamlessly stores dense embeddings along with rich payload metadata (`paper`, `page`, `modality`, `source_id`), allowing fast vector similarity search over text, tables, and figures.

---

## D. Why BGE Embeddings are Used
`BAAI/bge-small-en-v1.5` is a top-tier open embedding model on the MTEB (Massive Text Embedding Benchmark) leaderboard for retrieval tasks. It runs locally via `sentence-transformers`, generating 384-dimensional dense vectors with low memory footprint and high semantic accuracy for scientific text.

---

## E. How Chunking Works
The text chunking pipeline (`src/ingestion/chunker.py`) splits normalized page text into deterministic chunks targeting ~800 tokens with ~120 token overlap. It preserves strict metadata for every chunk: document ID, paper filename, page number, extraction method (`native` vs `ocr`), and word counts.

---

## F. How Citation Grounding Works
The model generates inline evidence tags (e.g. `[E1]`, `[E2]`). The `CitationValidator` (`src/citations/citation_validator.py`) intercepts the response and cross-references every tag against the authoritative request-scoped `EvidenceRegistry`. Any tag that does not exist in retrieved evidence or points to invalid metadata is stripped, ensuring the LLM cannot invent citations.

---

## G. Why an EvidenceRegistry Exists
The LLM cannot be trusted as the authoritative source of citation metadata. `EvidenceRegistry` (`src/citations/evidence_registry.py`) acts as a request-scoped single source of truth. As tools retrieve chunks, tables, figures, or calculation outputs, they register items into the registry, generating unique IDs (`E1`, `E2`, ...) and binding exact source provenance.

---

## H. How Table Extraction Works
Native text extraction destroys table column-row alignment. `TableParser` (`src/ingestion/table_parser.py`) uses `pdfplumber` to detect visual gridlines and bounding boxes, extracting matrix headers and cell arrays. Tables are formatted as both searchable Markdown matrices and structured JSON objects for accurate LLM table reasoning.

---

## I. How Figure Extraction Works
`FigureParser` (`src/ingestion/figure_parser.py`) uses `PyMuPDF` to locate embedded vector graphics and image blocks, extracting bounding coordinates and associated captions. Cropped PNG images are saved to `data/figures/` with explicit metadata linking paper name and page number.

---

## J. Why a VLM is Required
Text captions alone do not capture the dense visual information inside scientific charts, loss curves, and system architecture diagrams. `VLMFigureAnalyzer` (`src/models/vlm_analyzer.py`) passes cropped figure images alongside captions to `Gemini 2.5 Flash` to extract visual observations, trend summaries, and structural relationships.

---

## K. How LangGraph is Used
`LangGraph` (`src/agent/graph.py`) provides an explicit 8-node stateful graph workflow (`ResearchAgentState`). Rather than letting an autonomous LLM loop unpredictably, LangGraph enforces deterministic state transitions: Query Analysis -> Tool Selection -> Multimodal Retrieval -> Context Fusion -> Answer Generation -> Citation Validation.

---

## L. How Tool Selection Works
`QueryAnalyzer` (`src/agent/query_analyzer.py`) uses rule-based heuristic pattern matching to analyze query intent. It identifies whether a query requires text search (`search_text`), table metrics (`search_tables`), figure visual analysis (`search_figures`), numerical calculations (`calculate`), or cross-paper comparison (`requires_multiple_papers`).

---

## M. How Cross-Paper Reasoning Works
`TextRetriever.search_diversified()` implements round-robin document sampling. When a query asks to compare multiple papers, standard top-k similarity search might retrieve chunks from only one dominant document. Diversified search enforces balanced chunk candidates across distinct document IDs.

---

## N. How Insufficient Evidence is Handled
If retrieval score thresholds are not met or zero evidence items are returned, `ContextBuilder` sets `is_sufficient=False`. The prompt instructs Gemini to output an explicit fallback message: *"I don't have enough evidence in the provided research papers to answer this."* Validation status is flagged as `insufficient_evidence`.

---

## O. How the Calculator is Made Safe
Unrestricted Python `eval()` poses severe code execution vulnerabilities. `SafeCalculator` (`src/agent/calculator.py`) parses math expressions into an Abstract Syntax Tree (AST) using Python's `ast` module and evaluates nodes against a strict whitelist of constants and binary operators (`+`, `-`, `*`, `/`, `%`, `**`).

---

## P. How Evaluation Works
The benchmark evaluation framework (`src/evaluation/runner.py`) runs 30 ground-truth verified questions across text, table, figure, cross-paper, and calculation categories. It measures Recall@K, tool routing accuracy, citation validity rate, multimodal retrieval success, and cross-paper coverage, generating JSON and Markdown reports.

---

## Q. Main Limitations
1. **Document Scope:** Vector stores index local PDF files placed in `data/papers/`.
2. **OCR Dependency:** OCR fallback relies on Tesseract engine installation (`tesseract.exe`); if missing, native text extraction remains active.
3. **VLM API:** Figure visual interpretation requires an active `GEMINI_API_KEY`; without it, fallback text captions are used.

---

## R. Possible Future Improvements
1. **Dynamic Document Upload:** Adding real-time drag-and-drop PDF ingestion in Streamlit.
2. **Hybrid Search:** Combining dense vector embeddings with sparse BM25 keyword search.
3. **Reranking:** Incorporating a cross-encoder reranker (e.g. `bge-reranker-large`) after initial vector retrieval.
