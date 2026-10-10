"""
Streamlit Web Interface for Multimodal Research Intelligence Agent.
Provides an interactive research dashboard for evidence-grounded scientific QA over PDFs.
"""

import os
import sys
import tempfile
from typing import Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import get_config_summary
from src.api import answer_query, ResearchResponse
from src.evaluation.runner import EvaluationRunner
from src.agent.graph import ResearchAgent
from src.ingestion.pipeline import ingest_pdf, IngestionReport, PDFValidationError

# Page configuration
st.set_page_config(
    page_title="Multimodal Research Intelligence Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 1. Resource Caching: Index papers once per application runtime
@st.cache_resource(show_spinner="Loading sample research papers...\nBuilding search index...\nThis may take a moment on the first run.")
def get_cached_agent() -> ResearchAgent:
    """Initialize agent and index canonical PDF papers into in-memory/persistent vector stores."""
    runner = EvaluationRunner()
    papers_dir = os.path.join(project_root, "data", "papers")
    agent = runner.prepare_environment_and_index_papers(papers_dir=papers_dir)
    return agent


def parse_markdown_table_to_dataframe(md_text: str) -> Optional[pd.DataFrame]:
    """Helper to safely parse a Markdown table into a pandas DataFrame."""
    if not md_text or "|" not in md_text:
        return None
    try:
        lines = [line.strip() for line in md_text.strip().split("\n") if line.strip()]
        if len(lines) < 3 or not lines[0].startswith("|"):
            return None
        headers = [c.strip() for c in lines[0].split("|")[1:-1]]
        rows = []
        for line in lines[2:]:
            if line.startswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) == len(headers):
                    rows.append(cells)
                elif len(cells) > 0:
                    # Pad or truncate row to match header length
                    padded = cells[: len(headers)] + [""] * max(0, len(headers) - len(cells))
                    rows.append(padded)
        if headers and rows:
            return pd.DataFrame(rows, columns=headers)
    except Exception:
        pass
    return None


def main():
    cfg = get_config_summary()
    st.session_state.setdefault("uploaded_paper_names", [])

    # Initialize Agent
    try:
        agent = get_cached_agent()
    except Exception as e:
        st.error(f"Failed to initialize Research Agent index: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.markdown("---")

        # Custom PDF Upload
        st.subheader("📤 Upload Custom Research Paper")
        uploaded_file = st.file_uploader(
            "Upload a research paper PDF (<20MB, <50 pages):",
            type=["pdf"],
            key="custom_pdf_uploader",
        )

        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state["uploaded_paper_names"]:
                with st.spinner(f"Ingesting '{uploaded_file.name}' into vector index..."):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        report: IngestionReport = ingest_pdf(
                            pdf_path=tmp_path,
                            vector_store=agent.nodes.text_retriever.vector_store,
                            table_repo=agent.nodes.table_retriever.table_repo if hasattr(agent.nodes.table_retriever, 'table_repo') else None,
                            table_retriever=agent.nodes.table_retriever,
                            fig_extractor=agent.nodes.figure_retriever.fig_extractor if hasattr(agent.nodes.figure_retriever, 'fig_extractor') else None,
                            fig_repo=agent.nodes.figure_retriever.figure_repo if hasattr(agent.nodes.figure_retriever, 'figure_repo') else None,
                            figure_retriever=agent.nodes.figure_retriever,
                            paper_name=uploaded_file.name,
                        )

                        if report.success:
                            st.session_state["uploaded_paper_names"].append(uploaded_file.name)
                            st.success(
                                f"✅ Ingested **{report.paper_name}**\n\n"
                                f"• Pages: {report.pages} | Chunks: {report.chunks} | Tables: {report.tables} | Figures: {report.figures}"
                            )
                        else:
                            st.error(f"❌ Upload ingestion error: {', '.join(report.errors)}")
                    except Exception as ex:
                        st.error(f"❌ Failed to process PDF: {ex}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

        st.markdown("---")

        st.subheader("📚 Available Research Papers")
        canonical_pdfs = []
        papers_dir = os.path.join(project_root, "data", "papers")
        if os.path.exists(papers_dir):
            canonical_pdfs = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]

        all_available_pdfs = list(dict.fromkeys(canonical_pdfs + st.session_state["uploaded_paper_names"]))
        for pdf in all_available_pdfs:
            is_custom = pdf in st.session_state["uploaded_paper_names"]
            badge = " 🆕" if is_custom else ""
            st.markdown(f"• `{pdf}`{badge}")

        st.markdown("---")
        st.subheader("🛠️ Supported Modalities")
        st.markdown("- 📄 **Text Chunks:** BGE Dense Vector Search")
        st.markdown("- 📊 **Tables:** pdfplumber Structured Markdown & DataFrames")
        st.markdown("- 🖼️ **Figures:** PyMuPDF Cropped PNGs + Gemini 2.5 VLM")
        st.markdown("- 🧮 **Calculations:** AST SafeCalculator Engine")

        st.markdown("---")
        st.subheader("⚙️ System Status")
        api_status = "✅ Configured" if cfg["gemini_api_key_configured"] else "⚠️ Unconfigured (Missing Secret)"
        st.markdown(f"**Gemini API:** {api_status}")
        st.markdown(f"**LLM Model:** `{cfg['gemini_model']}`")
        st.markdown(f"**Embedding Model:** `{cfg['embedding_model']}` (`{cfg['embedding_provider']}`)")
        st.markdown(f"**OCR Status:** `{'Enabled' if cfg['ocr_enabled'] else 'Disabled'}`")

    # Main Interface Header
    st.title("🔬 Multimodal Research Intelligence Agent")
    st.markdown(
        "An evidence-grounded AI research assistant that parses, retrieves, reasons, and cites complex multimodal scientific research papers (PDFs)."
    )

    # Friendly API Key Check Screen
    if not cfg["gemini_api_key_configured"]:
        st.error("⚠️ **GEMINI_API_KEY is missing or unconfigured.**")
        st.info(
            "To enable Gemini 2.5 Flash synthesis and VLM figure analysis, please configure your API key:\n\n"
            "- **Streamlit Community Cloud:** Add `GEMINI_API_KEY = \"your_api_key\"` under **App Settings → Secrets**.\n"
            "- **Local Setup:** Create a `.env` file in the project root containing `GEMINI_API_KEY=your_actual_key`.\n\n"
            "[Get a Gemini API key from Google AI Studio](https://aistudio.google.com/)"
        )

    st.markdown("---")

    # Paper Filter Selection Control
    col_filter, col_preset = st.columns([1, 2])
    with col_filter:
        paper_filter_options = ["All Research Papers"] + all_available_pdfs
        selected_paper_filter = st.selectbox("🎯 Target Document Filter:", paper_filter_options, index=0)
        paper_filter_param = None if selected_paper_filter == "All Research Papers" else selected_paper_filter

    with col_preset:
        sample_queries = [
            "Select a sample question...",
            "What main research problem does the ConvLSTM paper address?",
            "Which model achieves the lowest Rainfall-MSE score in Table 1?",
            "What does Figure 1 illustrate regarding 2D image transformations over time?",
            "Compare the purpose of canonical_sample.pdf vs convlstm_precipitation_nowcasting.pdf.",
            "What is the percentage improvement from Rainfall-MSE 0.120 (FC-LSTM) to 0.089 (ConvLSTM 5x5)?",
            "What color is the author's cat in the ConvLSTM paper?",
        ]
        selected_sample = st.selectbox("💡 Preset Demo Query:", sample_queries, index=0)

    # Question Input
    default_text = selected_sample if selected_sample != "Select a sample question..." else ""
    user_query = st.text_input("Ask a research question about the papers:", value=default_text, placeholder="e.g., How does ConvLSTM differ from FC-LSTM?")

    col_ask, col_clear = st.columns([1, 5])
    ask_clicked = col_ask.button("🔍 Ask Agent", type="primary")

    if ask_clicked and user_query:
        filter_status = f" restricted to '{paper_filter_param}'" if paper_filter_param else " across all documents"
        with st.spinner(f"Analyzing query, selecting tools, retrieving evidence{filter_status}, and synthesizing answer..."):
            response: ResearchResponse = answer_query(user_query, agent=agent, paper_filter=paper_filter_param)

        st.markdown("---")

        # 1. Answer Display
        st.subheader("📝 Answer")
        if response.validation_status in ("insufficient_evidence", "quota_exceeded"):
            st.warning(f"⚠️ {response.answer}")
        elif response.validation_status == "error":
            st.error(f"❌ {response.answer}")
        else:
            st.info(response.answer)

        # 2. Citations Display
        if response.citations:
            st.subheader("📌 Verified Citations & Provenance")
            cit_cols = st.columns(min(len(response.citations), 3))
            for idx, cit in enumerate(response.citations):
                col = cit_cols[idx % len(cit_cols)]
                with col:
                    st.markdown(
                        f"**[{cit.get('evidence_id')}]** `{cit.get('source_filename')}`\n\n"
                        f"• **Page:** {cit.get('page_number')}\n\n"
                        f"• **Type:** `{cit.get('evidence_type')}`\n\n"
                        f"• **Citation:** {cit.get('formatted_citation')}"
                    )

        # 3. Multimodal Supporting Evidence (Grouped by Source Paper)
        if response.evidence:
            st.markdown("---")
            st.subheader("📚 Supporting Multimodal Evidence")

            # Group evidence by source paper to preserve explicit provenance
            paper_groups: dict = {}
            for item in response.evidence:
                source = item.get("source_filename", "Unknown Document")
                paper_groups.setdefault(source, []).append(item)

            for paper_name, items in paper_groups.items():
                with st.expander(f"📄 Paper: {paper_name} ({len(items)} Evidence Items)", expanded=True):
                    for item in items:
                        ev_id = item.get("evidence_id")
                        ev_type = item.get("evidence_type")
                        page_num = item.get("page_number")
                        score = item.get("score", 0.0)

                        st.markdown(f"### Evidence **[{ev_id}]** (`{ev_type.upper()}` - Page {page_num}, Similarity Score: {score:.4f})")

                        if ev_type == "text":
                            st.markdown(f"> {item.get('text_snippet')}")

                        elif ev_type == "table":
                            md_text = item.get("text_snippet", "")
                            df = parse_markdown_table_to_dataframe(md_text)
                            if df is not None:
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.markdown(md_text)

                        elif ev_type == "figure":
                            img_path = item.get("image_path")
                            caption = item.get("caption") or "No caption extracted."
                            st.markdown(f"**Caption:** {caption}")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=f"Extracted Figure [{ev_id}]", use_column_width=False, width=450)
                            else:
                                st.info(f"Figure Image Path: `{img_path}`")

                        elif ev_type == "calculation":
                            st.markdown(f"**Result:** `{item.get('text_snippet')}`")

                        st.markdown("---")

        # 4. Agent Execution Details (Safe Metadata Only)
        with st.expander("⚙️ Agent Execution Details (Metadata Trace)", expanded=False):
            st.markdown(f"**Query Processed:** `{response.query}`")
            st.markdown(f"**Target Paper Filter:** `{paper_filter_param or 'None (All Papers)'}`")
            st.markdown(f"**Selected Tools:** `{response.selected_tools}`")
            st.markdown(f"**Validation Status:** `{response.validation_status}`")
            st.markdown(f"**Total Evidence Items:** `{len(response.evidence)}`")
            st.markdown(f"**Verified Citations:** `{len(response.citations)}`")
            if response.warnings:
                st.warning(f"Warnings: {response.warnings}")
            if response.errors:
                st.error(f"Errors: {response.errors}")


if __name__ == "__main__":
    main()
