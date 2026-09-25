"""
Streamlit Web Interface for Multimodal Research Intelligence Agent.
Provides an interactive research dashboard for evidence-grounded scientific QA over PDFs.
"""

import os
import sys
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

# Page configuration
st.set_page_config(
    page_title="Multimodal Research Intelligence Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 1. Resource Caching: Index papers once per application runtime
@st.cache_resource(show_spinner="Indexing canonical research papers into vector store...")
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
    # Sidebar
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.markdown("---")

        st.subheader("📚 Available Research Papers")
        papers_dir = os.path.join(project_root, "data", "papers")
        if os.path.exists(papers_dir):
            pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
            for pdf in pdf_files:
                st.markdown(f"• `{pdf}`")
        else:
            st.warning("Papers directory not found.")

        st.markdown("---")
        st.subheader("🛠️ Supported Modalities")
        st.markdown("- 📄 **Text Chunks:** BGE Dense Vector Search")
        st.markdown("- 📊 **Tables:** pdfplumber Structured Markdown & DataFrames")
        st.markdown("- 🖼️ **Figures:** PyMuPDF Cropped PNGs + Gemini 2.5 VLM")
        st.markdown("- 🧮 **Calculations:** AST SafeCalculator Engine")

        st.markdown("---")
        st.subheader("⚙️ System Status")
        cfg = get_config_summary()
        api_status = "✅ Configured" if cfg["gemini_api_key_configured"] else "⚠️ Unconfigured (Fallback Mode)"
        st.markdown(f"**Gemini API:** {api_status}")
        st.markdown(f"**LLM Model:** `{cfg['gemini_model']}`")
        st.markdown(f"**Embedding Model:** `{cfg['embedding_model']}`")
        st.markdown(f"**OCR Status:** `{'Enabled' if cfg['ocr_enabled'] else 'Disabled'}`")

    # Main Interface Header
    st.title("🔬 Multimodal Research Intelligence Agent")
    st.markdown(
        "An evidence-grounded AI research assistant that parses, retrieves, reasons, and cites complex multimodal scientific research papers (PDFs)."
    )

    # Initialize Agent
    try:
        agent = get_cached_agent()
    except Exception as e:
        st.error(f"Failed to initialize Research Agent: {e}")
        st.stop()

    st.markdown("---")

    # Quick Sample Queries for Interview Demo
    st.subheader("💡 Try Sample Demo Queries")
    sample_queries = [
        "Select a sample question...",
        "What main research problem does the ConvLSTM paper address?",
        "Which model achieves the lowest Rainfall-MSE score in Table 1?",
        "What does Figure 1 illustrate regarding 2D image transformations over time?",
        "Compare the purpose of canonical_sample.pdf vs convlstm_precipitation_nowcasting.pdf.",
        "What is the percentage improvement from Rainfall-MSE 0.120 (FC-LSTM) to 0.089 (ConvLSTM 5x5)?",
        "What color is the author's cat in the ConvLSTM paper?",
    ]

    selected_sample = st.selectbox("Choose a preset query to run:", sample_queries, index=0)

    # Question Input
    default_text = selected_sample if selected_sample != "Select a sample question..." else ""
    user_query = st.text_input("Ask a research question about the papers:", value=default_text, placeholder="e.g., How does ConvLSTM differ from FC-LSTM?")

    col_ask, col_clear = st.columns([1, 5])
    ask_clicked = col_ask.button("🔍 Ask Agent", type="primary")

    if ask_clicked and user_query:
        with st.spinner("Analyzing query, selecting tools, retrieving multimodal evidence, and synthesizing answer..."):
            response: ResearchResponse = answer_query(user_query, agent=agent)

        st.markdown("---")

        # 1. Answer Display
        st.subheader("📝 Answer")
        if response.validation_status == "insufficient_evidence":
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
