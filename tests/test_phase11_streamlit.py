"""
Unit tests for Phase 11 Streamlit frontend helper functions and table parsing.
"""

import pandas as pd
from app.streamlit_app import parse_markdown_table_to_dataframe


def test_parse_markdown_table_to_dataframe_valid():
    """Test converting valid Markdown table string to pandas DataFrame."""
    md_text = (
        "| Model | Rainfall-MSE | CSI |\n"
        "|---|---|---|\n"
        "| ConvLSTM | 0.089 | 0.78 |\n"
        "| FC-LSTM | 0.140 | 0.65 |"
    )

    df = parse_markdown_table_to_dataframe(md_text)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Model", "Rainfall-MSE", "CSI"]
    assert len(df) == 2
    assert df.iloc[0]["Model"] == "ConvLSTM"
    assert df.iloc[0]["Rainfall-MSE"] == "0.089"


def test_parse_markdown_table_to_dataframe_invalid():
    """Test that invalid or plain text strings return None without raising exceptions."""
    assert parse_markdown_table_to_dataframe("") is None
    assert parse_markdown_table_to_dataframe("Plain text without tables") is None
    assert parse_markdown_table_to_dataframe("| Header Only |") is None
