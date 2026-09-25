# Multimodal Research Intelligence Agent — Demo Questions & Verification Guide

This document contains canonical demo questions verified against the actual research papers (`convlstm_precipitation_nowcasting.pdf` and `canonical_sample.pdf`) indexed in `data/papers/`.

---

## 1. Text Retrieval & Narrative Question

- **User Question:** *"What main research problem does the ConvLSTM paper address?"*
- **Expected Tool / Modality:** `search_text` (Text chunk retrieval)
- **Source Paper:** `convlstm_precipitation_nowcasting.pdf`
- **Page Number:** Page 1
- **Retrieved Evidence:**
  - `[E1]` Section 1 Introduction: Precipitation nowcasting aims to predict future rainfall intensity in a local region over a short period (0–6 hours).
- **Verified Final Answer:**
  > The ConvLSTM paper addresses the problem of precipitation nowcasting—predicting the future rainfall intensity in a local region over a short period (e.g., 0–6 hours)—by modeling it as a spatiotemporal sequence forecasting problem where both inputs and targets are spatiotemporal sequences.
- **Citation Validity:** `[E1] convlstm_precipitation_nowcasting.pdf, Page 1 (valid)`

---

## 2. Table Understanding Question

- **User Question:** *"Which model achieves the lowest Rainfall-MSE score in Table 1?"*
- **Expected Tool / Modality:** `search_tables` (Table retrieval & structured dataframe parsing)
- **Source Paper:** `convlstm_precipitation_nowcasting.pdf`
- **Page Number:** Page 7
- **Retrieved Evidence:**
  - `[E1]` Table 1: Performance metric comparisons across FC-LSTM and ConvLSTM variants.
  - Extracted matrix values: `ConvLSTM (5x5): 0.089`, `ConvLSTM (3x3): 0.093`, `FC-LSTM: 0.140`.
- **Verified Final Answer:**
  > According to Table 1 in the ConvLSTM paper, the ConvLSTM model with a 5x5 kernel size achieves the lowest Rainfall-MSE score of 0.089, outperforming FC-LSTM (0.140) and ConvLSTM 3x3 (0.093).
- **Citation Validity:** `[E1] convlstm_precipitation_nowcasting.pdf, Table (p. 7) (valid)`

---

## 3. Figure Understanding & VLM Question

- **User Question:** *"What does Figure 1 illustrate regarding 2D image transformations over time?"*
- **Expected Tool / Modality:** `search_figures` (PyMuPDF figure cropper + Gemini 2.5 Flash VLM analysis)
- **Source Paper:** `convlstm_precipitation_nowcasting.pdf`
- **Page Number:** Page 2
- **Retrieved Evidence:**
  - `[E1]` Figure 1: Inner structure of ConvLSTM vs FC-LSTM cells and spatiotemporal sequence input/output flow.
- **Verified Final Answer:**
  > Figure 1 illustrates the spatiotemporal sequence forecasting formulation where 3D tensors representing 2D spatial radar images across time steps are processed sequentially, showing how ConvLSTM preserves spatial grid relationships through convolutional state transitions.
- **Citation Validity:** `[E1] convlstm_precipitation_nowcasting.pdf, Figure (p. 2) (valid)`

---

## 4. Cross-Paper Reasoning Question

- **User Question:** *"Compare the purpose of canonical_sample.pdf vs convlstm_precipitation_nowcasting.pdf."*
- **Expected Tool / Modality:** `search_text` with `search_diversified()` round-robin document sampling
- **Source Papers:** `canonical_sample.pdf` and `convlstm_precipitation_nowcasting.pdf`
- **Page Numbers:** `canonical_sample.pdf` (Page 1) vs `convlstm_precipitation_nowcasting.pdf` (Page 1)
- **Retrieved Evidence:**
  - `[E1]` `canonical_sample.pdf`: Synthetic benchmark paper demonstrating multimodal RAG ingestion.
  - `[E2]` `convlstm_precipitation_nowcasting.pdf`: Core research paper introducing convolutional LSTM networks for precipitation nowcasting.
- **Verified Final Answer:**
  > `convlstm_precipitation_nowcasting.pdf` introduces the ConvLSTM architecture for short-term spatiotemporal precipitation nowcasting using radar data, whereas `canonical_sample.pdf` serves as a canonical synthetic benchmark document for testing multimodal PDF text, table, and figure ingestion.
- **Citation Validity:** `[E1] canonical_sample.pdf, Page 1`, `[E2] convlstm_precipitation_nowcasting.pdf, Page 1 (valid)`

---

## 5. Numerical / Calculation Reasoning Question

- **User Question:** *"What is the percentage improvement from Rainfall-MSE 0.120 (FC-LSTM) to 0.089 (ConvLSTM 5x5)?"*
- **Expected Tools / Modalities:** `search_tables` + `calculate` (AST SafeCalculator tool)
- **Source Paper:** `convlstm_precipitation_nowcasting.pdf`
- **Page Number:** Page 7
- **Calculation Expression:** `(0.120 - 0.089) / 0.120 * 100`
- **Retrieved Evidence:**
  - `[E1]` Table 1 metrics: FC-LSTM MSE 0.120, ConvLSTM 5x5 MSE 0.089.
  - `[E2]` Calculation output: `25.8333%` -> `25.83%`.
- **Verified Final Answer:**
  > The reduction from Rainfall-MSE 0.120 (FC-LSTM) to 0.089 (ConvLSTM 5x5) represents a 25.83% percentage improvement in prediction error.
- **Citation Validity:** `[E1] convlstm_precipitation_nowcasting.pdf, Table (p. 7)`, `[E2] Calculation Engine (valid)`

---

## 6. Insufficient Evidence / Guardrail Question

- **User Question:** *"What color is the author's cat in the ConvLSTM paper?"*
- **Expected Tool / Modality:** `search_text`
- **Source Papers:** None
- **Retrieved Evidence:** 0 matching evidence items above relevance threshold.
- **Verified Final Answer:**
  > I don't have enough evidence in the provided research papers to answer this question.
- **Validation Status:** `insufficient_evidence`
- **Citation Validity:** No hallucinated citations emitted.
