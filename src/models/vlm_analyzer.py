"""
Gemini 2.5 Flash VLM Figure Analyzer module with structured output parsing and caching.
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from src.models.llm import GeminiLLM, MissingAPIKeyError, GeminiAPIError
from src.models.vlm_cache import VLMCache
from src.ingestion.figure_parser import ExtractedFigure


@dataclass
class FigureAnalysisResult:
    """Structured analysis output from VLM visual inspection."""
    figure_summary: str
    figure_type: str
    observations: List[str] = field(default_factory=list)
    visible_values: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary format."""
        return {
            "figure_summary": self.figure_summary,
            "figure_type": self.figure_type,
            "observations": self.observations,
            "visible_values": self.visible_values,
            "uncertainties": self.uncertainties,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FigureAnalysisResult":
        """Instantiate FigureAnalysisResult from dictionary representation."""
        return cls(
            figure_summary=data.get("figure_summary", "No summary available."),
            figure_type=data.get("figure_type", "unknown"),
            observations=data.get("observations", []),
            visible_values=data.get("visible_values", []),
            uncertainties=data.get("uncertainties", []),
            raw_response=data.get("raw_response"),
        )


FIGURE_ANALYSIS_SYSTEM_INSTRUCTION = """
You are an expert scientific figure analyzer. Your job is to analyze scientific figures extracted from research paper PDFs.

When analyzing a figure:
1. Distinguish between:
   - (A) Directly visible visual observations in the image (plots, lines, shapes, labels, architecture blocks).
   - (B) Information provided by the caption or surrounding context.
   - (C) Uncertain or unreadable elements (due to image resolution or blur).
2. DO NOT invent numerical values, legend keys, or axis labels that are not clearly visible.
3. Explicitly report any uncertainties or unreadable parts under 'uncertainties'.

Return your output strictly as a JSON object with the following fields:
{
  "figure_summary": "A concise 2-3 sentence overview of what the figure illustrates.",
  "figure_type": "One of: architecture diagram, flowchart, line plot, bar chart, scatter plot, heatmap, map, visual output comparison, structural schematic, or unknown",
  "observations": ["List of key visual observations visible in the image"],
  "visible_values": ["List of clearly visible numerical values, labels, or metric scores"],
  "uncertainties": ["List of unreadable, blurred, or ambiguous details"]
}
"""


class VLMFigureAnalyzer:
    """Gemini 2.5 Flash VLM analyzer with caching and fallback parsing."""

    def __init__(self, llm_client: Optional[GeminiLLM] = None, cache: Optional[VLMCache] = None):
        """
        Initialize VLM figure analyzer.

        Args:
            llm_client: GeminiLLM wrapper instance.
            cache: VLMCache instance.
        """
        self.llm = llm_client or GeminiLLM()
        self.cache = cache or VLMCache()

    def analyze_figure(
        self,
        figure: ExtractedFigure,
        force_refresh: bool = False,
    ) -> FigureAnalysisResult:
        """
        Analyze a scientific figure using Gemini 2.5 Flash VLM.

        Args:
            figure: ExtractedFigure object.
            force_refresh: If True, bypass cache and re-query Gemini API.

        Returns:
            FigureAnalysisResult structured object.
        """
        cache_key = self.cache.make_key(figure.doc_id, figure.figure_id)

        # 1. Check Cache
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return FigureAnalysisResult.from_dict(cached_data)

        # 2. Check API Client Configuration
        if not self.llm.is_configured():
            # Fallback analysis when API key is unconfigured
            fallback_res = FigureAnalysisResult(
                figure_summary=f"Scientific figure from {figure.source_filename} (p. {figure.page_number}). Caption: {figure.caption or 'None'}.",
                figure_type="scientific_diagram",
                observations=[f"Extracted image located at {figure.image_path}."],
                visible_values=[],
                uncertainties=["VLM analysis unavailable: GEMINI_API_KEY is not configured."],
                raw_response="Fallback mock response.",
            )
            return fallback_res

        # 3. Read image file bytes
        if not os.path.exists(figure.image_path):
            raise FileNotFoundError(f"Figure image file not found: {figure.image_path}")

        with open(figure.image_path, "rb") as img_file:
            image_bytes = img_file.read()

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        # 4. Construct prompt text
        prompt_lines = [
            "Please analyze this scientific figure image.",
            f"Paper Filename: {figure.source_filename}",
            f"Page Number: {figure.page_number}",
        ]
        if figure.caption:
            prompt_lines.append(f"Caption: {figure.caption}")
        if figure.surrounding_text:
            prompt_lines.append(f"Surrounding Text Context: {figure.surrounding_text}")

        prompt_lines.append("\nReturn ONLY valid JSON according to the specified structure.")
        user_prompt = "\n".join(prompt_lines)

        # 5. Call Gemini 2.5 Flash VLM API
        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=FIGURE_ANALYSIS_SYSTEM_INSTRUCTION,
            )

            response = self.llm.client.models.generate_content(
                model=self.llm.model_name,
                contents=[image_part, user_prompt],
                config=config,
            )

            raw_text = response.text or ""
            analysis_res = self._parse_json_response(raw_text, figure)

            # Store in cache
            self.cache.set(cache_key, analysis_res.to_dict())
            return analysis_res

        except Exception as e:
            fallback = FigureAnalysisResult(
                figure_summary=f"Figure analysis from {figure.source_filename} (p. {figure.page_number}). Caption: {figure.caption or 'None'}.",
                figure_type="unknown",
                observations=[],
                visible_values=[],
                uncertainties=[f"VLM invocation failed: {e}"],
                raw_response=str(e),
            )
            return fallback

    def _parse_json_response(self, raw_text: str, figure: ExtractedFigure) -> FigureAnalysisResult:
        """
        Parse raw model text into structured FigureAnalysisResult safely.

        Args:
            raw_text: Gemini response text.
            figure: ExtractedFigure context.

        Returns:
            FigureAnalysisResult instance.
        """
        if not raw_text:
            return FigureAnalysisResult(
                figure_summary=f"Empty VLM response for figure {figure.figure_id}.",
                figure_type="unknown",
                uncertainties=["Empty response received from VLM."],
            )

        clean_text = raw_text.strip()
        # Remove ```json ... ``` code fence if present
        if "```" in clean_text:
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r"```\s*$", "", clean_text, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(clean_text)
            return FigureAnalysisResult(
                figure_summary=parsed.get("figure_summary", f"Figure {figure.figure_id}."),
                figure_type=parsed.get("figure_type", "unknown"),
                observations=parsed.get("observations", []),
                visible_values=parsed.get("visible_values", []),
                uncertainties=parsed.get("uncertainties", []),
                raw_response=raw_text,
            )
        except Exception:
            # Fallback if model output is not valid JSON
            return FigureAnalysisResult(
                figure_summary=raw_text[:300],
                figure_type="unknown",
                observations=[],
                visible_values=[],
                uncertainties=["Model output was not valid JSON."],
                raw_response=raw_text,
            )
