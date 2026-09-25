"""
Scientific figure extraction module using PyMuPDF (fitz) with caption parsing and deterministic figure IDs.
"""

import os
import re
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ExtractedFigure:
    """Dataclass representing an extracted scientific figure object with provenance metadata."""
    figure_id: str
    doc_id: str
    source_filename: str
    page_number: int  # 1-indexed
    figure_index: int  # 0-indexed order on page
    image_path: str
    caption: Optional[str]
    extraction_method: str  # e.g., "pymupdf_cropped" or "pymupdf_extracted"
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    surrounding_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted figure to dictionary representation."""
        return {
            "figure_id": self.figure_id,
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "figure_index": self.figure_index,
            "image_path": self.image_path,
            "caption": self.caption,
            "extraction_method": self.extraction_method,
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
            "surrounding_text": self.surrounding_text,
        }


class PyMuPDFFigureExtractor:
    """Page-by-page scientific figure extractor using PyMuPDF spatial layout analysis."""

    def __init__(self, output_dir: str = "data/figures", min_dimension: int = 50, min_area: int = 3600):
        """
        Initialize figure extractor.

        Args:
            output_dir: Base directory to store cropped figure PNG images.
            min_dimension: Minimum width and height in points to filter small icons/logos.
            min_area: Minimum bounding box area in sq points (default 60x60 = 3600).
        """
        self.output_dir = output_dir
        self.min_dimension = min_dimension
        self.min_area = min_area
        # Regex to detect caption headers like "Figure 1:", "Fig. 2.", "Figure 3 -", "FIG. 1"
        self.caption_regex = re.compile(
            r"^(?:Figure|Fig\.?|FIG\.?)\s*(\d+[a-z]?)(?:\s*[:.\-–—]|\s+)",
            re.IGNORECASE,
        )

    def extract_figures(
        self,
        pdf_path: str,
        doc_id: str,
        dpi: int = 200,
    ) -> List[ExtractedFigure]:
        """
        Extract scientific figures page-by-page from a PDF document.

        Args:
            pdf_path: Filepath to PDF paper.
            doc_id: Deterministic document identifier.
            dpi: Rendering DPI for cropping figure regions.

        Returns:
            List of ExtractedFigure dataclass objects.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        source_filename = os.path.basename(pdf_path)
        doc_figures_dir = os.path.join(self.output_dir, doc_id)
        os.makedirs(doc_figures_dir, exist_ok=True)

        extracted_figures: List[ExtractedFigure] = []
        doc = fitz.open(pdf_path)

        try:
            for page_idx in range(len(doc)):
                page_num = page_idx + 1  # 1-indexed page number
                page = doc[page_idx]

                # 1. Gather text blocks and candidate caption blocks on page
                blocks = page.get_text("blocks")
                caption_blocks: List[Tuple[fitz.Rect, str]] = []
                other_text_blocks: List[Tuple[fitz.Rect, str]] = []

                for b in blocks:
                    b_rect = fitz.Rect(b[:4])
                    b_text = b[4].strip()
                    if not b_text:
                        continue
                    if self.caption_regex.search(b_text):
                        caption_blocks.append((b_rect, b_text))
                    else:
                        other_text_blocks.append((b_rect, b_text))

                # 2. Gather image bounding boxes on page
                image_rects: List[fitz.Rect] = []

                # Method A: Direct embedded images
                images = page.get_images()
                for img_info in images:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    for r in rects:
                        if r.width >= self.min_dimension and r.height >= self.min_dimension and (r.width * r.height) >= self.min_area:
                            # Avoid duplicate rects
                            if not any(r.intersects(existing) for existing in image_rects):
                                image_rects.append(r)

                # Method B: If caption exists but no direct image rect found, use drawing rects or region above caption
                if not image_rects and caption_blocks:
                    # Look for drawing shapes or large non-text gaps above caption
                    drawings = page.get_drawings()
                    if drawings:
                        # Group drawings into a combined bounding box
                        bbox_list = [fitz.Rect(d["rect"]) for d in drawings if fitz.Rect(d["rect"]).width > 30]
                        if bbox_list:
                            union_rect = bbox_list[0]
                            for br in bbox_list[1:]:
                                union_rect |= br
                            if union_rect.width >= self.min_dimension and union_rect.height >= self.min_dimension:
                                image_rects.append(union_rect)

                # Sort image rects vertically (top to bottom)
                image_rects.sort(key=lambda r: (r.y0, r.x0))

                # 3. Process each figure candidate on the page
                for fig_idx, rect in enumerate(image_rects):
                    # Deterministic figure ID
                    figure_id = f"{doc_id}_p{page_num:03d}_f{fig_idx:02d}"
                    image_filename = f"{figure_id}.png"
                    image_path = os.path.join(doc_figures_dir, image_filename)

                    # Crop & render page region pixmap to PNG file
                    pix = page.get_pixmap(clip=rect, dpi=dpi)
                    pix.save(image_path)

                    # 4. Associate closest caption (spatially closest)
                    matched_caption: Optional[str] = None
                    min_dist = float("inf")

                    for c_rect, c_text in caption_blocks:
                        # Distance between image bottom/top and caption top/bottom
                        dist = min(
                            abs(c_rect.y0 - rect.y1),  # Caption below image
                            abs(rect.y0 - c_rect.y1),  # Caption above image
                        )
                        if dist < min_dist and dist < 250:  # Within 250pt proximity
                            min_dist = dist
                            matched_caption = c_text

                    # 5. Extract surrounding text context
                    surrounding_lines: List[str] = []
                    for t_rect, t_text in other_text_blocks:
                        if abs(t_rect.y0 - rect.y0) < 300:
                            surrounding_lines.append(t_text)

                    surrounding_text = " ".join(surrounding_lines[:3]) if surrounding_lines else None

                    fig_obj = ExtractedFigure(
                        figure_id=figure_id,
                        doc_id=doc_id,
                        source_filename=source_filename,
                        page_number=page_num,
                        figure_index=fig_idx,
                        image_path=image_path,
                        caption=matched_caption,
                        extraction_method="pymupdf_cropped",
                        bounding_box=(rect.x0, rect.y0, rect.x1, rect.y1),
                        surrounding_text=surrounding_text,
                    )
                    extracted_figures.append(fig_obj)

        finally:
            doc.close()

        return extracted_figures
