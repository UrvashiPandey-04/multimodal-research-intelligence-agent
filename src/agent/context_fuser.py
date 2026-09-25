"""
Multimodal Context Fusion module structuring text, table, figure, and calculation evidence into unified prompts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.citations.evidence_registry import EvidenceRegistry, EvidenceRegistryItem


@dataclass
class FusedContext:
    """Dataclass representing structured multimodal context ready for LLM answer generation."""
    evidence_blocks: List[str]
    fused_text: str
    evidence_count: int
    has_text: bool = False
    has_table: bool = False
    has_figure: bool = False
    has_calculation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert fused context to dictionary representation."""
        return {
            "fused_text": self.fused_text,
            "evidence_count": self.evidence_count,
            "has_text": self.has_text,
            "has_table": self.has_table,
            "has_figure": self.has_figure,
            "has_calculation": self.has_calculation,
        }


class MultimodalContextFuser:
    """Fuses multi-modal evidence items into structured, grounded LLM prompt context."""

    def __init__(self):
        """Initialize context fuser."""
        pass

    def fuse_context(self, registry: EvidenceRegistry) -> FusedContext:
        """
        Fuse all registered evidence items from EvidenceRegistry into a structured context string.

        Args:
            registry: Authoritative request EvidenceRegistry instance.

        Returns:
            FusedContext object containing modality-aware evidence blocks.
        """
        items = registry.get_all_items()
        if not items:
            return FusedContext(
                evidence_blocks=[],
                fused_text="No evidence retrieved.",
                evidence_count=0,
            )

        blocks: List[str] = []
        has_text = False
        has_table = False
        has_figure = False
        has_calc = False

        for item in items:
            eid = item.evidence_id
            etype = item.evidence_type.upper()

            if item.evidence_type == "text":
                has_text = True
                block = (
                    f"[{eid} TEXT EVIDENCE]\n"
                    f"Paper: {item.source_filename}\n"
                    f"Page: {item.page_number}\n"
                    f"Chunk ID: {item.chunk_id}\n"
                    f"Content:\n{item.text_snippet.strip()}\n"
                )
            elif item.evidence_type == "table":
                has_table = True
                block = (
                    f"[{eid} TABLE EVIDENCE]\n"
                    f"Paper: {item.source_filename}\n"
                    f"Page: {item.page_number}\n"
                    f"Table ID: {item.chunk_id}\n"
                    f"Markdown Table:\n{item.text_snippet.strip()}\n"
                )
            elif item.evidence_type == "figure":
                has_figure = True
                block = (
                    f"[{eid} FIGURE EVIDENCE]\n"
                    f"Paper: {item.source_filename}\n"
                    f"Page: {item.page_number}\n"
                    f"Figure ID: {item.chunk_id}\n"
                    f"Caption: {item.caption or 'None'}\n"
                    f"Visual Details:\n{item.text_snippet.strip()}\n"
                )
            elif item.evidence_type == "calculation":
                has_calc = True
                block = (
                    f"[{eid} CALCULATION EVIDENCE]\n"
                    f"Source: Safe Calculator Tool\n"
                    f"Details: {item.text_snippet.strip()}\n"
                )
            else:
                block = f"[{eid} EVIDENCE]\n{item.text_snippet.strip()}\n"

            blocks.append(block)

        fused_str = "\n" + "=" * 50 + "\n"
        fused_str += "\n\n".join(blocks)
        fused_str += "\n" + "=" * 50

        return FusedContext(
            evidence_blocks=blocks,
            fused_text=fused_str,
            evidence_count=len(items),
            has_text=has_text,
            has_table=has_table,
            has_figure=has_figure,
            has_calculation=has_calc,
        )
