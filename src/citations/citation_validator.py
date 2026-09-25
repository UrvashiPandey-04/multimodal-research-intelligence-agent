"""
Deterministic Citation Validator component enforcing authoritative evidence provenance.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from src.citations.evidence_registry import EvidenceRegistry, EvidenceRegistryItem
from src.citations.citation_model import Citation, ValidationResult


class CitationValidator:
    """Deterministic validator verifying that LLM-emitted citation references resolve to authentic retrieved chunks."""

    def __init__(self):
        # Regex to extract inline evidence tags like [E1], [E2], [1], [2]
        self._tag_pattern = re.compile(r"\[(E\d+|\d+)\]", re.IGNORECASE)

    def extract_evidence_ids(self, text: str) -> List[str]:
        """
        Extract inline evidence tag references from generated text.

        Args:
            text: LLM response string.

        Returns:
            List of raw evidence ID strings in order of appearance.
        """
        if not text:
            return []

        matches = self._tag_pattern.findall(text)
        extracted: List[str] = []

        for m in matches:
            clean = m.strip().upper()
            if clean.isdigit():
                clean = f"E{clean}"
            extracted.append(clean)

        return extracted

    def validate_citations(
        self,
        answer_text: str,
        registry: EvidenceRegistry,
        explicit_evidence_ids: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate citation references against the authoritative request EvidenceRegistry.

        Args:
            answer_text: LLM generated answer text.
            registry: Authoritative EvidenceRegistry instance for the current request.
            explicit_evidence_ids: Optional explicit list of evidence IDs to validate.

        Returns:
            ValidationResult object detailing valid citations, invalid citations, and status.
        """
        # 1. Gather evidence IDs from text regex or explicit list
        raw_ids = explicit_evidence_ids or self.extract_evidence_ids(answer_text)

        if not raw_ids:
            return ValidationResult(
                citations=[],
                invalid_citations=[],
                validation_status="none",
                validation_messages=["No evidence citations referenced in response."],
            )

        # 2. Deterministic deduplication preserving first-appearance order
        seen_ids: Set[str] = set()
        deduped_ids: List[str] = []

        for eid in raw_ids:
            clean_id = eid.strip().upper()
            if clean_id.isdigit():
                clean_id = f"E{clean_id}"

            if clean_id not in seen_ids:
                seen_ids.add(clean_id)
                deduped_ids.append(clean_id)

        valid_citations: List[Citation] = []
        invalid_citations: List[Citation] = []
        messages: List[str] = []

        # 3. Validate each evidence ID against EvidenceRegistry
        for idx, eid in enumerate(deduped_ids, start=1):
            cit_id = f"CIT_{idx}"
            reg_item = registry.get_item(eid)

            if reg_item is None:
                # Rule A & B Failure: Unknown ID or from another request
                reason = f"Unknown evidence ID '{eid}': Not present in current request EvidenceRegistry."
                invalid_cit = Citation(
                    citation_id=cit_id,
                    evidence_id=eid,
                    document_id="",
                    source_filename="",
                    page_number=0,
                    chunk_id="",
                    extraction_method="unknown",
                    validation_status="invalid",
                    validation_reason=reason,
                    formatted_citation=f"[{eid}] UNKNOWN SOURCE",
                )
                invalid_citations.append(invalid_cit)
                messages.append(reason)
                continue

            # Rule C & D Failure: Missing critical metadata
            if not reg_item.is_metadata_valid():
                reasons = []
                if not reg_item.doc_id:
                    reasons.append("missing document_id")
                if not reg_item.source_filename:
                    reasons.append("missing source_filename")
                if not reg_item.chunk_id:
                    reasons.append("missing chunk_id")
                if reg_item.page_number <= 0:
                    reasons.append(f"invalid page_number ({reg_item.page_number})")

                reason_str = f"Evidence ID '{eid}' failed metadata validation: " + ", ".join(reasons)
                invalid_cit = Citation(
                    citation_id=cit_id,
                    evidence_id=eid,
                    document_id=reg_item.doc_id,
                    source_filename=reg_item.source_filename,
                    page_number=reg_item.page_number,
                    chunk_id=reg_item.chunk_id,
                    extraction_method=reg_item.extraction_method,
                    validation_status="invalid",
                    validation_reason=reason_str,
                    formatted_citation=f"[{eid}] INVALID METADATA",
                )
                invalid_citations.append(invalid_cit)
                messages.append(reason_str)
                continue

            # Rule E: Valid Citation - Authoritative metadata comes from EvidenceRegistry
            if reg_item.evidence_type == "table":
                formatted_str = f"[{reg_item.evidence_id}] {reg_item.source_filename}, Table (p. {reg_item.page_number})"
            elif reg_item.evidence_type == "figure":
                formatted_str = f"[{reg_item.evidence_id}] {reg_item.source_filename}, Figure (p. {reg_item.page_number})"
            elif reg_item.evidence_type == "calculation":
                formatted_str = f"[{reg_item.evidence_id}] {reg_item.text_snippet}"
            else:
                formatted_str = f"[{reg_item.evidence_id}] {reg_item.source_filename}, p. {reg_item.page_number}"
            valid_cit = Citation(
                citation_id=cit_id,
                evidence_id=reg_item.evidence_id,
                document_id=reg_item.doc_id,
                source_filename=reg_item.source_filename,
                page_number=reg_item.page_number,
                chunk_id=reg_item.chunk_id,
                extraction_method=reg_item.extraction_method,
                validation_status="valid",
                validation_reason="Successfully validated against EvidenceRegistry provenance metadata.",
                formatted_citation=formatted_str,
            )
            valid_citations.append(valid_cit)

        # 4. Determine overall validation status
        if valid_citations and not invalid_citations:
            status = "valid"
            messages.append(f"All {len(valid_citations)} citation references successfully validated.")
        elif valid_citations and invalid_citations:
            status = "partial"
            messages.append(f"Partial validation: {len(valid_citations)} valid, {len(invalid_citations)} invalid.")
        else:
            status = "failed"
            messages.append(f"Validation failed: All {len(invalid_citations)} citation references were invalid.")

        return ValidationResult(
            citations=valid_citations,
            invalid_citations=invalid_citations,
            validation_status=status,
            validation_messages=messages,
        )
