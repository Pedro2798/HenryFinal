"""Pydantic data contracts for the contract-comparison pipeline.

These models are the single source of truth for every structured handoff in
the system:

* ``ParsedDocument``        - output of the GPT-4o Vision parsing step.
* ``SectionMapping`` /
  ``ContractStructureMap``  - output of Agent 1 (ContextualizationAgent).
* ``ContractChangeOutput``  - FINAL, strictly-validated output of Agent 2
                              (ExtractionAgent). This is the production
                              contract consumed by downstream LegalMove
                              systems and is exactly the schema required by
                              the assignment (3 mandatory fields).

Strictness rationale: every model sets ``extra="forbid"`` so an LLM that
hallucinates an extra key fails loudly at validation time instead of leaking
malformed data into production.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Vision parsing
# ---------------------------------------------------------------------------
class ParsedDocument(BaseModel):
    """Faithful text extraction of a single contract image."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="Human role of the document, e.g. 'original' or 'amendment'."
    )
    source_path: str = Field(description="Absolute path of the parsed image file.")
    text: str = Field(
        description="Verbatim text extracted by GPT-4o Vision, preserving the "
        "clause/section hierarchy of the source document."
    )
    model: str = Field(description="OpenAI model id used for the extraction.")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Vision parsing returned empty text - the image is unreadable "
                "or the model produced no output."
            )
        return v.strip()


# ---------------------------------------------------------------------------
# Agent 1 - Contextualization (comparative structure map)
# ---------------------------------------------------------------------------
class SectionMapping(BaseModel):
    """How one logical section corresponds across the two documents."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(
        description="Stable identifier for the section, preferably the clause "
        "number and title as written in the contract "
        "(e.g. 'Clause 4 - Fees', 'Section 2 - Confidential Information')."
    )
    present_in_original: bool = Field(
        description="True if this section exists in the ORIGINAL contract."
    )
    present_in_amendment: bool = Field(
        description="True if this section exists in / is referenced by the AMENDMENT."
    )
    purpose: str = Field(
        description="One sentence describing what this section legally governs."
    )
    correspondence_notes: str = Field(
        description="How the original and amendment versions of this section "
        "relate. NOT a change list - just structural correspondence "
        "(aligned, renumbered, newly introduced, removed, unchanged)."
    )


class ContractStructureMap(BaseModel):
    """Agent 1 output: a comparative map, NOT a change report.

    This is the explicit handoff artifact consumed by Agent 2. Keeping it a
    typed model (instead of free text) makes the agent collaboration auditable
    in Langfuse and prevents Agent 2 from re-deriving structure it can trust.
    """

    model_config = ConfigDict(extra="forbid")

    document_overview: str = Field(
        description="Two or three sentences on the nature/purpose of the "
        "contract and what kind of instrument the amendment is."
    )
    sections: List[SectionMapping] = Field(
        description="Every logical section found in either document, aligned."
    )
    high_level_observations: str = Field(
        description="Structural observations that orient the extraction agent "
        "(e.g. 'amendment only references clauses 3 and 4', "
        "'amendment introduces a new clause 7'). No detailed diffs."
    )

    @field_validator("sections")
    @classmethod
    def _at_least_one_section(cls, v: List[SectionMapping]) -> List[SectionMapping]:
        if not v:
            raise ValueError(
                "The structure map must contain at least one section; an empty "
                "map means contextualization failed."
            )
        return v


# ---------------------------------------------------------------------------
# Agent 2 - Extraction (FINAL production output)
# ---------------------------------------------------------------------------
class ChangeType(str, Enum):
    """Taxonomy the extraction agent reasons over internally."""

    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"


class ContractChangeOutput(BaseModel):
    """FINAL strictly-validated output. Exactly the 3 required fields.

    Downstream LegalMove systems depend on this schema, so it is intentionally
    minimal and strict. Field descriptions are sent to the model as part of
    the JSON schema (structured outputs), which materially improves accuracy.
    """

    model_config = ConfigDict(extra="forbid")

    sections_changed: List[str] = Field(
        description="Identifiers of the contract sections/clauses modified by "
        "the amendment (e.g. ['Clause 3 - Term', 'Clause 4 - Fees']). "
        "Use the same identifiers as in the source documents."
    )
    topics_touched: List[str] = Field(
        description="Legal/commercial categories affected by the changes "
        "(e.g. ['Pricing', 'Term & Termination', 'Territorial Scope', "
        "'Data Protection', 'Confidentiality'])."
    )
    summary_of_the_change: str = Field(
        description="Precise, audit-grade narrative of every change. For each "
        "change state whether it is an ADDITION, DELETION or "
        "MODIFICATION and quote the concrete before/after values "
        "(amounts, dates, scope) so a compliance reviewer can act "
        "without opening the source documents."
    )

    @field_validator("sections_changed", "topics_touched")
    @classmethod
    def _clean_list(cls, v: List[str]) -> List[str]:
        cleaned = [item.strip() for item in v if item and item.strip()]
        # De-duplicate while preserving order.
        seen: set[str] = set()
        deduped: List[str] = []
        for item in cleaned:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    @field_validator("summary_of_the_change")
    @classmethod
    def _summary_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError(
                "summary_of_the_change is empty or too short to be a usable "
                "audit summary."
            )
        return v.strip()
