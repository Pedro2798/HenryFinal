"""Agent 2 - Change Extraction.

Role: a *Compliance Change Auditor*. It receives the structure map produced by
Agent 1 (the explicit handoff) plus both verbatim texts, and its ONLY job is to
isolate, classify and describe every change the amendment introduces.

Output is a strictly-validated :class:`ContractChangeOutput` (the 3 mandatory
production fields), produced via OpenAI structured outputs so the JSON schema
itself constrains the model.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models import ContractChangeOutput, ContractStructureMap
from .base import AgentError

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a Compliance Change Auditor at LegalMove. You are \
the second analyst in a two-person review. A Senior Legal Analyst has already \
produced a COMPARATIVE STRUCTURE MAP of the two documents; rely on it to \
navigate and do not re-derive structure.

Your SOLE task: identify, isolate and describe EVERY substantive change the \
AMENDMENT applies to the ORIGINAL contract.

CRITICAL - how to read an amendment:
- The AMENDMENT is an instrument that edits the ORIGINAL. Report changes in \
terms of the ORIGINAL contract's clauses ONLY.
- The amendment's own instruction headings ("Amendment to Clause 3 (Fees)", \
"Modification of Clause 4", "Deletion in Clause 3", "New Clause 7"), its \
preamble/recitals, and its boilerplate ("No Other Changes"; "Except as \
expressly amended herein, ... remains in full force and effect") are \
SCAFFOLDING. They are NEVER changes. NEVER output them as additions and \
NEVER put them in sections_changed.
- 'sections_changed' must contain ORIGINAL contract clause identifiers \
affected (e.g. "Clause 3 - Fees"), never the amendment's own section \
numbers or titles.

Classify each change as EXACTLY one of:
- ADDITION: a genuinely new substantive clause/obligation the amendment \
inserts INTO the contract (e.g. a brand-new "Data Protection" clause). \
Amendment boilerplate is NOT an addition.
- DELETION: existing contract content the amendment removes, revokes, strikes \
or "deletes in its entirety". A removal is ALWAYS a DELETION - never label a \
removal as a MODIFICATION with an "(deleted)" after-value.
- MODIFICATION: existing content whose value, scope or wording is altered \
(amounts, dates, territory, parties...). Capture concrete BEFORE and AFTER \
values.

Hard rules:
- Report ONLY changes evidenced by the text. Never invent or infer changes \
that are not explicitly supported - a hallucinated change is a critical \
compliance failure.
- 'topics_touched' = legal/commercial categories (e.g. Pricing, Term & \
Termination, Territorial Scope, Confidentiality, Data Protection, \
Liability), not section numbers.
- 'summary_of_the_change' must be audit-grade: for each change name its type \
(ADDITION/DELETION/MODIFICATION) and quote the before/after values so a \
reviewer can act without opening the source files.

Return strictly the required structured object - nothing else."""

_HUMAN_PROMPT = """COMPARATIVE STRUCTURE MAP (from the Senior Legal Analyst - \
use it to navigate, do not contradict it):
\"\"\"
{structure_map}
\"\"\"

ORIGINAL CONTRACT (verbatim):
\"\"\"
{original_text}
\"\"\"

AMENDMENT / ADDENDUM (verbatim):
\"\"\"
{amendment_text}
\"\"\"

Extract every change the amendment applies to the ORIGINAL contract and \
return the structured output."""


class ExtractionAgent:
    """Produces the final :class:`ContractChangeOutput` from the handoff."""

    name = "extraction_agent"

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._structured_llm = llm.with_structured_output(
            ContractChangeOutput, method="json_schema", strict=True
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
        )
        self._chain = self._prompt | self._structured_llm

    def extract(
        self,
        structure_map: ContractStructureMap,
        original_text: str,
        amendment_text: str,
        *,
        callbacks: list | None = None,
    ) -> ContractChangeOutput:
        """Extract the changes, consuming Agent 1's structure map.

        The map is serialized to JSON and injected into the prompt so the
        handoff is explicit and fully visible in the Langfuse trace.
        """

        logger.info("ExtractionAgent: extracting changes from the amendment.")
        try:
            result = self._chain.invoke(
                {
                    "structure_map": structure_map.model_dump_json(indent=2),
                    "original_text": original_text,
                    "amendment_text": amendment_text,
                },
                config={"callbacks": callbacks or []},
            )
        except Exception as exc:
            raise AgentError(
                f"ExtractionAgent failed to extract changes: {exc}"
            ) from exc

        if not isinstance(result, ContractChangeOutput):
            raise AgentError(
                f"ExtractionAgent returned an unexpected type: {type(result)!r}"
            )
        logger.info(
            "ExtractionAgent: %d section(s) changed, %d topic(s) touched.",
            len(result.sections_changed),
            len(result.topics_touched),
        )
        return result
