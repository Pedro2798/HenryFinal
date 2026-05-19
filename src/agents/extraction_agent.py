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
produced a COMPARATIVE STRUCTURE MAP of the two documents; you must rely on \
that map to navigate the documents efficiently and not waste effort \
re-deriving structure.

Your SOLE task: identify, isolate and describe EVERY substantive change the \
AMENDMENT introduces relative to the ORIGINAL contract.

Method:
1. Use the structure map to locate corresponding sections.
2. For every section, classify each change as exactly one of:
   - ADDITION  : content/obligations the amendment adds (e.g. a brand-new \
clause).
   - DELETION  : content/obligations the amendment removes or revokes.
   - MODIFICATION : content present in both but altered (amounts, dates, \
scope, parties...).
3. For modifications, capture the concrete BEFORE and AFTER values.

Hard rules:
- Report ONLY changes that are evidenced by the actual text. If something is \
unchanged, do not mention it. Never invent or infer changes that are not \
explicitly supported - a hallucinated change is a critical compliance failure.
- Use the section identifiers exactly as they appear in the documents.
- 'topics_touched' must be legal/commercial categories (e.g. Pricing, Term & \
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

Extract every change the amendment introduces and return the structured \
output."""


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
