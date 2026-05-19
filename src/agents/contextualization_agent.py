"""Agent 1 - Contextualization.

Role: a *Senior Legal Contract Analyst*. Its ONLY job is to read the two
documents and produce a comparative **structure map** - which sections exist
in each document, how they correspond, and what each one governs.

It deliberately does **not** extract or describe changes. That separation of
concerns is the core of the two-agent design: a clean structural map lets the
extraction agent (Agent 2) focus 100% of its attention on diffing, which
measurably reduces hallucinated or missed changes.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models import ContractStructureMap
from .base import AgentError

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a Senior Legal Contract Analyst at LegalMove with \
15 years of experience structuring commercial agreements.

Your SOLE task is to build a COMPARATIVE STRUCTURE MAP of two documents: an \
ORIGINAL contract and its AMENDMENT (addendum). You are the first analyst in a \
two-person review; a second auditor will use your map to extract the actual \
changes, so your map must be accurate and complete.

What you MUST do:
- Identify every logical section/clause in EACH document, using the clause \
numbers and titles exactly as they appear in the text.
- Determine, for each section, whether it is present in the original, in the \
amendment, or both.
- Explain in one sentence the legal purpose of each section.
- Describe the structural correspondence between the two documents (aligned, \
renumbered, newly introduced by the amendment, removed, or merely referenced).

What you MUST NOT do:
- Do NOT list, quote, diff or describe the substantive CHANGES (amounts, \
dates, scope). That is the next analyst's job. Reporting changes here is an \
error.
- Do NOT invent sections that are not in the text.
- Do NOT summarize the contract's commercial intent beyond the brief overview.

Think like a structural cartographer, not an auditor. Output strictly the \
requested structured object."""

_HUMAN_PROMPT = """ORIGINAL CONTRACT (verbatim transcription):
\"\"\"
{original_text}
\"\"\"

AMENDMENT / ADDENDUM (verbatim transcription):
\"\"\"
{amendment_text}
\"\"\"

Build the comparative structure map now."""


class ContextualizationAgent:
    """Builds a :class:`ContractStructureMap` from two parsed documents."""

    name = "contextualization_agent"

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
        # Strict JSON-schema structured output -> returns a validated
        # ContractStructureMap instance directly.
        self._structured_llm = llm.with_structured_output(
            ContractStructureMap, method="json_schema", strict=True
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
        )
        self._chain = self._prompt | self._structured_llm

    def analyze(
        self,
        original_text: str,
        amendment_text: str,
        *,
        callbacks: list | None = None,
    ) -> ContractStructureMap:
        """Produce the comparative structure map.

        ``callbacks`` carries the Langfuse handler so this LLM call is traced
        as a generation nested under the agent's span.
        """

        logger.info("ContextualizationAgent: building structure map.")
        try:
            result = self._chain.invoke(
                {
                    "original_text": original_text,
                    "amendment_text": amendment_text,
                },
                config={"callbacks": callbacks or []},
            )
        except Exception as exc:
            raise AgentError(
                f"ContextualizationAgent failed to produce a structure map: {exc}"
            ) from exc

        if not isinstance(result, ContractStructureMap):
            raise AgentError(
                "ContextualizationAgent returned an unexpected type: "
                f"{type(result)!r}"
            )
        logger.info(
            "ContextualizationAgent: mapped %d sections.", len(result.sections)
        )
        return result
