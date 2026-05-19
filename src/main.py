"""LegalMove - Autonomous Contract Comparison Agent (entry point).

Usage
-----
    python -m src.main <original_image> <amendment_image> [options]
    python src/main.py  <original_image> <amendment_image> [options]

Pipeline (each stage is a child span of one root Langfuse trace):

    contract-analysis                 (root span = trace)
    |- parse_original_contract        GPT-4o Vision  -> ParsedDocument
    |- parse_amendment_contract       GPT-4o Vision  -> ParsedDocument
    |- contextualization_agent        Agent 1        -> ContractStructureMap
    |- extraction_agent               Agent 2        -> ContractChangeOutput
    \\- pydantic_validation            explicit model_validate()

The GPT-4o Vision calls are auto-traced as generations by the Langfuse OpenAI
drop-in; the agent LLM calls are auto-traced by the Langfuse LangChain
CallbackHandler. Both nest under their parent span via OpenTelemetry context,
so token usage and latency are captured at every stage.
"""

from __future__ import annotations

# --- run-as-script bootstrap: make `src` importable either way -------------
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

import argparse
import json
import logging
import textwrap

from dotenv import load_dotenv
from pydantic import ValidationError

from src.agents import ContextualizationAgent, ExtractionAgent
from src.agents.base import AgentError
from src.image_parser import (
    ContractParsingError,
    ImageValidationError,
    build_openai_client,
    parse_contract_image,
)
from src.models import ContractChangeOutput

logger = logging.getLogger("legalmove")

_REQUIRED_ENV = ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

# Exit codes (useful for CI / downstream automation).
_EXIT_OK = 0
_EXIT_USAGE = 2
_EXIT_CONFIG = 3
_EXIT_PARSING = 4
_EXIT_AGENT = 5
_EXIT_VALIDATION = 6
_EXIT_UNEXPECTED = 1


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers unless debugging.
    if not verbose:
        for noisy in ("httpx", "openai", "langfuse", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _check_environment() -> list[str]:
    """Return the list of missing required environment variables."""

    return [name for name in _REQUIRED_ENV if not os.getenv(name)]


def _preview(text: str, limit: int = 1500) -> str:
    """Trim long text for span payloads (keeps traces readable & cheap)."""

    return text if len(text) <= limit else text[:limit] + " […truncated]"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="legalmove",
        description="Compare an original contract image with its amendment and "
        "emit a Pydantic-validated JSON change report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("original_image", help="Path to the ORIGINAL contract image.")
    parser.add_argument("amendment_image", help="Path to the AMENDMENT image.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="Vision/reasoning model id (default: env OPENAI_MODEL or gpt-4o).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Optional path to also write the validated JSON report to.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser.parse_args(argv)


def run_pipeline(
    original_image: str,
    amendment_image: str,
    *,
    model: str,
) -> ContractChangeOutput:
    """Execute the full multi-agent pipeline under one Langfuse trace."""

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    langfuse = get_client()
    try:
        if langfuse.auth_check():
            logger.info("Langfuse authenticated - tracing enabled.")
        else:
            logger.warning(
                "Langfuse auth_check failed - check your keys/host. "
                "The pipeline will still run; traces may not be recorded."
            )
    except Exception as exc:  # network issues etc. - never block the demo
        logger.warning("Langfuse auth_check error (%s) - continuing.", exc)

    handler = CallbackHandler()
    client = build_openai_client()

    with langfuse.start_as_current_span(name="contract-analysis") as root:
        root.update_trace(
            name="contract-analysis",
            input={
                "original_image": original_image,
                "amendment_image": amendment_image,
                "model": model,
            },
            metadata={"pipeline": "vision -> contextualization -> extraction"},
            tags=["legalmove", "contract-comparison"],
        )
        try:
            # --- Step 1a: parse the ORIGINAL contract ----------------------
            with langfuse.start_as_current_span(
                name="parse_original_contract"
            ) as span:
                original = parse_contract_image(
                    original_image, label="original", client=client, model=model
                )
                span.update(
                    input={"image_path": original_image},
                    output={"text": _preview(original.text)},
                    metadata={
                        "characters": len(original.text),
                        "prompt_tokens": original.prompt_tokens,
                        "completion_tokens": original.completion_tokens,
                        "total_tokens": original.total_tokens,
                        "latency_seconds": original.latency_seconds,
                        "model": original.model,
                    },
                )

            # --- Step 1b: parse the AMENDMENT ------------------------------
            with langfuse.start_as_current_span(
                name="parse_amendment_contract"
            ) as span:
                amendment = parse_contract_image(
                    amendment_image, label="amendment", client=client, model=model
                )
                span.update(
                    input={"image_path": amendment_image},
                    output={"text": _preview(amendment.text)},
                    metadata={
                        "characters": len(amendment.text),
                        "prompt_tokens": amendment.prompt_tokens,
                        "completion_tokens": amendment.completion_tokens,
                        "total_tokens": amendment.total_tokens,
                        "latency_seconds": amendment.latency_seconds,
                        "model": amendment.model,
                    },
                )

            # --- Step 2: Agent 1 - Contextualization -----------------------
            with langfuse.start_as_current_span(
                name="contextualization_agent"
            ) as span:
                ctx_agent = ContextualizationAgent(model=model)
                structure_map = ctx_agent.analyze(
                    original.text, amendment.text, callbacks=[handler]
                )
                span.update(
                    input={
                        "original_text": _preview(original.text),
                        "amendment_text": _preview(amendment.text),
                    },
                    output=structure_map.model_dump(),
                    metadata={"sections_mapped": len(structure_map.sections)},
                )

            # --- Step 3: Agent 2 - Extraction (uses Agent 1's map) ---------
            with langfuse.start_as_current_span(name="extraction_agent") as span:
                ext_agent = ExtractionAgent(model=model)
                changes = ext_agent.extract(
                    structure_map,
                    original.text,
                    amendment.text,
                    callbacks=[handler],
                )
                span.update(
                    input={"structure_map": structure_map.model_dump()},
                    output=changes.model_dump(),
                    metadata={
                        "sections_changed": len(changes.sections_changed),
                        "topics_touched": len(changes.topics_touched),
                    },
                )

            # --- Step 4: explicit Pydantic validation ----------------------
            with langfuse.start_as_current_span(name="pydantic_validation") as span:
                # Round-trip through the schema to *prove* the production
                # contract holds, independently of structured-output guarantees.
                validated = ContractChangeOutput.model_validate(
                    changes.model_dump()
                )
                span.update(
                    input=changes.model_dump(),
                    output=validated.model_dump(),
                    metadata={
                        "schema": "ContractChangeOutput",
                        "valid": True,
                        "fields": list(validated.model_dump().keys()),
                    },
                )

            root.update(output=validated.model_dump())
            return validated

        except Exception as exc:  # annotate the trace, then re-raise
            root.update(level="ERROR", status_message=str(exc))
            raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    load_dotenv()

    missing = _check_environment()
    if missing:
        logger.error(
            "Missing required environment variables: %s\n"
            "Copy .env.example to .env and fill in your credentials.",
            ", ".join(missing),
        )
        return _EXIT_CONFIG

    try:
        result = run_pipeline(
            args.original_image, args.amendment_image, model=args.model
        )
    except ImageValidationError as exc:
        logger.error("Image validation error: %s", exc)
        return _EXIT_USAGE
    except ContractParsingError as exc:
        logger.error("Vision parsing error: %s", exc)
        return _EXIT_PARSING
    except AgentError as exc:
        logger.error("Agent error: %s", exc)
        return _EXIT_AGENT
    except ValidationError as exc:
        logger.error(
            "Final output failed Pydantic validation:\n%s", exc
        )
        return _EXIT_VALIDATION
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return _EXIT_UNEXPECTED
    except Exception as exc:  # last-resort safety net
        logger.exception("Unexpected error: %s", exc)
        return _EXIT_UNEXPECTED
    finally:
        # Flush buffered spans before the process exits, otherwise the trace
        # may never reach Langfuse (events are sent on a background thread).
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:  # pragma: no cover
            pass

    report = result.model_dump()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  CONTRACT CHANGE REPORT  (ContractChangeOutput - validated)")
    print("=" * 70)
    print(rendered)
    print("=" * 70)
    print("Summary:")
    print(textwrap.indent(textwrap.fill(report["summary_of_the_change"], 68), "  "))
    print("=" * 70 + "\n")

    if args.output:
        try:
            parent = os.path.dirname(os.path.abspath(args.output))
            os.makedirs(parent, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(rendered)
            logger.info("Report written to %s", args.output)
        except OSError as exc:
            logger.error("Could not write --output file: %s", exc)
            return _EXIT_UNEXPECTED

    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
