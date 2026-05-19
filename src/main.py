"""LegalMove - Autonomous Contract Comparison Agent (entry point).

Usage
-----
    python -m src.main <original_image> <amendment_image> [options]
    python src/main.py  <original_image> <amendment_image> [options]

Pipeline (each stage is a child observation of one root Langfuse trace):

    contract-analysis                 (root span = trace)
    |- parse_original_contract        GPT-4o Vision  -> ParsedDocument
    |- parse_amendment_contract       GPT-4o Vision  -> ParsedDocument
    |- contextualization_agent        Agent 1        -> ContractStructureMap
    |- extraction_agent               Agent 2        -> ContractChangeOutput
    \\- pydantic_validation            explicit model_validate()

Observability (Langfuse Python SDK v4, following the official Langfuse skill's
instrumentation best practices):

* GPT-4o Vision calls are auto-traced as *generations* by the Langfuse OpenAI
  drop-in; the agent LLM calls by the Langfuse LangChain ``CallbackHandler``.
  Both nest under their parent observation via OpenTelemetry context, so model
  name, token usage and latency are captured automatically.
* Trace-level context (``session_id``, ``user_id``, ``version``, ``tags``) is
  attached with ``propagate_attributes``; explicit trace input/output with
  ``set_trace_io`` (only meaningful data, never raw function args/secrets).
* A ``mask`` function redacts credential/PII patterns before any data leaves
  the process. Contract clause text is intentionally kept (it is the payload
  the compliance team needs to see), credentials/emails are not.
* ``flush()`` runs in ``finally`` so traces are sent before the CLI exits.
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
import re
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from src import __version__
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


# ---------------------------------------------------------------------------
# Data masking (Langfuse best practice: never leak secrets/PII into traces)
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = (
    re.compile(r"sk-lf-[A-Za-z0-9_\-]{6,}"),          # Langfuse secret key
    re.compile(r"pk-lf-[A-Za-z0-9_\-]{6,}"),          # Langfuse public key
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),            # OpenAI API key
    re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]+"),      # bearer tokens
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),          # emails (PII)
)


def _mask_sensitive(data, **kwargs):
    """Langfuse mask hook: redact credentials/PII, keep contract text.

    Called by the SDK on every observation input/output/metadata before it is
    sent. Recurses into dicts/lists. We deliberately do NOT redact the contract
    clause text - that is the payload the compliance reviewer must see in the
    trace - only secret-shaped tokens and email addresses.
    """

    if isinstance(data, str):
        redacted = data
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    if isinstance(data, dict):
        return {key: _mask_sensitive(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask_sensitive(item) for item in data]
    return data


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ("httpx", "openai", "langfuse", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _check_environment() -> list[str]:
    """Return the list of missing required environment variables."""

    return [name for name in _REQUIRED_ENV if not os.getenv(name)]


def _apply_langfuse_env_defaults() -> None:
    """Set Langfuse env-based config (v4 reads these from the environment).

    v4 removed the ``environment``/``release`` client params in favour of
    ``LANGFUSE_TRACING_ENVIRONMENT`` / ``LANGFUSE_RELEASE``. We default them so
    traces are always attributed even if the user did not set them in .env.
    """

    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    os.environ.setdefault("LANGFUSE_RELEASE", f"legalmove@{__version__}")


def _preview(text: str, limit: int = 1500) -> str:
    """Trim long text for trace payloads (keeps traces readable & cheap)."""

    return text if len(text) <= limit else text[:limit] + " […truncated]"


def _default_session_id(original_image: str, amendment_image: str) -> str:
    """Group every run of the same contract pair under one Langfuse session."""

    return (
        os.getenv("LANGFUSE_SESSION_ID")
        or f"contract-pair::{Path(original_image).stem}__{Path(amendment_image).stem}"
    )


def _default_user_id() -> str:
    """Attribute the run to the compliance analyst (cost/quality by user)."""

    return (
        os.getenv("LANGFUSE_USER_ID")
        or os.getenv("USERNAME")
        or os.getenv("USER")
        or "compliance-analyst"
    )


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
        "--session-id",
        default=None,
        help="Langfuse session id (default: derived from the contract pair).",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Langfuse user id for cost/quality attribution.",
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
    langfuse,
    session_id: str,
    user_id: str,
) -> ContractChangeOutput:
    """Execute the full multi-agent pipeline under one Langfuse v4 trace."""

    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

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

    trace_input = {
        "original_image": original_image,
        "amendment_image": amendment_image,
        "model": model,
    }

    with langfuse.start_as_current_observation(
        as_type="span", name="contract-analysis", input=trace_input
    ) as root:
        # Trace-level context: groups runs (session), attributes cost (user),
        # ties traces to a release & environment, and enables tag filtering.
        with propagate_attributes(
            trace_name="contract-analysis",
            session_id=session_id,
            user_id=user_id,
            version=__version__,
            tags=["legalmove", "contract-comparison"],
            metadata={"pipeline": "vision->contextualization->extraction"},
        ):
            try:
                # --- Step 1a: parse the ORIGINAL contract ------------------
                with langfuse.start_as_current_observation(
                    as_type="span", name="parse_original_contract"
                ) as span:
                    original = parse_contract_image(
                        original_image,
                        label="original",
                        client=client,
                        model=model,
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

                # --- Step 1b: parse the AMENDMENT --------------------------
                with langfuse.start_as_current_observation(
                    as_type="span", name="parse_amendment_contract"
                ) as span:
                    amendment = parse_contract_image(
                        amendment_image,
                        label="amendment",
                        client=client,
                        model=model,
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

                # --- Step 2: Agent 1 - Contextualization -------------------
                with langfuse.start_as_current_observation(
                    as_type="span", name="contextualization_agent"
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

                # --- Step 3: Agent 2 - Extraction (uses Agent 1's map) -----
                with langfuse.start_as_current_observation(
                    as_type="span", name="extraction_agent"
                ) as span:
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

                # --- Step 4: explicit Pydantic validation ------------------
                with langfuse.start_as_current_observation(
                    as_type="span", name="pydantic_validation"
                ) as span:
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

                # In v4 the trace input/output is derived from the ROOT
                # observation (input set at start_as_current_observation,
                # output here) - set_trace_io is deprecated.
                root.update(output=validated.model_dump())
                return validated

            except Exception as exc:  # annotate trace + re-raise
                root.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise


def _init_langfuse():
    """Construct the singleton Langfuse v4 client with the mask hook.

    Must be created before the OpenAI drop-in / LangChain handler are used so
    masking applies to integration-generated observations too.
    """

    from langfuse import Langfuse

    return Langfuse(mask=_mask_sensitive)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    load_dotenv()
    _apply_langfuse_env_defaults()

    missing = _check_environment()
    if missing:
        logger.error(
            "Missing required environment variables: %s\n"
            "Copy .env.example to .env and fill in your credentials.",
            ", ".join(missing),
        )
        return _EXIT_CONFIG

    langfuse = None
    try:
        langfuse = _init_langfuse()
        result = run_pipeline(
            args.original_image,
            args.amendment_image,
            model=args.model,
            langfuse=langfuse,
            session_id=args.session_id
            or _default_session_id(args.original_image, args.amendment_image),
            user_id=args.user_id or _default_user_id(),
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
        logger.error("Final output failed Pydantic validation:\n%s", exc)
        return _EXIT_VALIDATION
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return _EXIT_UNEXPECTED
    except Exception as exc:  # last-resort safety net
        logger.exception("Unexpected error: %s", exc)
        return _EXIT_UNEXPECTED
    finally:
        # Flush buffered observations before the process exits, otherwise the
        # trace may never reach Langfuse (sent on a background thread).
        if langfuse is not None:
            try:
                langfuse.flush()
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
