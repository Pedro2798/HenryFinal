"""Multimodal image parsing with GPT-4o Vision.

Responsibilities (Step 1 of the assignment):

* validate the image file (existence, type, size);
* encode it to a base64 data URI;
* call GPT-4o with a vision prompt engineered for *faithful* transcription
  that preserves the clause/section hierarchy of the contract;
* return a validated :class:`ParsedDocument` with token/latency metrics.

The OpenAI client is the Langfuse drop-in (``from langfuse.openai import
OpenAI``), so every vision call is automatically recorded as a Langfuse
*generation* (model, prompt/response, token usage, latency) nested under
whatever span is active in ``main``. If Langfuse is unavailable we transparently
fall back to the vanilla OpenAI SDK so the core pipeline never breaks during a
live demo.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

# Error classes live in the real openai package regardless of the wrapper.
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)

from .models import ParsedDocument

logger = logging.getLogger(__name__)

# OpenAI Vision accepts images up to 20 MB; we validate before sending.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ImageValidationError(ValueError):
    """Raised when the supplied image path is missing or unusable."""


class ContractParsingError(RuntimeError):
    """Raised when the GPT-4o Vision call fails or returns no usable text."""


# ---------------------------------------------------------------------------
# Vision prompt - engineered for fidelity, not summarization.
# ---------------------------------------------------------------------------
_VISION_SYSTEM_PROMPT = (
    "You are a forensic legal document transcription engine. You convert "
    "scanned contract images into faithful, machine-readable text. You never "
    "summarize, interpret, paraphrase or omit content."
)

_VISION_USER_PROMPT = (
    "Transcribe this contract image to text with maximum fidelity.\n\n"
    "Strict rules:\n"
    "1. Reproduce every clause, sub-clause, heading, party name, amount, date "
    "and defined term EXACTLY as written.\n"
    "2. Preserve the document hierarchy: keep clause/section numbers and "
    "titles, and keep their nesting (e.g. '4.', '4.1', '(a)').\n"
    "3. Preserve tables as readable text (one row per line, columns separated "
    "by ' | ').\n"
    "4. Do NOT add commentary, explanations, summaries or markdown fences.\n"
    "5. Do NOT correct or normalize numbers, currencies or dates - copy them "
    "verbatim.\n"
    "6. If a fragment is genuinely unreadable, write '[ILLEGIBLE]' in its "
    "place rather than guessing.\n\n"
    "Return only the transcribed contract text."
)


def build_openai_client(*, timeout: float = 60.0, max_retries: int = 3):
    """Return an OpenAI client wrapped with Langfuse instrumentation.

    Falls back to the vanilla SDK if the Langfuse drop-in is not importable so
    the pipeline degrades gracefully instead of crashing.
    """

    try:
        from langfuse.openai import OpenAI  # type: ignore

        logger.debug("Using Langfuse-instrumented OpenAI client.")
    except Exception:  # pragma: no cover - defensive import fallback
        from openai import OpenAI  # type: ignore

        logger.warning(
            "langfuse.openai unavailable - vision calls will not be traced."
        )
    return OpenAI(timeout=timeout, max_retries=max_retries)


def validate_image_path(image_path: str | Path) -> Path:
    """Validate the image file and return a resolved :class:`Path`.

    Raises :class:`ImageValidationError` with an actionable message on any
    problem (missing file, wrong type, empty or oversized image).
    """

    path = Path(image_path).expanduser()
    if not path.exists():
        raise ImageValidationError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ImageValidationError(f"Path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ImageValidationError(
            f"Unsupported image type '{suffix}'. Allowed: "
            f"{', '.join(sorted(_ALLOWED_SUFFIXES))}."
        )

    size = path.stat().st_size
    if size == 0:
        raise ImageValidationError(f"Image file is empty: {path}")
    if size > _MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image is {size / 1_048_576:.1f} MB; OpenAI Vision limit is 20 MB."
        )

    return path.resolve()


def encode_image_to_base64(path: Path) -> tuple[str, str]:
    """Read the file and return ``(base64_string, mime_type)``."""

    try:
        raw = path.read_bytes()
    except OSError as exc:  # permission / IO errors
        raise ImageValidationError(f"Could not read image '{path}': {exc}") from exc

    encoded = base64.b64encode(raw).decode("utf-8")
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    return encoded, mime


def parse_contract_image(
    image_path: str | Path,
    *,
    label: str,
    client,
    model: str = "gpt-4o",
    max_tokens: int = 4096,
) -> ParsedDocument:
    """Parse one contract image into faithful text via GPT-4o Vision.

    Parameters
    ----------
    image_path:
        Path to a JPEG/PNG/WEBP/GIF contract scan.
    label:
        Role of the document ("original" or "amendment") - used for logs,
        tracing metadata and the returned model.
    client:
        A (Langfuse-wrapped) OpenAI client, injected by ``main`` so the call
        is traced under the active span.
    model:
        Vision-capable model id (defaults to ``gpt-4o``).
    max_tokens:
        Upper bound on the transcription length.

    Returns
    -------
    ParsedDocument
        Validated text plus token/latency metrics.
    """

    path = validate_image_path(image_path)
    b64, mime = encode_image_to_base64(path)
    logger.info("Parsing %s contract image: %s", label, path.name)

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )
    except APITimeoutError as exc:
        raise ContractParsingError(
            f"GPT-4o Vision timed out while parsing the {label} contract. "
            f"Retry or increase the client timeout. ({exc})"
        ) from exc
    except RateLimitError as exc:
        raise ContractParsingError(
            f"OpenAI rate/quota limit hit while parsing the {label} contract. "
            f"Check your plan or retry later. ({exc})"
        ) from exc
    except BadRequestError as exc:
        # e.g. image too large for the model, context/token limit exceeded.
        raise ContractParsingError(
            f"OpenAI rejected the {label} contract request "
            f"(image or token limit). ({exc})"
        ) from exc
    except APIConnectionError as exc:
        raise ContractParsingError(
            f"Network failure contacting OpenAI for the {label} contract. ({exc})"
        ) from exc
    except APIError as exc:  # catch-all for any other API-side failure
        raise ContractParsingError(
            f"OpenAI API error while parsing the {label} contract. ({exc})"
        ) from exc

    latency = time.perf_counter() - start

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise ContractParsingError(
            f"GPT-4o Vision returned no text for the {label} contract "
            f"(finish_reason={choice.finish_reason})."
        )

    usage = getattr(response, "usage", None)
    try:
        return ParsedDocument(
            label=label,
            source_path=str(path),
            text=text,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_seconds=round(latency, 3),
        )
    except Exception as exc:  # pydantic ValidationError -> domain error
        raise ContractParsingError(
            f"Parsed {label} contract failed validation: {exc}"
        ) from exc
