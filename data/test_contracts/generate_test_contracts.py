"""Reproducibly render the synthetic test contracts as PNG images.

These are *clean* rendered document images (not noisy scans) on purpose:
they give GPT-4o Vision a deterministic, reliable target for the live demo
while still fully exercising the parsing -> contextualization -> extraction
pipeline. Re-generate with:

    pip install pillow==11.0.0
    python data/test_contracts/generate_test_contracts.py

Output (same directory):
    01_service_agreement_original.png   02_service_agreement_amendment.png
    03_nda_original.png                 04_nda_amendment.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent
PAGE_W = 1240          # ~A4 width @150 dpi
MARGIN = 96
INK = (17, 17, 17)
PAPER = (255, 255, 255)


def _font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    """Best-effort load of a common system font, with graceful fallback."""

    candidates = []
    if bold:
        candidates += ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
    if italic:
        candidates += ["ariali.ttf", "DejaVuSans-Oblique.ttf"]
    candidates += ["arial.ttf", "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Block = (style, text). Styles: title | preamble | heading | body | spacer
Document = list[tuple[str, str]]


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _render(doc: Document, path: Path) -> None:
    max_w = PAGE_W - 2 * MARGIN
    fonts = {
        "title": _font(40, bold=True),
        "preamble": _font(23, italic=True),
        "heading": _font(26, bold=True),
        "body": _font(23),
    }
    # Layout pass: build (font, line, x, gap_after) then measure height.
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    layout: list[tuple[object, str, int, int]] = []
    for style, text in doc:
        if style == "spacer":
            layout.append((fonts["body"], "", MARGIN, 18))
            continue
        font = fonts[style]
        line_h = font.size + 10
        for ln in _wrap(probe, text, font, max_w):
            layout.append((font, ln, MARGIN, line_h))
        layout.append((font, "", MARGIN, 14 if style == "heading" else 20))

    height = MARGIN
    for _, _, _, gap in layout:
        height += gap
    height += MARGIN

    img = Image.new("RGB", (PAGE_W, height), PAPER)
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for font, line, x, gap in layout:
        if line:
            if font is fonts["title"]:
                w = draw.textlength(line, font=font)
                draw.text(((PAGE_W - w) / 2, y), line, font=font, fill=INK)
            else:
                draw.text((x, y), line, font=font, fill=INK)
        y += gap
    img.save(path, "PNG")
    print(f"  wrote {path.name}  ({PAGE_W}x{height})")


# ---------------------------------------------------------------------------
# PAIR 1 - Service Agreement (SIMPLE change: monthly fee + end date)
# ---------------------------------------------------------------------------
SERVICE_ORIGINAL: Document = [
    ("title", "SERVICE AGREEMENT"),
    ("spacer", ""),
    ("preamble", 'This Service Agreement ("Agreement") is entered into on '
     "1 March 2025 between LegalMove S.A. (the \"Client\") and DataBridge "
     "Solutions LLC (the \"Provider\")."),
    ("spacer", ""),
    ("heading", "1. Scope of Services"),
    ("body", "The Provider shall deliver cloud data-integration services and "
     "monthly platform maintenance as described in Annex A."),
    ("heading", "2. Term"),
    ("body", "This Agreement is effective from 1 March 2025 and shall remain "
     "in force until 31 December 2025, unless earlier terminated in "
     "accordance with Clause 6."),
    ("heading", "3. Fees"),
    ("body", "The Client shall pay the Provider a monthly fee of USD 5,000, "
     "payable within fifteen (15) days of each invoice."),
    ("heading", "4. Confidentiality"),
    ("body", "Each party shall keep confidential all non-public information "
     "disclosed by the other party under this Agreement."),
    ("heading", "5. Governing Law"),
    ("body", "This Agreement shall be governed by and construed in accordance "
     "with the laws of the Republic of Argentina."),
    ("heading", "6. Termination"),
    ("body", "Either party may terminate this Agreement upon thirty (30) days "
     "prior written notice."),
]

SERVICE_AMENDMENT: Document = [
    ("title", "AMENDMENT No. 1 TO THE SERVICE AGREEMENT"),
    ("spacer", ""),
    ("preamble", 'This Amendment No. 1 ("Amendment"), dated 1 December 2025, '
     "modifies the Service Agreement dated 1 March 2025 between LegalMove "
     "S.A. and DataBridge Solutions LLC."),
    ("spacer", ""),
    ("heading", "1. Amendment to Clause 3 (Fees)"),
    ("body", "The monthly fee set out in Clause 3 is increased from "
     "USD 5,000 to USD 6,500, effective 1 January 2026."),
    ("heading", "2. Amendment to Clause 2 (Term)"),
    ("body", "The end date in Clause 2 is extended from 31 December 2025 to "
     "30 June 2026."),
    ("heading", "3. No Other Changes"),
    ("body", "All other terms and conditions of the Service Agreement remain "
     "in full force and effect."),
]

# ---------------------------------------------------------------------------
# PAIR 2 - NDA (COMPLEX: add clause + modify territory + delete restriction)
# ---------------------------------------------------------------------------
NDA_ORIGINAL: Document = [
    ("title", "NON-DISCLOSURE AGREEMENT"),
    ("spacer", ""),
    ("preamble", 'This Non-Disclosure Agreement ("NDA") is entered into on '
     "10 February 2025 between LegalMove S.A. (the \"Disclosing Party\") and "
     "Northwind Analytics Inc. (the \"Receiving Party\")."),
    ("spacer", ""),
    ("heading", "1. Definition of Confidential Information"),
    ("body", '"Confidential Information" means any non-public technical, '
     "commercial or financial information disclosed by the Disclosing Party "
     "to the Receiving Party, in any form."),
    ("heading", "2. Obligations of the Receiving Party"),
    ("body", "The Receiving Party shall protect the Confidential Information "
     "with at least the same degree of care it uses for its own confidential "
     "information, and no less than reasonable care."),
    ("heading", "3. Permitted Use"),
    ("body", "The Receiving Party shall use the Confidential Information "
     "solely to evaluate a potential business relationship. The Receiving "
     "Party shall not reverse engineer, decompile or disassemble any "
     "prototype or software provided by the Disclosing Party."),
    ("heading", "4. Territorial Scope"),
    ("body", "The obligations under this NDA apply within the territory of "
     "Argentina."),
    ("heading", "5. Term"),
    ("body", "The confidentiality obligations shall survive for three (3) "
     "years from the date of disclosure."),
    ("heading", "6. Remedies"),
    ("body", "The Disclosing Party shall be entitled to seek injunctive "
     "relief for any actual or threatened breach of this NDA."),
]

NDA_AMENDMENT: Document = [
    ("title", "AMENDMENT No. 1 TO THE NON-DISCLOSURE AGREEMENT"),
    ("spacer", ""),
    ("preamble", 'This Amendment No. 1 ("Amendment"), dated 5 December 2025, '
     "amends the Non-Disclosure Agreement dated 10 February 2025 between "
     "LegalMove S.A. and Northwind Analytics Inc."),
    ("spacer", ""),
    ("heading", "1. Modification of Clause 4 (Territorial Scope)"),
    ("body", 'Clause 4 is amended and restated to read: "The obligations '
     "under this NDA apply within Argentina, Brazil, Chile, Uruguay and the "
     'European Union."'),
    ("heading", "2. Deletion in Clause 3 (Permitted Use)"),
    ("body", "The last sentence of Clause 3 (\"The Receiving Party shall not "
     "reverse engineer, decompile or disassemble any prototype or software "
     "provided by the Disclosing Party.\") is hereby deleted in its "
     "entirety."),
    ("heading", "3. New Clause 7 (Data Protection)"),
    ("body", "A new Clause 7 is added: \"The Receiving Party shall process "
     "any personal data received under this NDA in compliance with "
     "Regulation (EU) 2016/679 (GDPR) and shall notify the Disclosing Party "
     'of any personal data breach within forty-eight (48) hours."'),
    ("heading", "4. No Other Changes"),
    ("body", "Except as expressly amended herein, the NDA remains in full "
     "force and effect."),
]


def main() -> None:
    print("Rendering synthetic test contracts ->", OUT_DIR)
    _render(SERVICE_ORIGINAL, OUT_DIR / "01_service_agreement_original.png")
    _render(SERVICE_AMENDMENT, OUT_DIR / "02_service_agreement_amendment.png")
    _render(NDA_ORIGINAL, OUT_DIR / "03_nda_original.png")
    _render(NDA_AMENDMENT, OUT_DIR / "04_nda_amendment.png")
    print("Done.")


if __name__ == "__main__":
    main()
