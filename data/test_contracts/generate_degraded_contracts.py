"""Generate degraded versions of the Service Agreement test pair.

Simulates the artifacts of a real-world contract photo / re-scan:
  1. mild rotation (handheld phone shot, not perfectly aligned)
  2. coffee-stain blobs over the page (semi-transparent brown ellipses)
  3. a fold-shadow horizontal band (paper crease)
  4. slight Gaussian blur (low-end camera / mild motion)
  5. JPEG re-encoding at quality 72 (recompression artefacts)

Stains are placed deliberately away from the top header so the contract
title and clause numbers stay legible - the goal is to demonstrate
robustness, not to hide critical data. If we *did* obscure a number,
GPT-4o is instructed by our vision prompt to emit ``[ILLEGIBLE]`` rather
than hallucinate (see src/image_parser.py).

Reproducible (fixed seed). Re-generate with:

    pip install pillow==11.0.0
    python data/test_contracts/generate_degraded_contracts.py
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = Path(__file__).parent
SEED = 42


def _degrade(
    src_path: Path,
    *,
    rotation_deg: float,
    n_stains: int,
    fold_y_offset: int,
) -> Image.Image:
    rng = random.Random(SEED + sum(ord(c) for c in src_path.name))
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # 1. Rotation (off-axis page on a desk). Paper-white fill for corners.
    img = img.rotate(
        rotation_deg,
        fillcolor=(252, 250, 246),
        expand=False,
        resample=Image.BICUBIC,
    )

    # 2. Coffee-stain blobs. Restrict to lower half so they never cover the
    #    contract title or the first clauses where the key amounts live.
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_o = ImageDraw.Draw(overlay)
    safe_top = int(h * 0.55)
    for _ in range(n_stains):
        cx = rng.randint(100, w - 250)
        cy = rng.randint(safe_top, h - 120)
        rx = rng.randint(55, 110)
        ry = rng.randint(40, 80)
        alpha = rng.randint(60, 100)
        # Coffee brown with alpha + darker rim.
        draw_o.ellipse(
            [cx, cy, cx + 2 * rx, cy + 2 * ry], fill=(139, 75, 30, alpha)
        )
        draw_o.ellipse(
            [cx, cy, cx + 2 * rx, cy + 2 * ry],
            outline=(70, 35, 10, min(255, alpha + 40)),
            width=3,
        )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # 3. Fold shadow band (paper crease).
    fold = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_f = ImageDraw.Draw(fold)
    band_y = max(0, min(h - 50, h // 2 + fold_y_offset))
    draw_f.rectangle([0, band_y, w, band_y + 30], fill=(0, 0, 0, 55))
    img = Image.alpha_composite(img.convert("RGBA"), fold).convert("RGB")

    # 4. Slight blur (low-res camera / motion).
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    # 5. JPEG-style recompression for artefacts (then back to PNG).
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    return img


_RECIPES = [
    # (source, destination, rotation_deg, n_stains, fold_offset)
    (
        "01_service_agreement_original.png",
        "05_service_agreement_original_dirty.png",
        3.5, 2, -80,
    ),
    (
        "02_service_agreement_amendment.png",
        "06_service_agreement_amendment_dirty.png",
        -2.8, 2, 40,
    ),
]


def main() -> None:
    print("Rendering degraded test contracts ->", OUT_DIR)
    for src_name, dst_name, rot, stains, fold_off in _RECIPES:
        src = OUT_DIR / src_name
        if not src.exists():
            raise SystemExit(f"Missing source image: {src}")
        out = OUT_DIR / dst_name
        img = _degrade(
            src,
            rotation_deg=rot,
            n_stains=stains,
            fold_y_offset=fold_off,
        )
        img.save(out, format="PNG")
        print(f"  wrote {dst_name}  ({img.size[0]}x{img.size[1]})")
    print("Done.")


if __name__ == "__main__":
    main()
