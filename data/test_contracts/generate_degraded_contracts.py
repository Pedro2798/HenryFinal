"""Generate degraded versions of the Service Agreement test pair.

Simulates the artifacts of a real-world contract photo / re-scan, with an
emphasis on *believable* coffee stains:

  1. mild rotation (handheld phone shot)
  2. realistic coffee stains: multi-lobe organic body + dark coffee-ring at
     the rim (the actual physical effect of drying coffee), splash droplets
     around the main blob, and a streaking drip
  3. fold-shadow with Gaussian falloff (paper crease, not a flat band)
  4. subtle paper yellowing (aged tint)
  5. slight Gaussian blur (low-end camera / mild motion)
  6. JPEG re-encoding at quality ~78 for recompression artefacts

Stains are positioned so they brush against, but do not fully obliterate,
critical values (amounts, dates). If we *did* obscure a number, GPT-4o is
instructed by our vision prompt to emit ``[ILLEGIBLE]`` rather than
hallucinate (see src/image_parser.py).

Reproducible (fixed seed). Re-generate with:

    pip install pillow==11.0.0
    python data/test_contracts/generate_degraded_contracts.py
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

OUT_DIR = Path(__file__).parent
SEED = 73


# ---------------------------------------------------------------------------
# Coffee stain (realistic)
# ---------------------------------------------------------------------------
def _coffee_stain(rng: random.Random, diameter: int) -> Image.Image:
    """Return an RGBA Image containing a single realistic coffee stain.

    Composition:
      - Multi-lobe organic body (overlapping ellipses around a center, each
        with slightly different brown tones for depth).
      - A concentrated dark ring near the rim (coffee-ring effect).
      - Small splash droplets in a radius around the body.
      - An optional drip / streak in a random direction.
    """

    canvas = diameter * 3  # padding for blur halo + drip overshoot
    layer = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = cy = canvas // 2
    r = diameter // 2

    # --- 1. Body: 7-10 overlapping ellipses for an organic blob ---
    n_lobes = rng.randint(7, 10)
    for i in range(n_lobes):
        ang = (360 / n_lobes) * i + rng.uniform(-25, 25)
        off = rng.uniform(0.10, 0.45) * r
        ox = int(math.cos(math.radians(ang)) * off)
        oy = int(math.sin(math.radians(ang)) * off)
        lobe_rx = int(r * rng.uniform(0.55, 1.00))
        lobe_ry = int(r * rng.uniform(0.55, 1.00))
        body_color = (
            rng.randint(125, 160),    # R - brown
            rng.randint(78, 105),     # G
            rng.randint(38, 58),      # B
            rng.randint(55, 85),      # alpha (translucent)
        )
        draw.ellipse(
            [cx + ox - lobe_rx, cy + oy - lobe_ry,
             cx + ox + lobe_rx, cy + oy + lobe_ry],
            fill=body_color,
        )

    # Soften body edges
    layer = layer.filter(ImageFilter.GaussianBlur(radius=6))

    # --- 2. Coffee-ring: concentrated darker pigment at the rim ---
    ring = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    # Draw a slightly-jittery ring as a series of small dots along the rim
    rim_r = int(r * rng.uniform(0.86, 0.95))
    n_dots = 90
    for k in range(n_dots):
        a = (360 / n_dots) * k + rng.uniform(-4, 4)
        jitter = rng.uniform(-r * 0.03, r * 0.04)
        rr = rim_r + jitter
        px = int(cx + math.cos(math.radians(a)) * rr)
        py = int(cy + math.sin(math.radians(a)) * rr)
        dot_r = rng.randint(2, 5)
        rd.ellipse(
            [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
            fill=(58, 28, 12, rng.randint(130, 200)),
        )
    ring = ring.filter(ImageFilter.GaussianBlur(radius=2.2))
    layer = Image.alpha_composite(layer, ring)

    # --- 3. Splash droplets in a halo around the body ---
    splash = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    sd = ImageDraw.Draw(splash)
    for _ in range(rng.randint(6, 12)):
        a = rng.uniform(0, 360)
        d = rng.uniform(r * 1.05, r * 1.7)
        dx = int(math.cos(math.radians(a)) * d)
        dy = int(math.sin(math.radians(a)) * d)
        dr = rng.randint(2, 7)
        sd.ellipse(
            [cx + dx - dr, cy + dy - dr, cx + dx + dr, cy + dy + dr],
            fill=(90, 50, 22, rng.randint(120, 200)),
        )
    splash = splash.filter(ImageFilter.GaussianBlur(radius=1.0))
    layer = Image.alpha_composite(layer, splash)

    # --- 4. Drip / streak (60% chance) ---
    if rng.random() < 0.65:
        streak = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        std = ImageDraw.Draw(streak)
        ang = rng.uniform(0, 360)
        length = rng.randint(int(r * 0.6), int(r * 1.5))
        for t in range(length):
            ratio = t / length
            wobble = math.sin(t * 0.12) * (r * 0.05)
            perp = ang + 90
            dx = int(math.cos(math.radians(ang)) * t +
                     math.cos(math.radians(perp)) * wobble)
            dy = int(math.sin(math.radians(ang)) * t +
                     math.sin(math.radians(perp)) * wobble)
            rad = max(1, int(r * 0.18 * (1 - ratio)))
            alpha = int(95 * (1 - ratio))
            std.ellipse(
                [cx + dx - rad, cy + dy - rad,
                 cx + dx + rad, cy + dy + rad],
                fill=(105, 60, 28, alpha),
            )
        streak = streak.filter(ImageFilter.GaussianBlur(radius=2.5))
        layer = Image.alpha_composite(layer, streak)

    return layer


def _paste_stain(img: Image.Image, rng: random.Random,
                 anchor_xy: tuple[int, int], diameter: int) -> Image.Image:
    """Paste a freshly generated stain centered at ``anchor_xy``."""

    stain = _coffee_stain(rng, diameter)
    sw, sh = stain.size
    rgba = img.convert("RGBA")
    rgba.alpha_composite(stain, dest=(anchor_xy[0] - sw // 2,
                                      anchor_xy[1] - sh // 2))
    return rgba.convert("RGB")


# ---------------------------------------------------------------------------
# Other artefacts
# ---------------------------------------------------------------------------
def _fold_shadow(img: Image.Image, y_position: int,
                 strength: int = 70) -> Image.Image:
    """Darken a thin horizontal line and blur it -> realistic paper crease."""

    w, h = img.size
    mask = Image.new("L", (w, h), 255)
    md = ImageDraw.Draw(mask)
    md.rectangle([0, y_position, w, y_position + 2], fill=255 - strength)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=14))
    shadow_rgb = Image.merge("RGB", (mask, mask, mask))
    return ImageChops.multiply(img.convert("RGB"), shadow_rgb)


def _paper_yellowing(img: Image.Image, alpha: int = 20) -> Image.Image:
    """Subtle warm overlay to simulate aged paper."""

    w, h = img.size
    tint = Image.new("RGBA", (w, h), (255, 232, 195, alpha))
    out = img.convert("RGBA")
    out.alpha_composite(tint)
    return out.convert("RGB")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _degrade(src_path: Path, recipe: dict) -> Image.Image:
    img = Image.open(src_path).convert("RGB")
    rng = random.Random(SEED + sum(ord(c) for c in src_path.name))

    # 1. Rotation (off-axis page)
    img = img.rotate(
        recipe["rotation_deg"],
        fillcolor=(252, 250, 246),
        expand=False,
        resample=Image.BICUBIC,
    )

    # 2. Subtle aged-paper tint
    img = _paper_yellowing(img, alpha=18)

    # 3. Coffee stains
    for (ax, ay, dia) in recipe["stain_specs"]:
        img = _paste_stain(img, rng, (ax, ay), dia)

    # 4. Fold-shadow with Gaussian falloff
    img = _fold_shadow(img, recipe["fold_y"], strength=65)

    # 5. Slight blur (low-end camera / motion)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.55))

    # 6. JPEG recompression for artefacts
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    return img


_RECIPES = [
    {
        "src": "01_service_agreement_original.png",
        "dst": "05_service_agreement_original_dirty.png",
        "rotation_deg": 3.2,
        "fold_y": 540,
        "stain_specs": [
            (970, 880, 260),    # right side, lower body
            (260, 1050, 200),   # bottom-left
        ],
    },
    {
        "src": "02_service_agreement_amendment.png",
        "dst": "06_service_agreement_amendment_dirty.png",
        "rotation_deg": -2.5,
        "fold_y": 360,
        "stain_specs": [
            (880, 470, 270),    # brushes "30 June 2026" without erasing it
            (220, 640, 180),    # bottom-left
        ],
    },
]


def main() -> None:
    print("Rendering degraded contracts ->", OUT_DIR)
    for recipe in _RECIPES:
        src = OUT_DIR / recipe["src"]
        if not src.exists():
            raise SystemExit(f"Missing source image: {src}")
        out_img = _degrade(src, recipe)
        out_path = OUT_DIR / recipe["dst"]
        out_img.save(out_path, format="PNG")
        print(f"  wrote {recipe['dst']}  "
              f"({out_img.size[0]}x{out_img.size[1]})")
    print("Done.")


if __name__ == "__main__":
    main()
