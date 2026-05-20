"""Generate "phone-photo of a coffee-stained contract" test pair.

Stacks several artefacts in order, simulating a real-world capture:

  1. Multiple realistic coffee stains directly on the paper:
       - Multi-lobular organic body (overlapping ellipses of different brown
         tones for depth).
       - Concentrated dark rim ("coffee-ring" effect from drying pigment).
       - Splash droplets in a halo around the main blob.
       - Drip / streak with slight wobble in a random direction.
       - Intensity parameter to make some stains heavy (truly opaque) and
         others light.
  2. One stain is positioned deliberately over a non-critical line so part
     of the document becomes illegible - GPT-4o is instructed by our vision
     prompt to emit ``[ILLEGIBLE]`` rather than guess (see image_parser.py).
     The amendment's changed *values* (amounts, dates) stay readable so the
     final report is still correct.
  3. Subtle warm yellowing tint (aged paper).
  4. Fold-shadow with Gaussian falloff (paper crease).
  5. PERSPECTIVE TRANSFORM: the page is warped to a trapezoid on a desk-tan
     background, mimicking a phone camera angle - this is a real perspective
     (not just rotation), with proper paper shadow.
  6. Light Gaussian blur (low-end camera / mild motion).
  7. JPEG re-encoding at quality 76 (recompression artefacts).

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
SEED = 91


# ---------------------------------------------------------------------------
# Perspective warp (phone-photo angle)
# ---------------------------------------------------------------------------
def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for an 8x8 system."""

    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(M[k][i]))
        M[i], M[pivot] = M[pivot], M[i]
        for k in range(i + 1, n):
            if M[i][i] == 0:
                continue
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x


def _perspective_coeffs(src_corners, dst_corners) -> list[float]:
    """8 coefficients for Pillow's Image.PERSPECTIVE (output -> source)."""

    A: list[list[float]] = []
    b: list[float] = []
    for (sx, sy), (dx, dy) in zip(src_corners, dst_corners):
        A.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        b.append(sx)
        A.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        b.append(sy)
    return _solve_linear(A, b)


def _phone_photo_warp(img: Image.Image, rng: random.Random) -> Image.Image:
    """Trapezoidal perspective + desk-tan background + paper drop-shadow."""

    w, h = img.size
    pad = 110
    out_w, out_h = w + pad * 2, h + pad * 2

    # Random-but-bounded trapezoid: page slightly tilted, top a bit narrower
    # (camera angled slightly down).
    dx_tl = rng.randint(28, 55)
    dx_tr = rng.randint(40, 75)
    dy_tl = rng.randint(20, 40)
    dy_tr = rng.randint(50, 80)
    dx_br = rng.randint(8, 22)
    dx_bl = rng.randint(6, 18)
    dy_br = rng.randint(10, 22)
    dy_bl = rng.randint(12, 25)

    src_corners = [(0, 0), (w, 0), (w, h), (0, h)]
    dst_corners = [
        (pad + dx_tl,             pad + dy_tl),
        (out_w - pad - dx_tr,     pad + dy_tr),
        (out_w - pad - dx_br,     out_h - pad - dy_br),
        (pad + dx_bl,             out_h - pad - dy_bl),
    ]
    coeffs = _perspective_coeffs(src_corners, dst_corners)

    # Paper-shaped alpha mask
    mask = Image.new("L", (w, h), 255).transform(
        (out_w, out_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=0
    )
    # Warped document
    warped = img.transform(
        (out_w, out_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
        fillcolor=(252, 250, 246),
    )

    # Desk background: warm tan with vertical gradient (top a bit darker)
    desk = Image.new("RGB", (out_w, out_h), (178, 148, 112))
    grad = Image.new("L", (out_w, out_h), 0)
    gd = ImageDraw.Draw(grad)
    for y in range(out_h):
        gd.line([(0, y), (out_w, y)], fill=int(35 * (1 - y / out_h)))
    dark = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    desk = Image.composite(dark, desk, grad)

    # Paper drop-shadow (offset, blurred mask)
    shadow = Image.new("L", (out_w, out_h), 0)
    shadow.paste(mask, (8, 12))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=12))
    shadow_layer = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    desk_with_shadow = Image.composite(shadow_layer, desk, shadow)
    desk_with_shadow = Image.blend(desk, desk_with_shadow, 0.55)

    # Compose paper on desk
    result = desk_with_shadow.copy()
    result.paste(warped, (0, 0), mask)
    return result


# ---------------------------------------------------------------------------
# Coffee stain (realistic, with intensity parameter)
# ---------------------------------------------------------------------------
def _coffee_stain(rng: random.Random, diameter: int,
                  intensity: float = 1.0,
                  aspect: float = 1.0) -> Image.Image:
    """RGBA stain with multi-lobe body + coffee-ring + splashes + optional drip.

    ``intensity`` ~1.0 normal, 1.4-1.6 heavy, 2.0+ heavy with opaque core that
    makes text underneath illegible. 0.7-0.8 light.
    ``aspect`` > 1 stretches the final stain horizontally (smudge / wipe).
    """

    canvas = diameter * 3
    layer = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = cy = canvas // 2
    r = diameter // 2

    # Body: 7-11 overlapping ellipses for organic shape
    n_lobes = rng.randint(7, 11)
    for i in range(n_lobes):
        ang = (360 / n_lobes) * i + rng.uniform(-25, 25)
        off = rng.uniform(0.08, 0.45) * r
        ox = int(math.cos(math.radians(ang)) * off)
        oy = int(math.sin(math.radians(ang)) * off)
        lobe_rx = int(r * rng.uniform(0.55, 1.0))
        lobe_ry = int(r * rng.uniform(0.55, 1.0))
        alpha = min(255, int(rng.randint(60, 95) * intensity))
        draw.ellipse(
            [cx + ox - lobe_rx, cy + oy - lobe_ry,
             cx + ox + lobe_rx, cy + oy + lobe_ry],
            fill=(rng.randint(118, 158), rng.randint(72, 105),
                  rng.randint(35, 55), alpha),
        )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=6))

    # Opaque dark core (only for "kill the line" heavy stains)
    if intensity >= 1.8:
        core = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)
        core_r = int(r * rng.uniform(0.55, 0.70))
        n_core = 5
        for i in range(n_core):
            ang = (360 / n_core) * i + rng.uniform(-30, 30)
            off = rng.uniform(0, core_r * 0.15)
            ox = int(math.cos(math.radians(ang)) * off)
            oy = int(math.sin(math.radians(ang)) * off)
            crx = int(core_r * rng.uniform(0.85, 1.0))
            cry = int(core_r * rng.uniform(0.80, 1.0))
            cd.ellipse(
                [cx + ox - crx, cy + oy - cry,
                 cx + ox + crx, cy + oy + cry],
                fill=(68, 38, 16, 245),     # near-opaque dark coffee
            )
        core = core.filter(ImageFilter.GaussianBlur(radius=3))
        layer = Image.alpha_composite(layer, core)

    # Coffee-ring (concentrated rim)
    ring = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rim_r = int(r * rng.uniform(0.86, 0.95))
    for k in range(110):
        a = (360 / 110) * k + rng.uniform(-3, 3)
        jitter = rng.uniform(-r * 0.03, r * 0.05)
        rr = rim_r + jitter
        px = int(cx + math.cos(math.radians(a)) * rr)
        py = int(cy + math.sin(math.radians(a)) * rr)
        dot_r = rng.randint(2, 6)
        ring_alpha = min(255, int(rng.randint(130, 210) * intensity))
        rd.ellipse(
            [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
            fill=(55, 27, 11, ring_alpha),
        )
    ring = ring.filter(ImageFilter.GaussianBlur(radius=2.2))
    layer = Image.alpha_composite(layer, ring)

    # Splash droplets in halo
    splash = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    sd = ImageDraw.Draw(splash)
    for _ in range(rng.randint(7, 14)):
        a = rng.uniform(0, 360)
        d = rng.uniform(r * 1.05, r * 1.85)
        dx = int(math.cos(math.radians(a)) * d)
        dy = int(math.sin(math.radians(a)) * d)
        dr = rng.randint(2, 8)
        salpha = min(255, int(rng.randint(120, 210) * intensity))
        sd.ellipse(
            [cx + dx - dr, cy + dy - dr, cx + dx + dr, cy + dy + dr],
            fill=(90, 50, 22, salpha),
        )
    splash = splash.filter(ImageFilter.GaussianBlur(radius=1.0))
    layer = Image.alpha_composite(layer, splash)

    # Drip / streak (70% chance)
    if rng.random() < 0.7:
        streak = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        std = ImageDraw.Draw(streak)
        ang = rng.uniform(0, 360)
        length = rng.randint(int(r * 0.5), int(r * 1.5))
        for t in range(length):
            ratio = t / length
            wobble = math.sin(t * 0.12) * (r * 0.05)
            perp_ang = ang + 90
            dx = int(math.cos(math.radians(ang)) * t +
                     math.cos(math.radians(perp_ang)) * wobble)
            dy = int(math.sin(math.radians(ang)) * t +
                     math.sin(math.radians(perp_ang)) * wobble)
            rad = max(1, int(r * 0.17 * (1 - ratio)))
            alpha = min(255, int(95 * (1 - ratio) * intensity))
            std.ellipse(
                [cx + dx - rad, cy + dy - rad,
                 cx + dx + rad, cy + dy + rad],
                fill=(105, 60, 28, alpha),
            )
        streak = streak.filter(ImageFilter.GaussianBlur(radius=2.5))
        layer = Image.alpha_composite(layer, streak)

    # Horizontal stretch for smudge effect (wiping across a line)
    if aspect != 1.0:
        new_w = int(layer.width * aspect)
        layer = layer.resize((new_w, layer.height), Image.BILINEAR)

    return layer


def _paste_stain(img: Image.Image, rng: random.Random,
                 anchor: tuple[int, int], diameter: int,
                 intensity: float = 1.0,
                 aspect: float = 1.0) -> Image.Image:
    stain = _coffee_stain(rng, diameter, intensity, aspect=aspect)
    sw, sh = stain.size
    rgba = img.convert("RGBA")
    rgba.alpha_composite(stain,
                         dest=(anchor[0] - sw // 2, anchor[1] - sh // 2))
    return rgba.convert("RGB")


# ---------------------------------------------------------------------------
# Other paper artefacts
# ---------------------------------------------------------------------------
def _fold_shadow(img: Image.Image, y: int, strength: int = 60) -> Image.Image:
    w, h = img.size
    m = Image.new("L", (w, h), 255)
    ImageDraw.Draw(m).rectangle([0, y, w, y + 2], fill=255 - strength)
    m = m.filter(ImageFilter.GaussianBlur(radius=14))
    return ImageChops.multiply(img.convert("RGB"),
                               Image.merge("RGB", (m, m, m)))


def _yellowing(img: Image.Image, alpha: int = 22) -> Image.Image:
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

    # 1. Stains on flat paper (warp later so they ride with the page)
    for spec in recipe["stains"]:
        ax, ay, dia, intensity = spec[:4]
        aspect = spec[4] if len(spec) > 4 else 1.0
        img = _paste_stain(img, rng, (ax, ay), dia,
                           intensity=intensity, aspect=aspect)

    # 2. Subtle aged-paper tint
    img = _yellowing(img, alpha=22)

    # 3. Fold-shadow on paper
    img = _fold_shadow(img, recipe["fold_y"], strength=60)

    # 4. Perspective warp (phone-photo angle on a desk)
    img = _phone_photo_warp(img, rng)

    # 5. Slight camera blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    # 6. JPEG recompression
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=76)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# (anchor_x, anchor_y, diameter, intensity[, aspect])
# Heavy "kill line" smudges use aspect>1 to wipe across an entire text line
# with an opaque core (intensity >= 1.8). All non-critical lines so the change
# report still comes out correctly while [ILLEGIBLE] fires on the obscured row.
_RECIPES = [
    {
        "src": "01_service_agreement_original.png",
        "dst": "05_service_agreement_original_dirty.png",
        "fold_y": 540,
        "stains": [
            (900, 200, 180, 1.0),               # top-right
            (620, 880, 220, 2.2, 3.5),          # HEAVY SMUDGE over "5. Governing Law" body
            (250, 1050, 220, 1.0),              # bottom-left
            (1050, 700, 150, 0.85),             # right near Confidentiality
            (150, 480, 130, 0.9),               # left near Term
            (1080, 1020, 170, 1.0),             # bottom-right (extra density)
            (450, 320, 110, 0.7),               # top-middle small drip
        ],
    },
    {
        "src": "02_service_agreement_amendment.png",
        "dst": "06_service_agreement_amendment_dirty.png",
        "fold_y": 360,
        "stains": [
            (880, 470, 200, 1.0),               # brushes "30 June 2026"
            (580, 545, 160, 1.85, 2.4),         # HEAVY SMUDGE over "No Other Changes" body
            (200, 220, 160, 0.95),              # top-left
            (1090, 280, 140, 0.8),              # right top
            (350, 130, 90, 0.7),                # tiny near title
            (1050, 620, 130, 0.85),             # bottom right
        ],
    },
]


def main() -> None:
    print("Rendering degraded phone-photo contracts ->", OUT_DIR)
    for recipe in _RECIPES:
        src = OUT_DIR / recipe["src"]
        if not src.exists():
            raise SystemExit(f"Missing source image: {src}")
        out_img = _degrade(src, recipe)
        out = OUT_DIR / recipe["dst"]
        out_img.save(out, format="PNG")
        print(f"  wrote {recipe['dst']}  "
              f"({out_img.size[0]}x{out_img.size[1]})")
    print("Done.")


if __name__ == "__main__":
    main()
