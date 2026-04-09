"""Generate the HF Space card thumbnail for the GST Invoice Compliance Checker.

The thumbnail is a 1200x630 PNG (HF Space card aspect ratio) showing:
  - Project name
  - One-line tagline
  - A stylized invoice with red-highlighted violation fields
  - Score badge

Run:
    uv run python scripts/generate_thumbnail.py

Output:
    docs/thumbnail.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "thumbnail.png"
WIDTH, HEIGHT = 1200, 630

# Brand palette (matches openenv.yaml colorFrom=blue colorTo=green)
BG_TOP = (12, 26, 56)        # deep navy
BG_BOTTOM = (24, 64, 92)     # teal
ACCENT_GREEN = (76, 222, 128)
ACCENT_RED = (248, 113, 113)
TEXT_WHITE = (245, 247, 250)
TEXT_DIM = (170, 188, 214)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (210, 220, 235)
ROW_ALT = (244, 247, 252)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Best-effort font loader. Falls back through common system fonts."""
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    # Last resort — Pillow's default bitmap font (no scaling, but works).
    return ImageFont.load_default()


def _vertical_gradient(size: tuple[int, int], top: tuple, bottom: tuple) -> Image.Image:
    """Build a top-to-bottom vertical gradient image."""
    w, h = size
    grad = Image.new("RGB", (1, h))
    for y in range(h):
        ratio = y / max(h - 1, 1)
        r = int(top[0] + (bottom[0] - top[0]) * ratio)
        g = int(top[1] + (bottom[1] - top[1]) * ratio)
        b = int(top[2] + (bottom[2] - top[2]) * ratio)
        grad.putpixel((0, y), (r, g, b))
    return grad.resize(size)


def _draw_pill(draw: ImageDraw.ImageDraw, xy: tuple, fill, text: str, font, text_color):
    x1, y1, x2, y2 = xy
    radius = (y2 - y1) // 2
    draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=fill)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((x1 + x2 - tw) // 2, (y1 + y2 - th) // 2 - 2),
        text,
        font=font,
        fill=text_color,
    )


def render() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    img = _vertical_gradient((WIDTH, HEIGHT), BG_TOP, BG_BOTTOM)
    draw = ImageDraw.Draw(img)

    # Subtle decorative grid
    for x in range(0, WIDTH, 60):
        draw.line([(x, 0), (x, HEIGHT)], fill=(255, 255, 255, 6), width=1)

    # ── Header band ──────────────────────────────────────────────────────
    title_font = _load_font(50, bold=True)
    tagline_font = _load_font(24)
    badge_font = _load_font(20, bold=True)

    draw.text((60, 50), "GST Invoice Compliance Checker", font=title_font, fill=TEXT_WHITE)
    draw.text(
        (60, 116),
        "An OpenEnv benchmark for Indian GST audit reasoning",
        font=tagline_font,
        fill=TEXT_DIM,
    )

    # Two pill badges in a row, under the tagline
    _draw_pill(
        draw,
        (60, 162, 280, 204),
        (255, 255, 255),
        "Avg score: 0.9911",
        badge_font,
        (16, 36, 70),
    )
    _draw_pill(
        draw,
        (296, 162, 460, 204),
        ACCENT_GREEN,
        "10 tasks",
        badge_font,
        (8, 36, 24),
    )
    _draw_pill(
        draw,
        (476, 162, 780, 204),
        (255, 255, 255),
        "Hybrid grader: rules + LLM",
        badge_font,
        (16, 36, 70),
    )

    # ── Mock invoice panel ──────────────────────────────────────────────
    panel_x1, panel_y1 = 60, 230
    panel_x2, panel_y2 = WIDTH - 60, HEIGHT - 50
    draw.rounded_rectangle(
        (panel_x1, panel_y1, panel_x2, panel_y2),
        radius=18,
        fill=PANEL_BG,
        outline=PANEL_BORDER,
        width=2,
    )

    panel_title_font = _load_font(26, bold=True)
    row_label_font = _load_font(20, bold=True)
    row_value_font = _load_font(20)
    flag_font = _load_font(16, bold=True)

    draw.text(
        (panel_x1 + 30, panel_y1 + 22),
        "Invoice INV-H4-003 (hard_4 sample)",
        font=panel_title_font,
        fill=(28, 40, 70),
    )
    draw.text(
        (panel_x2 - 280, panel_y1 + 30),
        "Real issues: 1",
        font=tagline_font,
        fill=(60, 90, 130),
    )

    rows = [
        ("Supplier",          "Pune Office Furniture",                       False),
        ("Supplier GSTIN",    "27AABCO5555V1Z6",                              False),
        ("Recipient GSTIN",   "07BCDEM6666W1Z4",                              True),
        ("Recipient State",   "27 (state code does not match GSTIN prefix 07)", True),
        ("HSN / Rate / Tax",  "9403 / 18% / CGST+SGST    Total: INR 1,41,600", False),
    ]

    row_y = panel_y1 + 76
    row_h = 44
    for i, (label, value, is_flag) in enumerate(rows):
        if i % 2 == 1:
            draw.rounded_rectangle(
                (panel_x1 + 20, row_y, panel_x2 - 20, row_y + row_h),
                radius=6,
                fill=ROW_ALT,
            )
        draw.text((panel_x1 + 36, row_y + 12), label, font=row_label_font, fill=(60, 80, 110))
        draw.text((panel_x1 + 260, row_y + 12), value, font=row_value_font, fill=(28, 40, 70))
        if is_flag:
            badge_x = panel_x2 - 140
            _draw_pill(
                draw,
                (badge_x, row_y + 8, badge_x + 110, row_y + 36),
                ACCENT_RED,
                "VIOLATION",
                flag_font,
                (255, 255, 255),
            )
        row_y += row_h

    img.save(OUT_PATH, "PNG", optimize=True)
    return OUT_PATH


if __name__ == "__main__":
    out = render()
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
