from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from optisql.render.html_templates import RenderStyle, render_table_html


@dataclass
class RenderMeta:
    header_bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    style_id: str
    transposed: bool


def _render_with_pil(headers: list[str], rows: list[list[str]], output_path: Path, style_id: str, transposed: bool) -> RenderMeta:
    cell_w = 180
    cell_h = 36
    cols = max(len(headers), 1)
    rows_n = len(rows) + 1
    width = cols * cell_w + 2
    height = rows_n * cell_h + 2

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for r in range(rows_n):
        for c in range(cols):
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            fill = (242, 242, 242) if r == 0 else ((250, 250, 250) if r % 2 == 0 else (255, 255, 255))
            draw.rectangle([x0, y0, x1, y1], outline=(51, 51, 51), fill=fill)
            text = headers[c] if r == 0 else (rows[r - 1][c] if c < len(rows[r - 1]) else "")
            draw.text((x0 + 6, y0 + 10), str(text), fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return RenderMeta(
        header_bbox=(0.0, 0.0, float(width), float(cell_h)),
        image_width=width,
        image_height=height,
        style_id=style_id,
        transposed=transposed,
    )


def render_table_to_image(
    headers: list[str],
    rows: list[list[str]],
    style: RenderStyle,
    style_id: str,
    output_path: Path,
    transposed: bool = False,
) -> RenderMeta:
    # Keep HTML generation to preserve compatibility with browser rendering path.
    _ = render_table_html(headers, rows, style)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.set_content(render_table_html(headers, rows, style), wait_until="domcontentloaded")
            table = page.query_selector("#optisql-table")
            if table is None:
                raise RuntimeError("Table element not found for rendering.")
            bbox = table.bounding_box()
            if bbox is None:
                raise RuntimeError("Failed to get bounding box for table.")
            header = page.query_selector("#optisql-table thead")
            header_bbox = header.bounding_box() if header else bbox
            if header_bbox is None:
                header_bbox = bbox
            clip = {
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(output_path), clip=clip, omit_background=False)
            browser.close()

        return RenderMeta(
            header_bbox=(
                header_bbox["x"],
                header_bbox["y"],
                header_bbox["width"],
                header_bbox["height"],
            ),
            image_width=int(bbox["width"]),
            image_height=int(bbox["height"]),
            style_id=style_id,
            transposed=transposed,
        )
    except Exception:
        # Fallback keeps pipeline runnable without Playwright/Chromium.
        return _render_with_pil(headers, rows, output_path, style_id, transposed)
