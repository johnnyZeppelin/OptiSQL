from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

from optisql.render.html_templates import RenderStyle, render_table_html


@dataclass
class RenderMeta:
    header_bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    style_id: str
    transposed: bool


def render_table_to_image(
    headers: list[str],
    rows: list[list[str]],
    style: RenderStyle,
    style_id: str,
    output_path: Path,
    transposed: bool = False,
) -> RenderMeta:
    html = render_table_html(headers, rows, style)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.set_content(html, wait_until="domcontentloaded")
        table = page.query_selector("#optisql-table")
        if table is None:
            raise RuntimeError("Table element not found for rendering.")
        bbox = table.bounding_box()
        if bbox is None:
            raise RuntimeError("Failed to get bounding box for table.")
        header = page.query_selector("#optisql-table thead")
        header_bbox = header.bounding_box() if header else bbox
        clip = {
            "x": bbox["x"],
            "y": bbox["y"],
            "width": bbox["width"],
            "height": bbox["height"],
        }
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
