from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderStyle:
    font_family: str = "Arial"
    font_size: int = 16
    header_bg: str = "#f2f2f2"
    stripe_bg: str = "#fafafa"
    border_color: str = "#333"
    padding_px: int = 6
    header_bold: bool = True


def render_table_html(headers: list[str], rows: list[list[str]], style: RenderStyle) -> str:
    header_cells = "".join(
        f"<th>{header}</th>" for header in headers
    )
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_rows)
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
body {{ background: white; margin: 0; padding: 12px; }}
table {{ border-collapse: collapse; font-family: {style.font_family}; font-size: {style.font_size}px; }}
th, td {{ border: 1px solid {style.border_color}; padding: {style.padding_px}px; }}
th {{ background: {style.header_bg}; font-weight: {'bold' if style.header_bold else 'normal'}; }}
tr:nth-child(even) td {{ background: {style.stripe_bg}; }}
</style>
</head>
<body>
<table id="optisql-table">
<thead><tr>{header_cells}</tr></thead>
<tbody>{body_html}</tbody>
</table>
</body>
</html>
"""
