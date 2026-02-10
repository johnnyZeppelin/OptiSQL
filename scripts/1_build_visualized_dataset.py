from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

from optisql.data.spider2_snow import Spider2SnowDataset
from optisql.render.sql_table_extract import extract_tables, pick_single_table
from optisql.render.table_grid import build_table_grid
from optisql.render.html_templates import RenderStyle
from optisql.render.renderer_playwright import render_table_to_image
from optisql.render.augment import build_style_pool
from optisql.utils.io import write_jsonl
from optisql.utils.logging import setup_logging


def build_manifest(data_root: Path, output_root: Path, split: str) -> None:
    logger = setup_logging("build")
    dataset = Spider2SnowDataset(data_root)
    rows = []
    for item in dataset.load_split(split):
        sql = item.get("sql") or item.get("query")
        if not sql:
            continue
        tables = extract_tables(sql)
        table_name = pick_single_table(tables)
        if not table_name:
            continue
        db_path = Path(item["db_path"])
        conn = sqlite3.connect(db_path)
        grid = build_table_grid(conn, table_name)
        conn.close()
        base_style = RenderStyle()
        style_pool = build_style_pool(base_style, size=3)
        img_dir = output_root / "images" / split
        base_path = img_dir / f"{item['id']}_base.png"
        meta = render_table_to_image(grid.headers, grid.rows, base_style, "base", base_path)
        style_paths = []
        for idx, style in enumerate(style_pool):
            style_path = img_dir / f"{item['id']}_style_{idx}.png"
            render_table_to_image(grid.headers, grid.rows, style, f"style_{idx}", style_path)
            style_paths.append(style_path.relative_to(output_root).as_posix())
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "sql": sql,
                "db_path": str(db_path),
                "images": {
                    "base": base_path.relative_to(output_root).as_posix(),
                    "styles": style_paths,
                },
                "render_meta": {
                    "header_bbox": meta.header_bbox,
                    "style_id": meta.style_id,
                    "transposed": meta.transposed,
                },
            }
        )
    write_jsonl(output_root / f"manifest_{split}.jsonl", rows)
    logger.info("Wrote %s items for %s", len(rows), split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    build_manifest(args.data_root, args.output_root, args.split)


if __name__ == "__main__":
    main()
