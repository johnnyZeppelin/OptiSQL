from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlite3


@dataclass
class TableColumn:
    name: str
    col_type: str


@dataclass
class TableGrid:
    headers: list[str]
    rows: list[list[str]]
    row_index: bool = False


def load_table_schema(conn: sqlite3.Connection, table_name: str) -> list[TableColumn]:
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    cols = []
    for _, name, col_type, *_ in cursor.fetchall():
        cols.append(TableColumn(name=name, col_type=col_type or ""))
    return cols


def load_table_rows(conn: sqlite3.Connection, table_name: str, max_rows: int = 1000) -> list[list[Any]]:
    cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT {max_rows}")
    return [list(row) for row in cursor.fetchall()]


def format_cell(value: Any, col_type: str) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_table_grid(
    conn: sqlite3.Connection,
    table_name: str,
    max_rows: int = 1000,
    include_row_index: bool = False,
) -> TableGrid:
    schema = load_table_schema(conn, table_name)
    rows = load_table_rows(conn, table_name, max_rows)
    headers = [col.name for col in schema]
    if include_row_index:
        headers = ["row_id"] + headers
    formatted_rows: list[list[str]] = []
    for idx, row in enumerate(rows, start=1):
        formatted = [format_cell(value, schema[i].col_type) for i, value in enumerate(row)]
        if include_row_index:
            formatted = [str(idx)] + formatted
        formatted_rows.append(formatted)
    return TableGrid(headers=headers, rows=formatted_rows, row_index=include_row_index)
