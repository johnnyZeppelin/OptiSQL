from __future__ import annotations

from typing import Iterable, Optional

import sqlglot


def extract_tables(sql: str) -> list[str]:
    try:
        parsed = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        return []
    tables = sorted({t.name for t in parsed.find_all(sqlglot.exp.Table) if t.name})
    return tables


def pick_single_table(tables: Iterable[str]) -> Optional[str]:
    tables = list(tables)
    if len(tables) == 1:
        return tables[0]
    return None
