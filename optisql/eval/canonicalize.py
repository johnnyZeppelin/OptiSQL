from __future__ import annotations

import re
import sqlglot


def canonicalize_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    try:
        parsed = sqlglot.parse_one(cleaned)
    except sqlglot.errors.ParseError:
        return normalize_sql(cleaned)
    return normalize_sql(parsed.sql(dialect="sqlite"))


def normalize_sql(sql: str) -> str:
    sql = sql.lower()
    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\s*([=<>])\s*", r"\1", sql)
    return sql
