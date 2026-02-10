from __future__ import annotations

import re
import sqlglot
from sqlglot import exp


def canonicalize_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    try:
        parsed = sqlglot.parse_one(cleaned)
    except sqlglot.errors.ParseError:
        return normalize_sql(cleaned)
    parsed = reorder_flat_conditions(parsed)
    return normalize_sql(parsed.sql(dialect="sqlite"))


def normalize_sql(sql: str) -> str:
    sql = sql.lower()
    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\s*([=<>])\s*", r"\1", sql)
    return sql


def reorder_flat_conditions(parsed: exp.Expression) -> exp.Expression:
    def reorder(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.And):
            flat = list(node.flatten())
            ordered = sorted((reorder(child) for child in flat), key=lambda c: c.sql())
            return exp.and_(*ordered)
        if isinstance(node, exp.Or):
            flat = list(node.flatten())
            ordered = sorted((reorder(child) for child in flat), key=lambda c: c.sql())
            return exp.or_(*ordered)
        return node

    for where in parsed.find_all(exp.Where):
        if where.this is not None:
            where.set("this", reorder(where.this))
    return parsed
