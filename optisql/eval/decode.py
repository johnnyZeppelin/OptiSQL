from __future__ import annotations

from typing import Iterable

from optisql.eval.canonicalize import canonicalize_sql
from optisql.eval.exec_eval_sqlite import execute_with_timeout, compare_results


def eval_exacc(pred_sql: str, gold_sql: str, db_path: str) -> bool:
    pred_result = execute_with_timeout(db_path, pred_sql)
    gold_result = execute_with_timeout(db_path, gold_sql)
    return compare_results(pred_result, gold_result)


def eval_excan(pred_sql: str, gold_sql: str) -> bool:
    return canonicalize_sql(pred_sql) == canonicalize_sql(gold_sql)
