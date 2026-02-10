from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any
import sqlite3


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple[Any, ...]] | None
    error: str | None = None


def _run_query(db_path: str, sql: str, queue: Queue) -> None:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute(sql)
        rows = cur.fetchall()
        queue.put(ExecResult(ok=True, rows=rows))
    except Exception as exc:  # noqa: BLE001
        queue.put(ExecResult(ok=False, rows=None, error=str(exc)))
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def execute_with_timeout(db_path: str, sql: str, timeout_s: float = 10.0) -> ExecResult:
    queue: Queue = Queue()
    proc = Process(target=_run_query, args=(db_path, sql, queue))
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ExecResult(ok=False, rows=None, error="timeout")
    return queue.get() if not queue.empty() else ExecResult(ok=False, rows=None, error="no_result")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def compare_results(result_a: ExecResult, result_b: ExecResult) -> bool:
    if not result_a.ok or not result_b.ok:
        return False
    rows_a = sorted(tuple(_normalize_value(v) for v in row) for row in (result_a.rows or []))
    rows_b = sorted(tuple(_normalize_value(v) for v in row) for row in (result_b.rows or []))
    return rows_a == rows_b
