"""Day 01 - Data Quality Checker.

Run column-level quality checks on a CSV: nulls, duplicates, range
violations, and regex format checks. Prints a small quality report.
"""

import csv
import re
from collections import Counter


def load_rows(path):
    """Read a CSV into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_nulls(rows, columns):
    """Return {column: null_count} for empty/missing values."""
    report = {}
    for col in columns:
        report[col] = sum(1 for r in rows if not (r.get(col) or "").strip())
    return report


def check_duplicates(rows, key_columns):
    """Count rows whose key-column combination appears more than once."""
    keys = [tuple((r.get(c) or "").strip() for c in key_columns) for r in rows]
    counts = Counter(keys)
    return sum(n for n in counts.values() if n > 1)


def check_range(rows, column, lo, hi):
    """Count numeric values outside [lo, hi]; non-numeric counts as violation."""
    bad = 0
    for r in rows:
        raw = (r.get(column) or "").strip()
        try:
            val = float(raw)
        except ValueError:
            bad += 1
            continue
        if val < lo or val > hi:
            bad += 1
    return bad


def check_format(rows, column, pattern):
    """Count values not matching a regex pattern."""
    rx = re.compile(pattern)
    return sum(1 for r in rows if not rx.fullmatch((r.get(column) or "").strip()))


def quality_report(rows, config):
    """Run all configured checks and return a dict report."""
    report = {"row_count": len(rows)}
    report["nulls"] = check_nulls(rows, config.get("not_null", []))
    if config.get("unique_key"):
        report["duplicate_rows"] = check_duplicates(rows, config["unique_key"])
    for col, (lo, hi) in config.get("ranges", {}).items():
        report[f"range_violations_{col}"] = check_range(rows, col, lo, hi)
    for col, pattern in config.get("formats", {}).items():
        report[f"format_violations_{col}"] = check_format(rows, col, pattern)
    return report


if __name__ == "__main__":
    sample = [
        {"id": "1", "email": "a@x.com", "age": "34"},
        {"id": "2", "email": "bad-email", "age": "212"},
        {"id": "2", "email": "c@x.com", "age": ""},
    ]
    cfg = {
        "not_null": ["id", "email", "age"],
        "unique_key": ["id"],
        "ranges": {"age": (0, 120)},
        "formats": {"email": r"[^@\s]+@[^@\s]+\.[^@\s]+"},
    }
    for k, v in quality_report(sample, cfg).items():
        print(f"{k}: {v}")
