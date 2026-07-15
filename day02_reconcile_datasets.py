"""Day 02 - Dataset Reconciliation.

Compare a source extract against a target table and report what drifted:
keys missing on either side, duplicate keys, and field-level mismatches.
Prints a small reconciliation report.
"""

import csv
from collections import Counter


def load_rows(path):
    """Read a CSV into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_by_key(rows, key_columns):
    """Return {key_tuple: row}. Last row wins on duplicate keys."""
    return {
        tuple((r.get(c) or "").strip() for c in key_columns): r
        for r in rows
    }


def duplicate_keys(rows, key_columns):
    """Return keys appearing more than once, with their counts."""
    keys = [tuple((r.get(c) or "").strip() for c in key_columns) for r in rows]
    return {k: n for k, n in Counter(keys).items() if n > 1}


def compare_row(src, tgt, compare_columns):
    """Return {column: (source_value, target_value)} for differing columns."""
    diffs = {}
    for col in compare_columns:
        a = (src.get(col) or "").strip()
        b = (tgt.get(col) or "").strip()
        if a != b:
            diffs[col] = (a, b)
    return diffs


def reconcile(source_rows, target_rows, key_columns, compare_columns):
    """Reconcile two datasets on a key and return a dict report."""
    src = index_by_key(source_rows, key_columns)
    tgt = index_by_key(target_rows, key_columns)

    mismatched = {}
    for key in src.keys() & tgt.keys():
        diffs = compare_row(src[key], tgt[key], compare_columns)
        if diffs:
            mismatched[key] = diffs

    matched = len(src.keys() & tgt.keys())
    return {
        "source_rows": len(source_rows),
        "target_rows": len(target_rows),
        "missing_in_target": sorted(src.keys() - tgt.keys()),
        "missing_in_source": sorted(tgt.keys() - src.keys()),
        "duplicate_keys_source": duplicate_keys(source_rows, key_columns),
        "duplicate_keys_target": duplicate_keys(target_rows, key_columns),
        "mismatched": mismatched,
        "match_rate": round(100 * (matched - len(mismatched)) / len(src), 1) if src else 0.0,
    }


if __name__ == "__main__":
    source = [
        {"claim_id": "C1", "member_id": "M10", "amount": "250.00", "status": "PAID"},
        {"claim_id": "C2", "member_id": "M11", "amount": "80.00", "status": "PAID"},
        {"claim_id": "C3", "member_id": "M12", "amount": "310.50", "status": "DENIED"},
        {"claim_id": "C3", "member_id": "M12", "amount": "310.50", "status": "DENIED"},
    ]
    target = [
        {"claim_id": "C1", "member_id": "M10", "amount": "250.00", "status": "PAID"},
        {"claim_id": "C2", "member_id": "M11", "amount": "80.00", "status": "PENDING"},
        {"claim_id": "C4", "member_id": "M13", "amount": "45.00", "status": "PAID"},
    ]

    report = reconcile(
        source,
        target,
        key_columns=["claim_id"],
        compare_columns=["member_id", "amount", "status"],
    )
    for k, v in report.items():
        print(f"{k}: {v}")
