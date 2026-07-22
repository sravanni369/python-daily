"""Day 08 — CSV -> JSON wrangler with defensive parsing.

Reads a messy CSV (mixed types, blank cells, bad rows), cleans it,
and writes structured JSON. Practices the unglamorous skills real
pipelines live on: file I/O, type coercion, and exception handling
that skips bad rows without killing the run.
"""

import csv
import io
import json


def coerce(value: str):
    """Best-effort type coercion: int -> float -> stripped string -> None."""
    v = value.strip()
    if v == "":
        return None
    for caster in (int, float):
        try:
            return caster(v)
        except ValueError:
            pass
    return v


def wrangle(csv_text: str, required: list[str]) -> tuple[list[dict], list[str]]:
    """Parse CSV text into clean dicts.

    Rows missing any *required* field are collected as errors, not raised —
    one bad row must never abort a batch load.
    Returns (clean_rows, error_messages).
    """
    clean, errors = [], []
    reader = csv.DictReader(io.StringIO(csv_text))
    for lineno, raw in enumerate(reader, start=2):  # header is line 1
        row = {k.strip(): coerce(v if v is not None else "") for k, v in raw.items()}
        missing = [f for f in required if row.get(f) is None]
        if missing:
            errors.append(f"line {lineno}: missing {', '.join(missing)} -> skipped")
            continue
        clean.append(row)
    return clean, errors


def to_json(rows: list[dict]) -> str:
    """Serialize rows to pretty JSON (stable key order for diffs)."""
    return json.dumps(rows, indent=2, sort_keys=True)


if __name__ == "__main__":
    MESSY_CSV = """member_id,name,age,monthly_premium
101, Asha Rao ,34,220.50
102,Ben Ortiz,,180
 ,Cara Lin,29,205.75
104,Dev Patel,41,
105,Elena Cruz,52,240.00
"""
    rows, errs = wrangle(MESSY_CSV, required=["member_id", "age", "monthly_premium"])

    print(f"clean rows: {len(rows)}")
    for e in errs:
        print("REJECT:", e)

    out = to_json(rows)
    print(out)

    # round-trip check: JSON loads back identical
    assert json.loads(out) == rows
    print("round-trip OK")
