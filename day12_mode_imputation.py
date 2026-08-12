"""Day 12 - Mode Imputation for Missing Categorical Values.

From Low-Code AI (Stripling & Abel, O'Reilly), ch. 2: the two ways to handle
missing data are deletion and imputation, and for a categorical column the
usual fill is the mode, the most frequently occurring value. Real CSVs spell
"missing" a dozen ways, and ties must break deterministically or two runs on
the same data disagree.
"""

from collections import Counter

MISSING = {None, "", "NA", "N/A", "null", "NULL", "-", "?"}


def mode(values):
    """Most frequent non-missing value. Ties break alphabetically for stability."""
    counts = Counter(v for v in values if v not in MISSING)
    if not counts:
        return None
    top = max(counts.values())
    return sorted(k for k, n in counts.items() if n == top)[0]


def impute(rows, column):
    """Fill missing cells in one column with its mode. Returns (rows, fill, count)."""
    fill = mode(row.get(column) for row in rows)
    out, filled = [], 0
    for row in rows:
        row = dict(row)
        if row.get(column) in MISSING:
            row[column], filled = fill, filled + 1
        out.append(row)
    return out, fill, filled


if __name__ == "__main__":
    data = [{"city": "Hyderabad"}, {"city": ""}, {"city": "Chennai"}, {"city": "?"},
            {"city": "Hyderabad"}, {"city": "NA"}, {"city": "Chennai"},
            {"city": "Hyderabad"}, {"city": None}]
    out, fill, n = impute(data, "city")
    print(f"mode = {fill!r}, filled {n} of {len(data)} rows")
    print("after:", dict(Counter(r["city"] for r in out)))
    print("all-missing column ->", impute([{"city": ""}, {"city": "NA"}], "city")[1])
