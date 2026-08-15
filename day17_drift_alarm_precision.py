"""Day 17 - A drift monitor is judged on its false alarms, not on whether it fires.

Source: "Practical MLOps: Operationalizing Machine Learning Models", Noah Gift &
Alfredo Deza (O'Reilly, 2021), ch. 6 "Monitoring and Observability", PDF
pp. 200-204 (printed pp. 180-184).

The book runs SageMaker Model Monitor against a churn endpoint. Its auto-suggested
baseline demanded 100% data integrity, so a batch that was 99.717% integral raised a
violation - "violations that are not useful", in the book's words. In the same report
the label column came back 0.0% integral: "a 0% is a critical situation here." Both
land in one undifferentiated list.

The book also asks, about a dataset of swimsuit sales, whether weekly sales dropping
to zero means the data has drifted. It lists unit changes, nulls and seasonality as
causes of drift but never says how a detector tells them apart.

So this file scores detectors on precision and recall against known ground truth.
Two things fall out of the run. The suggested constraint set fires on the harmless
99.7% batch and is silent on a Fahrenheit-to-Celsius unit swap, because completeness,
type and sign checks cannot see a distribution move at all - and its type check is
inert on float columns to begin with, since floats are almost never whole numbers.

The second is a limit rather than a bug. The seasonal batch and the dead-feed batch
are generated identically on purpose, because a winter month and a broken pipeline
leave the same trace. No detector reading only these columns can separate them, so it
must either fire on both and give up precision or stay quiet on both and give up
recall. Both branches are computed below rather than argued.

Standard library only. No pip, no network. Seeded, so the numbers repeat.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

Row = Dict[str, str]          # raw cells, as they arrive from CSV
Profile = Dict[str, dict]     # column -> stats
Violation = Tuple[str, str]   # (column, what tripped)

COLUMNS = ("account_length", "temp_f", "sales", "churn")


def make_batch(n: int, seed: int, *, junk_cells: int = 0, celsius: bool = False,
               churn_as_text: bool = False, sales_zero: bool = False) -> List[Row]:
    """One hour of traffic. Every cell is a string, exactly as a CSV would hand it over."""
    rng = random.Random(seed)
    rows: List[Row] = []
    for _ in range(n):
        temp_f = rng.gauss(72.0, 4.0)
        rows.append({
            "account_length": str(rng.randint(1, 240)),
            "temp_f": f"{(temp_f - 32) * 5 / 9:.2f}" if celsius else f"{temp_f:.2f}",
            "sales": "0" if sales_zero else str(rng.randint(0, 40)),
            "churn": rng.choice(["true", "false"]) if churn_as_text else str(rng.randint(0, 1)),
        })
    for i in range(junk_cells):            # a handful of cells arrive as text
        rows[i * 97 % n]["account_length"] = "N/A"
    return rows


def profile(rows: Sequence[Row], columns: Sequence[str] = COLUMNS) -> Profile:
    """Per-column stats. An all-blank column reports None, it does not divide by zero."""
    out: Profile = {}
    for col in columns:
        cells = [r.get(col, "").strip() for r in rows]
        filled = [c for c in cells if c]
        numeric = []
        integral = 0
        for c in filled:
            try:
                numeric.append(float(c))
                integral += float(c).is_integer()
            except ValueError:
                pass
        out[col] = {
            "n": len(cells),
            "completeness": len(filled) / len(cells) if cells else 0.0,
            "integral": integral / len(filled) if filled else 0.0,
            "mean": sum(numeric) / len(numeric) if numeric else None,
            "min": min(numeric) if numeric else None,
        }
    return out


def strict_violations(base: Profile, cur: Profile) -> List[Violation]:
    """The book's auto-suggested constraints: completeness, type and sign at 100%."""
    found = []
    for col, b in base.items():
        c = cur[col]
        if c["completeness"] < 1.0:
            found.append((col, f"completeness {c['completeness']:.3%} < 100%"))
        if b["integral"] == 1.0 and c["integral"] < 1.0:
            found.append((col, f"only {c['integral']:.3%} of data is Integral"))
        if b["min"] is not None and b["min"] >= 0 and (c["min"] or 0) < 0:
            found.append((col, "is_non_negative violated"))
    return found


def tuned_violations(base: Profile, cur: Profile, tol: float = 0.99,
                     mean_shift: float = 0.40) -> List[Violation]:
    """The same checks with the fine-tuning the book says a baseline needs."""
    found = []
    for col, b in base.items():
        c = cur[col]
        if c["completeness"] < tol:
            found.append((col, f"completeness {c['completeness']:.1%}"))
        if b["integral"] == 1.0 and c["integral"] < tol:
            found.append((col, f"only {c['integral']:.1%} of data is Integral"))
        if b["min"] is not None and b["min"] >= 0 and (c["min"] or 0) < 0:
            found.append((col, "is_non_negative violated"))
        if b["mean"] not in (None, 0) and c["mean"] is not None:
            shift = abs(c["mean"] - b["mean"]) / abs(b["mean"])
            if shift > mean_shift:
                found.append((col, f"mean moved {shift:.0%} ({b['mean']:.1f} -> {c['mean']:.1f})"))
    return found


def score(fired: Sequence[bool], truth: Sequence[bool]) -> Tuple[int, float, float]:
    alarms = sum(fired)
    hits = sum(f and t for f, t in zip(fired, truth))
    real = sum(truth)
    return alarms, (hits / alarms if alarms else 0.0), (hits / real if real else 0.0)


if __name__ == "__main__":
    N = 1000
    base = profile(make_batch(N, seed=17))

    # (label, batch, is this a real incident?)
    hours = [
        ("clean traffic",                make_batch(N, 21), False),
        ("3 junk cells in 1,000",        make_batch(N, 22, junk_cells=3), False),
        ("temp switched F -> C",         make_batch(N, 23, celsius=True), True),
        ("churn arrives as true/false",  make_batch(N, 24, churn_as_text=True), True),
        ("winter: sales are zero",       make_batch(N, 25, sales_zero=True), False),
        ("feed died: sales are zero",    make_batch(N, 25, sales_zero=True), True),
    ]

    print("Baseline profile from 1,000 clean rows:")
    for col, s in base.items():
        print(f"  {col:<15} completeness {s['completeness']:.1%}  integral {s['integral']:.1%}"
              f"  mean {s['mean']:.2f}")

    print("  (temp_f is only 1.5% integral even when healthy - floats are rarely whole,")
    print("   so the suggested type check is inert on that column from the start.)")

    print(f"\n  {'hour':<28}{'real?':<7}{'strict':<8}{'tuned':<8} first strict violation")
    strict_fired, tuned_fired, truth = [], [], []
    for label, rows, is_incident in hours:
        cur = profile(rows)
        sv, tv = strict_violations(base, cur), tuned_violations(base, cur)
        strict_fired.append(bool(sv))
        tuned_fired.append(bool(tv))
        truth.append(is_incident)
        detail = f"{sv[0][0]}: {sv[0][1]}" if sv else "-"
        print(f"  {label:<28}{'YES' if is_incident else 'no':<7}"
              f"{('FIRE' if sv else 'quiet'):<8}{('FIRE' if tv else 'quiet'):<8} {detail}")

    print("\nScored against ground truth:")
    for name, fired in (("strict (book's suggested baseline)", strict_fired),
                        ("tuned  (99% + mean-shift)", tuned_fired)):
        alarms, prec, rec = score(fired, truth)
        print(f"  {name:<36} {alarms} alarms   precision {prec:.2f}   recall {rec:.2f}")

    winter, dead = profile(hours[4][1]), profile(hours[5][1])
    assert winter == dead, "the two zero-sales batches must be indistinguishable"
    print("\nWhy the last false alarm is not a threshold problem:")
    print(f"  the winter batch and the dead-feed batch profile identically - sales")
    print(f"  completeness {winter['sales']['completeness']:.0%}, mean {winter['sales']['mean']:.1f}, "
          f"integral {winter['sales']['integral']:.0%} in both. One is normal, one is")
    print("  an outage. A detector reading only these columns has two options:")
    for choice in (True, False):
        hypothetical = [f if i < 4 else choice for i, f in enumerate(tuned_fired)]
        alarms, prec, rec = score(hypothetical, truth)
        verb = "fire on both" if choice else "stay quiet on both"
        print(f"    {verb:<20} -> {alarms} alarms, precision {prec:.2f}, recall {rec:.2f}")
    print("  Neither reaches 1.00 on both. The trade is forced by the data, not by the")
    print("  threshold, and closing it needs a calendar or a feed heartbeat instead.")

    print("\nFAILURE - a column that is entirely blank, and an empty batch:")
    blank = profile([{"account_length": "", "temp_f": "", "sales": "", "churn": ""}] * 5)
    print(f"  all-blank column -> completeness {blank['sales']['completeness']:.0%}, "
          f"mean {blank['sales']['mean']}, no ZeroDivisionError")
    print(f"  empty batch      -> {profile([])['sales']}")
