"""Day 13 — Min-max scaling, and the two ways it quietly ruins a feature.

Source: "Low-Code AI: A Practical Project-Driven Introduction to Machine
Learning", Gwendolyn Stripling & Michael Abel (O'Reilly), PDF p. 24.

The book warns: "This min-max normalization example can have detrimental
effects if there are outliers. For example, when scaling to [0,1], it
essentially maps the outlier to 1 and squashes all other values to 0."

Two failure modes handled here:
  1. A constant column has range 0, so the textbook formula divides by zero.
     Returning 0.5 keeps the pipeline alive and carries no false signal.
  2. One outlier collapses every real value toward 0 — the book's warning.
     Median/IQR scaling is the standard answer, so both are shown side by side.

Standard library only. No pip, no network.
"""

from typing import List


def min_max_scale(values: List[float]) -> List[float]:
    """Scale to [0, 1]. A constant column returns 0.5 instead of dividing by zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def robust_scale(values: List[float]) -> List[float]:
    """Centre on the median, divide by the IQR. Outliers move, the bulk does not."""
    if not values:
        return []
    ordered = sorted(values)
    q1, med, q3 = (ordered[len(ordered) // 4], ordered[len(ordered) // 2],
                   ordered[(3 * len(ordered)) // 4])
    spread = q3 - q1
    if spread == 0:
        return [0.0] * len(values)
    return [(v - med) / spread for v in values]


def _show(label: str, values: List[float]) -> None:
    print(f"  {label:<14}", "[" + ", ".join(f"{v:6.2f}" for v in values) + "]")


if __name__ == "__main__":
    clean = [12.0, 15.0, 14.0, 13.0, 16.0]
    with_outlier = clean + [900.0]          # one bad sensor reading
    constant = [7.0, 7.0, 7.0, 7.0]         # a column that never varies

    print("1. Well-behaved column — min-max works fine")
    _show("raw", clean)
    _show("min-max", min_max_scale(clean))

    print("\n2. FAILURE the book warns about — one outlier squashes the rest")
    _show("raw", with_outlier)
    _show("min-max", min_max_scale(with_outlier))
    _show("robust", robust_scale(with_outlier))
    squashed = [v for v in min_max_scale(with_outlier) if v < 0.01]
    print(f"  -> {len(squashed)} of {len(with_outlier)} real values crushed below 0.01")

    print("\n3. FAILURE the formula hides — constant column, range is zero")
    _show("raw", constant)
    _show("min-max", min_max_scale(constant))
    print("  -> returned 0.5 each instead of raising ZeroDivisionError")

    print("\n4. Empty input")
    print(f"  min-max([]) = {min_max_scale([])}, robust([]) = {robust_scale([])}")
