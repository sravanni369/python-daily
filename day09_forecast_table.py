"""Day 09 - National forecast table parser.

Newspaper weather tables pack three facts into one token: "98/74/t" is a
high of 98F, a low of 74F, and a sky code (t = thunderstorms). Each city
carries today's forecast and the next day's. This parses that grid, guards
the messy tokens instead of crashing on them, and answers the questions a
reader actually asks: who is hottest, where is the sky turning, and which
city swings most from today to tomorrow.

Standard library only. Run: python day09_forecast_table.py
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Sky codes as printed in the paper's legend.
SKY = {
    "s": "sunny", "pc": "partly cloudy", "c": "cloudy", "sh": "showers",
    "t": "t-storms", "r": "rain", "sn": "snow", "f": "flurries",
    "i": "ice", "w": "windy", "rs": "rain/snow",
}

# One token like "116/95/s": high / low / sky. Sky is letters only.
CELL = re.compile(r"^(-?\d{1,3})/(-?\d{1,3})/([a-z]{1,2})$")


@dataclass(frozen=True)
class Forecast:
    hi: int
    lo: int
    sky: str

    @property
    def spread(self) -> int:
        return self.hi - self.lo


@dataclass(frozen=True)
class Row:
    city: str
    today: Forecast
    next_day: Forecast

    @property
    def hi_change(self) -> int:
        return self.next_day.hi - self.today.hi


def parse_cell(token: str) -> Forecast | None:
    """Turn 'hi/lo/sky' into a Forecast, or None if the token is malformed."""
    m = CELL.match(token.strip().lower())
    if not m:
        return None
    hi, lo, sky = int(m.group(1)), int(m.group(2)), m.group(3)
    if sky not in SKY or lo > hi:
        return None
    return Forecast(hi, lo, sky)


def parse_row(line: str) -> Row | None:
    """A line is 'City Name TODAY NEXT'. City may be several words."""
    parts = line.split()
    if len(parts) < 3:
        return None
    today, nxt = parse_cell(parts[-2]), parse_cell(parts[-1])
    city = " ".join(parts[:-2]).strip()
    if today is None or nxt is None or not city:
        return None
    return Row(city, today, nxt)


def parse_table(text: str) -> tuple[list[Row], list[str]]:
    """Return (good rows, rejected raw lines) so nothing fails silently."""
    rows, bad = [], []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        row = parse_row(line)
        (rows.append(row) if row else bad.append(line.strip()))
    return rows, bad


def report(rows: list[Row], bad: list[str]) -> None:
    if not rows:
        print("no valid rows"); return

    hottest = max(rows, key=lambda r: r.today.hi)
    coolest = min(rows, key=lambda r: r.today.hi)
    widest = max(rows, key=lambda r: r.today.spread)
    warming = max(rows, key=lambda r: r.hi_change)
    cooling = min(rows, key=lambda r: r.hi_change)
    avg_hi = sum(r.today.hi for r in rows) / len(rows)
    skies = Counter(SKY[r.today.sky] for r in rows)

    print(f"cities parsed:   {len(rows)}   (rejected {len(bad)})")
    print(f"hottest today:   {hottest.city} {hottest.today.hi}")
    print(f"coolest today:   {coolest.city} {coolest.today.hi}")
    print(f"average high:    {avg_hi:.1f}")
    print(f"widest hi-lo:    {widest.city} {widest.today.spread} "
          f"({widest.today.hi}/{widest.today.lo})")
    print(f"warming most:    {warming.city} {warming.hi_change:+d} tomorrow")
    print(f"cooling most:    {cooling.city} {cooling.hi_change:+d} tomorrow")
    print("sky today:      ", ", ".join(f"{k} {v}" for k, v in skies.most_common()))
    if bad:
        print("rejected lines: ", "; ".join(bad))


# A slice of a real national forecast grid: City TODAY NEXT (hi/lo/sky).
SAMPLE = """
Phoenix 116/95/s 118/92/s
Las Vegas 111/91/t 114/92/s
Salt Lake City 100/78/t 103/78/pc
Albuquerque 98/74/t 93/72/t
Dallas 97/80/s 101/81/pc
Denver 95/71/t 98/69/pc
Minneapolis 80/69/s 95/74/s
Miami 91/80/t 94/79/sh
New Orleans 94/77/t 94/79/sh
Chicago 81/66/s 82/71/pc
Boston 79/64/s 77/61/s
New York City 82/66/s 82/67/s
Seattle 87/62/s 74/60/pc
San Francisco 79/60/pc 73/61/pc
Anchorage 67/51/pc 68/53/s
Honolulu 88/75/sh 88/75/sh
Nashville 86/70/pc 83/74/t
Bad Row 999/xx/z 12/9/q
"""

if __name__ == "__main__":
    rows, bad = parse_table(SAMPLE)
    report(rows, bad)
