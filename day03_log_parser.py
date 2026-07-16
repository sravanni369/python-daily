"""Day 03 - Log Parser.

Parse a stream of web-server-style log lines, extract structured fields with a
single regex, and roll them up: request counts, error rate, slowest endpoints,
and top talkers. Prints a small traffic report.
"""

import re
from collections import Counter, defaultdict

# Common Log Format-ish line, plus response time in ms at the end:
#   1.2.3.4 - - [10/Jul/2026:13:55:36] "GET /api/users HTTP/1.1" 200 342 18
LINE = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+) '
    r'\S+ \S+ '
    r'\[(?P<ts>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<path>\S+) [^"]*" '
    r'(?P<status>\d{3}) '
    r'(?P<bytes>\d+|-) '
    r'(?P<ms>\d+)'
)


def parse_line(line):
    """Parse one log line into a dict, or None if it doesn't match."""
    m = LINE.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    d["status"] = int(d["status"])
    d["ms"] = int(d["ms"])
    d["bytes"] = 0 if d["bytes"] == "-" else int(d["bytes"])
    return d


def parse(lines):
    """Parse an iterable of lines. Returns (records, malformed_count)."""
    records, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        rec = parse_line(line)
        if rec is None:
            bad += 1
        else:
            records.append(rec)
    return records, bad


def report(records, malformed=0, slow_ms=500):
    """Roll parsed records up into a traffic summary."""
    total = len(records)
    statuses = Counter(r["status"] for r in records)
    errors = sum(n for s, n in statuses.items() if s >= 400)

    # average latency per endpoint (path), so one slow route stands out
    times = defaultdict(list)
    for r in records:
        times[r["path"]].append(r["ms"])
    avg_by_path = {p: sum(v) / len(v) for p, v in times.items()}

    return {
        "total_requests": total,
        "malformed_lines": malformed,
        "status_counts": dict(sorted(statuses.items())),
        "error_rate_pct": round(100 * errors / total, 1) if total else 0.0,
        "slow_requests": sum(1 for r in records if r["ms"] >= slow_ms),
        "slowest_endpoints": sorted(
            avg_by_path.items(), key=lambda kv: kv[1], reverse=True
        )[:3],
        "top_talkers": Counter(r["ip"] for r in records).most_common(3),
    }


if __name__ == "__main__":
    sample = [
        '1.1.1.1 - - [10/Jul/2026:13:55:36] "GET /api/users HTTP/1.1" 200 342 18',
        '1.1.1.1 - - [10/Jul/2026:13:55:37] "GET /api/orders HTTP/1.1" 200 1204 640',
        '2.2.2.2 - - [10/Jul/2026:13:55:38] "POST /api/login HTTP/1.1" 401 - 25',
        '3.3.3.3 - - [10/Jul/2026:13:55:39] "GET /api/orders HTTP/1.1" 500 0 880',
        '1.1.1.1 - - [10/Jul/2026:13:55:40] "GET /health HTTP/1.1" 200 2 3',
        'this line is corrupted and will not match',
        '2.2.2.2 - - [10/Jul/2026:13:55:41] "GET /api/users HTTP/1.1" 404 - 12',
    ]

    records, bad = parse(sample)
    for k, v in report(records, malformed=bad).items():
        print(f"{k}: {v}")
