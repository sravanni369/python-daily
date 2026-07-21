"""Day 06 — SQL via Python (sqlite3): joins, aggregation, and validation queries.

Builds a tiny in-memory claims database (patients, claims) and demonstrates
the SQL patterns used daily in data-quality work: INNER/LEFT joins,
aggregation with GROUP BY/HAVING, subqueries, and reconciliation-style
validation queries (orphan records, duplicates, out-of-range values).

Standard library only — sqlite3 ships with Python.
"""

import sqlite3
from datetime import date


def build_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with two related tables and sample rows."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            state      TEXT NOT NULL
        );

        CREATE TABLE claims (
            claim_id   INTEGER PRIMARY KEY,
            patient_id INTEGER,          -- intentionally no FK constraint: we validate it ourselves
            service    TEXT NOT NULL,
            amount     REAL NOT NULL,
            claim_date TEXT NOT NULL
        );
        """
    )
    patients = [
        (1, "Asha Rao", "GA"),
        (2, "Ben Ortiz", "TX"),
        (3, "Chen Wei", "GA"),
        (4, "Dana Smith", "NY"),
    ]
    claims = [
        (101, 1, "Lab Panel", 250.00, "2026-07-01"),
        (102, 1, "X-Ray", 410.50, "2026-07-03"),
        (103, 2, "Office Visit", 125.00, "2026-07-03"),
        (104, 3, "MRI", 1899.99, "2026-07-05"),
        (105, 99, "Office Visit", 130.00, "2026-07-06"),   # orphan: patient 99 doesn't exist
        (106, 2, "Office Visit", 125.00, "2026-07-03"),    # looks like a duplicate of 103
        (107, 3, "Lab Panel", -40.00, "2026-07-08"),       # invalid negative amount
    ]
    conn.executemany("INSERT INTO patients VALUES (?, ?, ?)", patients)
    conn.executemany("INSERT INTO claims VALUES (?, ?, ?, ?, ?)", claims)
    return conn


def spend_per_patient(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """INNER JOIN + GROUP BY: total claim spend per matched patient."""
    return conn.execute(
        """
        SELECT p.name, p.state, COUNT(c.claim_id) AS n_claims,
               ROUND(SUM(c.amount), 2) AS total_amount
        FROM patients p
        JOIN claims c ON c.patient_id = p.patient_id
        GROUP BY p.patient_id
        ORDER BY total_amount DESC
        """
    ).fetchall()


def patients_without_claims(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """LEFT JOIN ... IS NULL: patients who have no claims at all."""
    return conn.execute(
        """
        SELECT p.patient_id, p.name
        FROM patients p
        LEFT JOIN claims c ON c.patient_id = p.patient_id
        WHERE c.claim_id IS NULL
        """
    ).fetchall()


def orphan_claims(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Validation: claims pointing at a patient_id that doesn't exist."""
    return conn.execute(
        """
        SELECT c.claim_id, c.patient_id, c.service, c.amount
        FROM claims c
        WHERE c.patient_id NOT IN (SELECT patient_id FROM patients)
        """
    ).fetchall()


def duplicate_claims(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Validation: same patient + service + amount + date appearing more than once."""
    return conn.execute(
        """
        SELECT patient_id, service, amount, claim_date, COUNT(*) AS n
        FROM claims
        GROUP BY patient_id, service, amount, claim_date
        HAVING COUNT(*) > 1
        """
    ).fetchall()


def invalid_amounts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Validation: business rule — claim amounts must be positive."""
    return conn.execute(
        "SELECT claim_id, service, amount FROM claims WHERE amount <= 0"
    ).fetchall()


def high_spend_states(conn: sqlite3.Connection, threshold: float) -> list[sqlite3.Row]:
    """Subquery + HAVING: states whose valid-claim spend exceeds a threshold."""
    return conn.execute(
        """
        SELECT p.state, ROUND(SUM(c.amount), 2) AS state_total
        FROM claims c
        JOIN patients p ON p.patient_id = c.patient_id
        WHERE c.amount > 0
        GROUP BY p.state
        HAVING SUM(c.amount) > ?
        ORDER BY state_total DESC
        """,
        (threshold,),
    ).fetchall()


if __name__ == "__main__":
    conn = build_db()

    print(f"Claims DB demo — run date {date.today().isoformat()}\n")

    print("1) Spend per patient (INNER JOIN + GROUP BY):")
    for r in spend_per_patient(conn):
        print(f"   {r['name']:<11} {r['state']}  claims={r['n_claims']}  total=${r['total_amount']}")

    print("\n2) Patients with no claims (LEFT JOIN ... IS NULL):")
    for r in patients_without_claims(conn):
        print(f"   #{r['patient_id']} {r['name']}")

    print("\n3) Orphan claims (patient_id not in patients):")
    for r in orphan_claims(conn):
        print(f"   claim {r['claim_id']} -> missing patient {r['patient_id']} ({r['service']} ${r['amount']})")

    print("\n4) Duplicate claims (GROUP BY + HAVING COUNT>1):")
    for r in duplicate_claims(conn):
        print(f"   patient {r['patient_id']} {r['service']} ${r['amount']} on {r['claim_date']} x{r['n']}")

    print("\n5) Invalid amounts (business-rule check):")
    for r in invalid_amounts(conn):
        print(f"   claim {r['claim_id']} {r['service']} ${r['amount']}")

    print("\n6) States with valid spend > $300:")
    for r in high_spend_states(conn, 300):
        print(f"   {r['state']}: ${r['state_total']}")

    conn.close()
