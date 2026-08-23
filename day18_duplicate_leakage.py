"""Day 18 - "No duplicates" is a checklist item the standard recipe cannot deliver.

Source: Paulo Ribeiro, "Data Science, Complete lifecycle of ML models, from
understanding the business problem to choosing the architecture" (Part 2, 2026),
ch. 3 "Data Preparation and Engineering", PDF p. 36 (SS 3.2, "Cleaning in Python")
and p. 44 (SS 3.13, "Creating the Final Dataset").

The book gives one line for duplicates - `df = df.drop_duplicates()` - and then, ten
pages later, lists "No duplicates" as a property the final modelling dataset should
have. Nothing in between says what counts as a duplicate.

Two things break here, and both are measured below rather than asserted.

The first is arithmetic. `drop_duplicates()` with no `subset=` compares whole rows,
so it only ever catches the case where a file was appended to itself and the key came
along with it. Re-key every row once - which is exactly what a table with an enforced
primary key gives you - and the identical call drops zero rows while the same
customers are still in there twice. That number is measured below, not argued.

The second survives even after the key is excluded. The copies that matter are the
same customer entered twice: a re-keyed record, an income stored to fewer decimals,
a region typed `  SOUTH ` one time and `South` the next. None of those are equal
rows, so none are dropped, and a random split then scatters copies of one customer
across train and test. The score that comes back is a memory test.

The instrument is 1-nearest-neighbour, because it can only memorise: any accuracy
above the honest number is the leak talking, not a better model. Every number is
averaged over 10 splits, since a single split moves by several points on its own and
a one-split gap would not be evidence of anything. The majority-class baseline is
printed alongside, because a leaked score that beats the floor and an honest score
that does not are the same run.

Standard library only. No pip, no network. Seeded, so the numbers repeat.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

Row = Dict[str, object]

FEATURES = ("age", "income", "region", "plan")
NUMERIC = ("age", "income")
VISIBLE = ("row_id",) + FEATURES + ("churn",)   # what a real CSV would contain
REGIONS = ("North", "South", "East", "West")
PLANS = ("basic", "plus", "premium")
NOISE = 0.25                                    # label flip rate -> Bayes accuracy 0.75


def make_customers(n: int, seed: int) -> List[Row]:
    """n distinct customers. `base_id` is who the row really is, not a feature."""
    rng = random.Random(seed)
    rows: List[Row] = []
    for i in range(n):
        age = rng.randint(21, 74)
        income = round(rng.uniform(28_000, 190_000), 2)
        plan = rng.choice(PLANS)
        signal = (income < 100_000) != (plan == "premium")   # learnable, roughly balanced
        rows.append({
            "row_id": i,
            "base_id": i,
            "age": age,
            "income": income,
            "region": rng.choice(REGIONS),                   # deliberately uninformative
            "plan": plan,
            "churn": int(signal) if rng.random() > NOISE else int(not signal),
        })
    return rows


def inject_duplicates(rows: Sequence[Row], seed: int, rate: float = 0.5) -> List[Row]:
    """Re-enter half the customers. Four flavours, only one of them an exact row."""
    rng = random.Random(seed)
    out = [dict(r) for r in rows]
    next_row_id = max(int(r["row_id"]) for r in rows) + 1
    for src in rng.sample(list(rows), int(len(rows) * rate)):
        copy = dict(src)
        flavour = rng.choice(("exact", "rekeyed", "rounded", "typed"))
        if flavour == "exact":
            pass                                            # every column identical
        elif flavour == "rekeyed":
            copy["row_id"] = next_row_id                    # new key, same person
        elif flavour == "rounded":
            copy["row_id"] = next_row_id
            copy["income"] = round(float(src["income"]), 0)  # 91240.37 -> 91240.0
        else:
            copy["row_id"] = next_row_id
            copy["region"] = f"  {str(src['region']).upper()} "   # "  SOUTH "
        copy["flavour"] = flavour
        next_row_id += 1
        out.append(copy)
    rng.shuffle(out)
    return out


def drop_duplicates(rows: Sequence[Row], subset: Optional[Sequence[str]] = None) -> List[Row]:
    """The book's line. Keeps the first occurrence of each distinct row.

    `base_id` and `flavour` are bookkeeping, not columns of the CSV, so they are never
    part of the comparison - letting them in would hand the cleaner the answer.
    """
    cols = tuple(subset) if subset else VISIBLE
    seen, kept = set(), []
    for r in rows:
        key = tuple(r.get(c) for c in cols)
        if key not in seen:
            seen.add(key)
            kept.append(r)
    return kept


def random_split(rows: Sequence[Row], seed: int, test_frac: float = 0.3
                 ) -> Tuple[List[Row], List[Row]]:
    """Rows go left or right independently - which is the whole problem."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_frac))
    return shuffled[:cut], shuffled[cut:]


def grouped_split(rows: Sequence[Row], seed: int, test_frac: float = 0.3
                  ) -> Tuple[List[Row], List[Row]]:
    """Split customers, then take their rows. A customer lands on one side only."""
    groups = sorted({r["base_id"] for r in rows})
    random.Random(seed).shuffle(groups)
    train_ids = set(groups[:int(len(groups) * (1 - test_frac))])
    train = [r for r in rows if r["base_id"] in train_ids]
    test = [r for r in rows if r["base_id"] not in train_ids]
    return train, test


def _norm(value: object) -> object:
    """What a human means by equal: strip and casefold text, leave numbers alone."""
    return value.strip().casefold() if isinstance(value, str) else value


def _scales(train: Sequence[Row]) -> Dict[str, float]:
    scales = {}
    for col in NUMERIC:
        vals = [float(r[col]) for r in train]
        scales[col] = (max(vals) - min(vals)) or 1.0    # a constant column cannot divide
    return scales


def _distance(a: Row, b: Row, scales: Dict[str, float]) -> float:
    d = 0.0
    for col in NUMERIC:
        d += abs(float(a[col]) - float(b[col])) / scales[col]
    for col in ("region", "plan"):
        d += 0.0 if _norm(a[col]) == _norm(b[col]) else 1.0
    return d


def one_nn_accuracy(train: Sequence[Row], test: Sequence[Row]) -> float:
    """1-NN memorises and nothing else, so it reads the leak directly."""
    if not train or not test:
        return float("nan")
    scales = _scales(train)
    correct = sum(min(train, key=lambda tr: _distance(t, tr, scales))["churn"] == t["churn"]
                  for t in test)
    return correct / len(test)


def majority_baseline(train: Sequence[Row], test: Sequence[Row]) -> float:
    """Guess the commonest training label for everyone. The floor any score must clear."""
    if not train or not test:
        return float("nan")
    guess = int(sum(int(r["churn"]) for r in train) * 2 >= len(train))
    return sum(int(r["churn"]) == guess for r in test) / len(test)


def leaked_fraction(train: Sequence[Row], test: Sequence[Row]) -> float:
    """Share of test rows whose customer also appears in train."""
    ids = {r["base_id"] for r in train}
    return sum(r["base_id"] in ids for r in test) / len(test) if test else float("nan")


def evaluate(rows: Sequence[Row], splitter, seeds: Sequence[int]
             ) -> Tuple[float, float, float, float]:
    """Mean 1-NN accuracy, mean baseline, mean leak share, and the spread of accuracy."""
    accs, bases, leaks = [], [], []
    for s in seeds:
        tr, te = splitter(rows, seed=s)
        accs.append(one_nn_accuracy(tr, te))
        bases.append(majority_baseline(tr, te))
        leaks.append(leaked_fraction(tr, te))
    n = len(accs)
    return sum(accs) / n, sum(bases) / n, sum(leaks) / n, max(accs) - min(accs)


if __name__ == "__main__":
    SEEDS = list(range(1800, 1810))
    customers = make_customers(400, seed=18)
    raw = inject_duplicates(customers, seed=180)

    tally: Dict[str, int] = {}
    for r in raw:
        tally[str(r.get("flavour", "-"))] = tally.get(str(r.get("flavour", "-")), 0) + 1
    print(f"400 distinct customers, {len(raw) - len(customers)} of them re-entered "
          f"-> {len(raw)} rows.")
    for flavour in ("exact", "rekeyed", "rounded", "typed"):
        print(f"  {flavour:<9} {tally.get(flavour, 0):>3}")

    with_key = drop_duplicates(raw)
    without_key = drop_duplicates(raw, subset=FEATURES + ("churn",))
    primary_key = [dict(r, row_id=i) for i, r in enumerate(raw)]   # what a real PK gives
    pk_cleaned = drop_duplicates(primary_key)
    survivors = tally["rekeyed"] + tally["rounded"] + tally["typed"]
    print("\nWhat the book's line actually removes:")
    print(f"  df.drop_duplicates()                {len(raw)} -> {len(with_key)} rows "
          f"({len(raw) - len(with_key)} dropped)")
    print(f"    only the {tally['exact']} copies that kept their key - the file appended to itself.")
    print(f"  ...after re-keying every row        {len(primary_key)} -> {len(pk_cleaned)} rows "
          f"({len(primary_key) - len(pk_cleaned)} dropped)")
    print("    with a real primary key no two rows are equal, so the same call is a no-op")
    print("    and all 200 duplicated customers remain.")
    print(f"  df.drop_duplicates(subset=features) {len(raw)} -> {len(without_key)} rows "
          f"({len(raw) - len(without_key)} dropped)")
    print(f"    excluding the key catches the {tally['exact']} exact copies; the {survivors} "
          "re-keyed, rounded and")
    print("    retyped ones are not equal rows. All three datasets pass \"No duplicates\".")

    datasets = (
        ("raw, uncleaned", raw),
        ("df.drop_duplicates()", with_key),
        ("drop_duplicates(subset=features)", without_key),
    )
    print(f"\n1-NN accuracy, mean of {len(SEEDS)} splits (spread = best minus worst split):")
    print(f"  {'dataset':<34}{'split':<10}{'leaked':<9}{'1-NN acc':<11}{'spread':<9}baseline")
    scores = {}
    for name, splitter in (("random", random_split), ("grouped", grouped_split)):
        for label, data in datasets:
            acc, base, leak, spread = evaluate(data, splitter, SEEDS)
            scores[(name, label)] = acc
            print(f"  {label:<34}{name:<10}{leak:<9.1%}{acc:<11.3f}{spread:<9.3f}{base:.3f}")

    cleaned = "drop_duplicates(subset=features)"
    leaked, honest = scores[("random", cleaned)], scores[("grouped", cleaned)]
    _, floor, leak_share, _ = evaluate(without_key, random_split, SEEDS)
    print(f"\nOn the cleaned dataset the random split reports {leaked:.3f} and the grouped split")
    print(f"reports {honest:.3f}. The {leaked - honest:+.3f} between them is the {leak_share:.0%} of test rows")
    print("whose customer the model had already been shown, answered from memory.")
    verdict = "clears" if honest > floor else "does not clear"
    print(f"The honest {honest:.3f} sits against a majority baseline of {floor:.3f}, so 1-NN "
          f"{verdict} the")
    print(f"floor by {honest - floor:+.3f} - the comparison the leaked number was hiding.")
    print(f"Bayes accuracy is {1 - NOISE:.2f} by construction: {NOISE:.0%} of labels are flipped, so no")
    print(f"model can honestly exceed it, and the leaked {leaked:.3f} is a score of a different test.")

    print("\nWhat this does not fix: `base_id` is known here because the rows were generated.")
    print("On a real customer table nothing marks two rows as one person, and recovering")
    print("that is entity resolution, not a cleaning step.")

    print("\nFAILURE - inputs that break the recipe rather than the model:")
    print(f"  empty dataset    -> drop_duplicates([]) = {drop_duplicates([])}, "
          f"1-NN acc = {one_nn_accuracy([], [])}")
    one = make_customers(1, seed=7) * 40
    tr, te = grouped_split(one, seed=1)
    print(f"  one customer x40 -> grouped split gives train {len(tr)}, test {len(te)}, "
          f"1-NN acc = {one_nn_accuracy(tr, te)}")
    print("                      a random split on the same 40 rows reports "
          f"{one_nn_accuracy(*random_split(one, seed=1)):.3f}.")
    flat = [dict(r, income=50_000.0) for r in make_customers(30, seed=9)]
    print(f"  income constant  -> 1-NN acc = {one_nn_accuracy(*grouped_split(flat, seed=2)):.3f}, "
          "no ZeroDivisionError in the scaler.")
