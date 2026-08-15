"""Day 16 — Why the book keeps a third dataset the model never sees.

Source: "Low-Code AI: A Practical Project-Driven Introduction to Machine
Learning", Gwendolyn Stripling & Michael Abel (O'Reilly), PDF pp. 268-269
(printed pp. 252-253), "Loading the Datasets and the Training-Validation-Test
Data Split".

The book: "The role of the test dataset is to have a final independent dataset
to verify the final model's performance. If the final model has similar
performance on the test dataset as it does to the validation data, then that
model is ready to use. In the case that the model has significantly worse
performance on the test dataset, then you know you have a problem."

What it does not spell out is why the validation score goes wrong. Compare many
models on one validation set and the winner's score stops being an estimate of
its accuracy and becomes the maximum of a pile of noisy scores, and the maximum
of noise is flattering. That bias is measured here on data built with no signal
at all, where the only honest answer is 50%.

One run cannot show this: a single test set is noisy enough to land above the
validation score by luck, which is what the first version of this file did. The
bias is a property of the procedure, not of one split, so the procedure is
repeated many times and the average gap is what gets reported.

Standard library only. No pip, no network. Seeded, so the numbers repeat.
"""

import random
from typing import List, Sequence, Tuple

Row = Tuple[Tuple[int, ...], int]     # (features, label)
Model = Tuple[Tuple[int, ...], int]   # (feature indices, flip bit)

N_FEATURES = 8


def make_rows(n: int, rng: random.Random, signal: bool) -> List[Row]:
    """Binary features. With signal=False the label is an independent coin flip."""
    rows = []
    for _ in range(n):
        feats = tuple(rng.randint(0, 1) for _ in range(N_FEATURES))
        if signal:
            label = feats[0] if rng.random() > 0.10 else 1 - feats[0]
        else:
            label = rng.randint(0, 1)
        rows.append((feats, label))
    return rows


def split_three_ways(rows: Sequence[Row], rng: random.Random,
                     valid_frac: float = 0.1, test_frac: float = 0.4):
    """Shuffle once, cut into train/valid/test. Refuses to return an empty split."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_valid, n_test = int(n * valid_frac), int(n * test_frac)
    n_train = n - n_valid - n_test
    if min(n_train, n_valid, n_test) < 1:
        raise ValueError(
            f"{n} rows at {valid_frac:.0%}/{test_frac:.0%} gives splits "
            f"{n_train}/{n_valid}/{n_test} - an empty split scores 0.0 and hides it"
        )
    return shuffled[:n_train], shuffled[n_train:n_train + n_valid], shuffled[n_train + n_valid:]


def predict(model: Model, feats: Tuple[int, ...]) -> int:
    subset, flip = model
    return (sum(feats[i] for i in subset) % 2) ^ flip


def accuracy(model: Model, rows: Sequence[Row]) -> float:
    if not rows:
        return 0.0
    return sum(predict(model, feats) == label for feats, label in rows) / len(rows)


def fit(subset: Tuple[int, ...], train: Sequence[Row]) -> Model:
    """The one thing learned from the training rows: which way round the rule goes."""
    return (subset, 0) if accuracy((subset, 0), train) >= 0.5 else (subset, 1)


def bakeoff(train, valid, test, rng: random.Random, n_candidates: int = 60):
    """Train n candidates, keep the best validation score, then score it on test."""
    scores, best, best_valid = [], None, -1.0
    for _ in range(n_candidates):
        subset = tuple(sorted(rng.sample(range(N_FEATURES), rng.randint(1, 3))))
        model = fit(subset, train)
        score = accuracy(model, valid)
        scores.append(score)
        if score > best_valid:
            best, best_valid = model, score
    return best, scores, accuracy(best, train), best_valid, accuracy(best, test)


def one_run(seed: int, signal: bool, n_rows: int = 1000, n_candidates: int = 60):
    rng = random.Random(seed)
    train, valid, test = split_three_ways(make_rows(n_rows, rng, signal), rng)
    best, scores, tr, va, te = bakeoff(train, valid, test, rng, n_candidates)
    return {"sizes": (len(train), len(valid), len(test)), "winner": best,
            "mean_candidate_valid": sum(scores) / len(scores),
            "train": tr, "valid": va, "test": te}


def repeat_runs(n_trials: int, signal: bool, base_seed: int = 1000):
    """The same procedure end to end, n_trials times, with fresh data each time."""
    runs = [one_run(base_seed + i, signal) for i in range(n_trials)]
    mean_valid = sum(r["valid"] for r in runs) / n_trials
    mean_test = sum(r["test"] for r in runs) / n_trials
    optimistic = sum(1 for r in runs if r["valid"] > r["test"])
    return mean_valid, mean_test, optimistic, n_trials


def _print_run(run) -> None:
    tr, va, te = run["sizes"]
    print(f"  split            train={tr} valid={va} test={te}")
    print(f"  mean candidate   {run['mean_candidate_valid']:.3f} validation acc")
    print(f"  winner           parity of features {list(run['winner'][0])}, flip={run['winner'][1]}")
    print(f"  train acc        {run['train']:.3f}")
    print(f"  validation acc   {run['valid']:.3f}   <- the number you would report")
    print(f"  test acc         {run['test']:.3f}   <- the number that is true")


if __name__ == "__main__":
    TRIALS = 200

    print("1. NO SIGNAL - labels are coin flips, so 0.500 is the honest ceiling.")
    print("   Best of 60 candidate models, chosen on the validation set:")
    _print_run(one_run(seed=16, signal=False))

    print(f"\n   One run proves nothing - the test set is noisy too. Same procedure,")
    print(f"   {TRIALS} times, fresh data each time:")
    mv, mt, opt, n = repeat_runs(TRIALS, signal=False)
    print(f"     mean validation acc of the winner : {mv:.3f}")
    print(f"     mean test acc of the same model   : {mt:.3f}")
    print(f"     average overstatement             : {mv - mt:+.3f}")
    print(f"     runs where validation flattered   : {opt}/{n}")
    print("   The winners learned nothing. They won a lottery held on the")
    print("   validation set, and only the untouched test set says so.")

    print("\n\n2. REAL SIGNAL - label is feature 0 with 10% of labels flipped.")
    print("   Identical procedure, identical 60 candidates:")
    _print_run(one_run(seed=16, signal=True))
    mv, mt, opt, n = repeat_runs(TRIALS, signal=True)
    print(f"\n   Over {n} runs: validation {mv:.3f}, test {mt:.3f}, gap {mv - mt:+.3f}")
    print("   Validation and test agree, which is the book's ready-to-use case.")
    print("   The test set cannot tell you a model is good - only whether the")
    print("   validation score was a measurement or a selection artifact.")

    print("\n\n3. FAILURE - a dataset too small to cut three ways.")
    try:
        split_three_ways(make_rows(4, random.Random(0), signal=True), random.Random(0))
    except ValueError as err:
        print(f"   ValueError: {err}")
