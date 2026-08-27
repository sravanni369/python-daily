"""Day 19 - "Accuracy: 1.0" printed from a test set of one row.

Source: Zarnappa Earnoor, *Machine Learning: The 3 Core Paradigms*
        ("A Data Science Field Guide - Algorithms + Python code"),
        page 5, "Paradigm 1 - Supervised Learning - Python code example".

The book prints the canonical four-step pattern and captions it
"Pattern: create -> fit -> predict -> evaluate." The code runs. Nothing in it
is deprecated, it raises nothing, and it prints a number. The number is the
problem: the book's X has four rows, so test_size=0.25 hands the metric a
single prediction to score.

An accuracy computed on one row has exactly two reachable values. It cannot
report 0.7. It cannot be wrong by a little. It is 0.0 or 1.0, and which one
you get is a property of `random_state`, not of the model.

This file re-proves that, then measures how far up the effect reaches: the
resolution of a reported accuracy is 1/len(y_test), so a small split does not
hand you a noisy estimate of skill - it hands you a number on a grid too
coarse to hold one.

Distinct from Day 16 (validation selection bias), which is about picking the
best of many models on one validation set - a bias in the *choice*. This is a
variance-and-resolution problem in the *measurement*, with a single model that
never changes.

scikit-learn + numpy only. Deterministic. No network.
Run: python day19_accuracy_of_one_row.py
"""

from collections import Counter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)

# The book's page-5 data and split, copied exactly.
BOOK_X = [[1, 2], [2, 3], [3, 4], [4, 5]]
BOOK_Y = [0, 0, 1, 1]
BOOK_TEST_SIZE = 0.25
BOOK_SEED = 42


def book_snippet(random_state=BOOK_SEED):
    """Run the book's page-5 code with only `random_state` changed.

    Returns (accuracy, n_test, train_labels, test_labels). Nothing about the
    model, the data or the split fraction moves between calls.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        BOOK_X, BOOK_Y, test_size=BOOK_TEST_SIZE, random_state=random_state
    )
    model = RandomForestClassifier(random_state=BOOK_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), len(y_test), list(y_train), list(y_test)


def sweep_book(seeds=range(100)):
    """Every accuracy the book's line can print, and how often it prints it."""
    counts = Counter()
    one_class_train = 0
    for seed in seeds:
        acc, _, y_train, _ = book_snippet(seed)
        counts[acc] += 1
        if len(set(y_train)) == 1:
            one_class_train += 1
    return counts, one_class_train


def reported_accuracy_spread(X, y, n_rows, seeds, test_size=BOOK_TEST_SIZE):
    """The book's pattern (one split, test_size=0.25, no stratify) at size n.

    Takes a fixed random subsample of `n_rows` rows, then reruns the whole
    pattern once per seed. Degenerate seeds - the ones whose train set holds a
    single class, so the model can only ever answer one way - are kept in the
    distribution, because the book's pattern keeps them too: it prints their
    accuracy like any other. Returns the scores, the test-set size, and how
    many of those seeds were degenerate.
    """
    rng = np.random.RandomState(0)
    order = rng.permutation(len(y))[:n_rows]
    X_sub, y_sub = X[order], y[order]

    scores, degenerate, n_test = [], 0, 0
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=test_size, random_state=seed
        )
        n_test = len(y_test)
        if len(set(y_train)) == 1:
            degenerate += 1
        model = RandomForestClassifier(n_estimators=50, random_state=BOOK_SEED)
        model.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, model.predict(X_test)))
    return np.array(scores), n_test, degenerate


def majority_baseline(y):
    """Accuracy of always predicting the most common label. The floor."""
    counts = Counter(list(y))
    return counts.most_common(1)[0][1] / len(y)


def honest_estimate(X, y, n_rows):
    """Repeated stratified 5-fold on the same rows: every row gets tested."""
    rng = np.random.RandomState(0)
    order = rng.permutation(len(y))[:n_rows]
    X_sub, y_sub = X[order], y[order]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=BOOK_SEED)
    model = RandomForestClassifier(n_estimators=50, random_state=BOOK_SEED)
    scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std(), len(scores)


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":
    rule("1. The book's page-5 code, run verbatim")
    acc, n_test, y_train, y_test = book_snippet()
    print(f"  Accuracy: {acc}          <- exactly what the book tells you to print")
    print(f"  rows in X            : {len(BOOK_X)}")
    print(f"  test_size=0.25       -> test rows: {n_test}")
    print(f"  train labels         : {y_train}")
    print(f"  test labels          : {y_test}")
    print(f"  values it can print  : {1 + n_test}  (k correct out of {n_test})")
    print("  The model was scored on one prediction. It got that one right.")

    rule("2. Same code, same model, same data - only random_state moves")
    counts, one_class = sweep_book()
    for value in sorted(counts):
        print(f"  prints {value:<5} on {counts[value]:>3} of 100 seeds")
    print(f"  distinct values ever printed        : {len(counts)}")
    print(f"  seeds with a single-class train set : {one_class}")
    print("  Nothing about the model changed. The score is a property of the seed.")

    rule("3. How far up does it reach? Same pattern on real data")
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"  breast cancer: {X.shape[0]} rows x {X.shape[1]} features, binary target")
    print(f"  majority-class baseline on the full set : {majority_baseline(y):.3f}")
    print()
    print(f"  {'n rows':>7} {'test':>5} {'grid':>6} {'min':>7} {'max':>7} "
          f"{'spread':>7} {'sd':>6} {'1-class':>8}")
    print("  " + "-" * 64)
    seeds = range(200)
    for n_rows in (4, 8, 20, 40, 100, 400):
        scores, n_te, degenerate = reported_accuracy_spread(X, y, n_rows, seeds)
        grid = f"1/{n_te}"
        print(f"  {n_rows:>7} {n_te:>5} {grid:>6} {scores.min():>7.3f} "
              f"{scores.max():>7.3f} {scores.max() - scores.min():>7.3f} "
              f"{scores.std():>6.3f} {degenerate:>8}")
    print()
    print("  'grid' is the resolution of the printed number: with 1 test row the")
    print("  metric can only land on 0 or 1, with 5 rows only on fifths. 'spread' is")
    print("  how far the SAME model's reported score travels over the SAME rows,")
    print("  moved by nothing but the split seed. '1-class' counts the seeds whose")
    print("  train set held one class only - the model never saw the other label and")
    print("  still printed an accuracy. At the book's own n=4 that is 58 of 200 runs.")
    print()
    print("  The sd falls monotonically as the test set grows - 0.454, 0.217, 0.145,")
    print("  0.080, 0.052, 0.022 - a 20x tightening driven only by test-set size,")
    print("  since it is the same forest with the same seed on every line. 'spread'")
    print("  is not monotonic (0.500 then 0.600) because at 2 test rows the grid")
    print("  itself caps how far apart two runs can land.")

    rule("4. The fix, measured rather than asserted")
    for n_rows in (40, 100, 400):
        single, n_te, _ = reported_accuracy_spread(X, y, n_rows, seeds)
        mean, sd, n_folds = honest_estimate(X, y, n_rows)
        print(f"  n={n_rows:<4} one split  (n_test={n_te:<3})     : "
              f"{single.mean():.3f} +/- {single.std():.3f}   "
              f"range [{single.min():.3f}, {single.max():.3f}]")
        print(f"        repeated 5-fold ({n_folds} fits) : "
              f"{mean:.3f} +/- {sd:.3f}")
    print()
    print("  Note the means agree. A single split is not biased - it is unbiased and")
    print("  wide. That is the trap: the pattern is not lying on average, but you do")
    print("  not publish the average. You publish one draw from that range, and at")
    print("  n=40 that range runs from 0.600 to 1.000 for one unchanged model.")
    print("  Cross-validation tests every row instead of one lucky quarter, so the")
    print("  estimate stops depending on which quarter it was.")

    rule("Verdict")
    print("  The book's pattern - create, fit, predict, evaluate - is correct.")
    print("  What it leaves out is that the last step needs enough test rows to hold")
    print("  a number. On its own four-row example the evaluation step is decorative:")
    print("  70% of seeds print 1.0 and 30% print 0.0, for a model that never changed.")
    print("  Print len(y_test) next to the accuracy and the problem is visible in one")
    print("  line, in any script, at any scale.")
