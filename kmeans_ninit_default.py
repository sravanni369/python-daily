"""The book prints KMeans(n_clusters=2). That line does 1/10th of the work it used to.

Book:     50 Algorithms Every Programmer Should Know - Imran Ahmad, Packt
Chapter:  6, Unsupervised Machine Learning Algorithms
Source:   Unsupervised_Machine_Learning_Algorithms.ipynb, code cell 5, printed as:

              kmeans = cluster.KMeans(n_clusters=2)
              kmeans.fit(dataset)

The book never passes n_init, so it inherits whatever scikit-learn's default is.
That default changed in scikit-learn 1.4: n_init went from 10 to the string 'auto'.
With the default init='k-means++', 'auto' resolves to 1.

No error. No warning. The same printed line now runs one k-means restart where it
used to run ten, and k-means only guarantees a local optimum per restart.

The book's own dataset is a 2-column toy set where this cannot show up. Optimisation
stability needs real structure, so this runs on a real public dataset instead:
UCI Optical Recognition of Handwritten Digits (1797 samples, 64 features, 10 classes),
which ships offline with scikit-learn as load_digits(). Directory source:
'Datascience public datasets.pdf', p.23, "UCI Machine Learning Repository".

Every number below is printed by this script. Nothing here is asserted from memory.
"""

import inspect
import platform
import time

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score

SEEDS = range(30)
K = 10


def resolved_n_init(init):
    """What 'auto' actually becomes for a given init, measured by fitting."""
    km = KMeans(n_clusters=K, init=init).fit(load_digits().data)
    return km._n_init


def run(X, y, n_init, seeds):
    """Fit KMeans once per seed. Return (inertias, aris, total_seconds)."""
    inertias, aris = [], []
    t0 = time.perf_counter()
    for s in seeds:
        km = KMeans(n_clusters=K, n_init=n_init, random_state=s).fit(X)
        inertias.append(km.inertia_)
        aris.append(adjusted_rand_score(y, km.labels_))
    return np.array(inertias), np.array(aris), time.perf_counter() - t0


def main():
    print("python      ", platform.python_version())
    print("scikit-learn", sklearn.__version__)
    print("numpy       ", np.__version__)

    default = inspect.signature(KMeans.__init__).parameters["n_init"].default
    print(f"\nKMeans default n_init is {default!r} (it was 10 before scikit-learn 1.4)")

    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"data: UCI handwritten digits, {X.shape[0]} samples, {X.shape[1]} features, k={K}")

    # --- 1. what the book's line resolves to today -------------------------
    print("\n1. WHAT 'auto' ACTUALLY MEANS  (fitted, not read from docs)")
    for init in ("k-means++", "random"):
        print(f"   init={init:<11} -> n_init resolves to {resolved_n_init(init)}")
    print("   'auto' is not one value. It depends on an argument the book also never sets.")

    # --- 2. the book's line vs the book-era default ------------------------
    print(f"\n2. SAME LINE, {len(SEEDS)} SEEDS, ONE ARGUMENT DIFFERENT")
    today_i, today_a, today_t = run(X, y, "auto", SEEDS)   # what the book's line does now
    book_i, book_a, book_t = run(X, y, 10, SEEDS)          # what it did when written

    print(f"   {'':<22}{'best':>12}{'mean':>12}{'worst':>12}{'spread':>12}")
    for name, arr in (("n_init='auto' (=1)", today_i), ("n_init=10 (book era)", book_i)):
        print(f"   {name:<22}{arr.min():>12.1f}{arr.mean():>12.1f}"
              f"{arr.max():>12.1f}{arr.max() - arr.min():>12.1f}")

    worse = int((today_i > book_i).sum())
    gap = (today_i - book_i) / book_i * 100
    print(f"\n   inertia is lower=better. today's default is worse on "
          f"{worse}/{len(SEEDS)} seeds")
    print(f"   median gap {np.median(gap):+.2f}%   worst seed {gap.max():+.2f}%")

    # --- 3. does the worse optimum mean worse clustering? ------------------
    print("\n3. DOES IT REACH THE LABELS? (adjusted Rand index vs the true digit)")
    print(f"   n_init='auto' (=1)    mean ARI {today_a.mean():.4f}   min {today_a.min():.4f}")
    print(f"   n_init=10 (book era)  mean ARI {book_a.mean():.4f}   min {book_a.min():.4f}")
    print(f"   difference in mean ARI {book_a.mean() - today_a.mean():+.4f}")

    # --- 4. what the old default cost ---------------------------------------
    print("\n4. WHAT THE OLD DEFAULT COST  (this is why it was changed)")
    print(f"   n_init='auto' (=1)    {today_t:.2f}s for {len(SEEDS)} fits")
    print(f"   n_init=10 (book era)  {book_t:.2f}s for {len(SEEDS)} fits"
          f"  ({book_t / today_t:.1f}x)")

    # --- 5. the edge case the book's line hides -----------------------------
    print("\n5. THE CASE THAT LOOKS FINE AND IS NOT")
    km = KMeans(n_clusters=K).fit(X)          # exactly the book's call shape
    print(f"   KMeans(n_clusters={K}).fit(X) ran clean. no error, no warning.")
    print(f"   it silently used n_init={km._n_init}, inertia {km.inertia_:.1f}")
    print(f"   the same call in the book's era used n_init=10, inertia "
          f"{KMeans(n_clusters=K, n_init=10).fit(X).inertia_:.1f}")
    print("   nothing in the output tells you which one you got.")


if __name__ == "__main__":
    main()
