"""Day 20 - the Netflix sparse matrix, measured on a real downloaded dataset.

Two recipes from the same book, chained:

  Machine Learning with Python Cookbook
    - Chapter 3.0 (PDF p.37)  : download a dataset straight from a URL
          url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
          dataframe = pd.read_csv(url)
    - Recipe 1.3 (PDF p.6)    : "Creating a Sparse Matrix"

Recipe 1.3 is motivated with Netflix:

    "imagine a matrix where the columns are every movie on Netflix, the rows are
    every Netflix user, and the values are how many times a user has watched that
    particular movie. This matrix would have tens of thousands of columns and
    millions of rows! However, since most users do not watch most movies, the
    vast majority of elements would be zero."

    ... "leading to significant computational savings."

The book states that saving and then demonstrates it on a 3x2 matrix holding two
non-zero values. It never measures the saving, and it never says where the saving
stops. This file downloads a real dataset with the book's own method, builds a
real people x items matrix from it, and measures both.

No Netflix subscriber or catalogue figure is claimed as fact. The only numbers
below are measured on this machine or computed from the CSR storage formula, and
each is labelled.
"""

import os

import numpy as np
import pandas as pd
import scipy
from scipy import sparse

BOOK_URL = "https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv"
CACHE = "titanic.csv"


# ------------------------------------------------- book: download a dataset
def load_titanic():
    """Chapter 3.0, p.37. Cached after the first run so this stays offline."""
    if os.path.exists(CACHE):
        print(f"using cached {CACHE}")
        return pd.read_csv(CACHE)
    print(f"downloading {BOOK_URL}")
    df = pd.read_csv(BOOK_URL)
    df.to_csv(CACHE, index=False)
    return df


# ---------------------------------------------- book: Recipe 1.3, verbatim
def book_recipe_1_3():
    """The listing exactly as printed on p.6. Not edited, not improved."""
    matrix = np.array([[0, 0],
                       [0, 1],
                       [3, 0]])
    matrix_sparse = sparse.csr_matrix(matrix)
    print(matrix_sparse)
    return matrix_sparse


# ------------------------------------------------------------- measurement
def csr_bytes(m):
    """Bytes actually held by a CSR matrix."""
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


def csr_bytes_formula(rows, cols, density, value_bytes=8, index_bytes=4):
    nnz = int(rows * cols * density)
    return nnz * value_bytes + nnz * index_bytes + (rows + 1) * index_bytes


def dense_bytes_formula(rows, cols, value_bytes=8):
    return rows * cols * value_bytes


def crossover_density(rows, cols, step=0.01):
    """Density at which CSR stops being the cheaper representation.

    The book does not mention that this point exists.
    """
    dense = dense_bytes_formula(rows, cols)
    d = step
    while d <= 1.0:
        if csr_bytes_formula(rows, cols, d) >= dense:
            return d
        d += step
    return None


def mb(n):
    return n / 1024 / 1024


# -------------------------------------------------------------------- demo
if __name__ == "__main__":
    print("=" * 70)
    print("1. THE BOOK'S OWN DOWNLOAD METHOD  (Chapter 3.0, p.37)")
    print("=" * 70)
    df = load_titanic()
    print(f"real dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head(3).to_string())

    print()
    print("=" * 70)
    print("2. THE BOOK'S SPARSE RECIPE, RUN UNCHANGED  (Recipe 1.3, p.6)")
    print("=" * 70)
    m = book_recipe_1_3()
    coords = [(int(r), int(c)) for r, c in zip(*m.nonzero())]
    print()
    print("book prints:  (1, 1) 1  and  (2, 0) 3  -- nothing else.")
    print(f"scipy {scipy.__version__} prints those same two coordinate lines,")
    print("preceded by a header line the book's edition did not have.")
    print("Stored values are unchanged:", m.data.tolist(), "at", coords)

    print()
    print("=" * 70)
    print("3. THE CLAIM THE BOOK NEVER MEASURES - ON THE REAL DATA")
    print("=" * 70)
    print("Netflix shape, in miniature: rows are people, columns are items,")
    print("and almost every cell is zero. One-hot encode the Name column.")
    onehot = pd.get_dummies(df["Name"]).to_numpy(dtype=np.float64)
    csr = sparse.csr_matrix(onehot)
    rows, cols = onehot.shape
    zeros = rows * cols - csr.nnz
    print(f"  matrix        : {rows:,} passengers x {cols:,} names")
    print(f"  zero cells    : {zeros:,} of {rows * cols:,} "
          f"= {100 * zeros / (rows * cols):.2f}% zero")
    print(f"  dense  float64: {mb(onehot.nbytes):8,.2f} MB   (MEASURED, ndarray.nbytes)")
    print(f"  CSR           : {mb(csr_bytes(csr)):8,.2f} MB   (MEASURED, "
          f"data+indices+indptr)")
    print(f"  saving        : {100 * (1 - csr_bytes(csr) / onehot.nbytes):8.2f} %"
          f"   = {onehot.nbytes / csr_bytes(csr):,.0f}x smaller")
    print("  round trip    :", np.array_equal(csr.toarray(), onehot),
          "(no information was lost)")

    print()
    print("at the scale the book actually describes -- FORMULA, NOT MEASURED:")
    for u, mv in ((1_000_000, 20_000), (10_000_000, 20_000)):
        d_f = dense_bytes_formula(u, mv)
        s_f = csr_bytes_formula(u, mv, 0.005)
        print(f"  {u:>12,} rows x {mv:,} cols at 0.5% density : "
              f"dense {mb(d_f) / 1024:8,.1f} GB  vs  CSR {mb(s_f) / 1024:7,.1f} GB")

    print()
    print("=" * 70)
    print("4. WHERE THE BOOK'S ADVICE STOPS BEING TRUE")
    print("=" * 70)
    print("CSR stores a value AND a column index per non-zero, plus a row")
    print("pointer per row. It is not free. Past some density it costs more")
    print("than the dense array it replaced.")
    print()
    print("  density   CSR vs dense   verdict")
    rng = np.random.default_rng(0)
    for d in (0.001, 0.01, 0.1, 0.5, 0.6, 0.7, 0.9):
        ratio = csr_bytes_formula(rows, cols, d) / dense_bytes_formula(rows, cols)
        print(f"  {d:7.1%}      {ratio:5.2f}x       "
              f"{'cheaper' if ratio < 1 else 'MORE EXPENSIVE'}")
    x = crossover_density(rows, cols)
    print()
    print(f"crossover: CSR becomes the more expensive choice at ~{x:.0%} density.")
    print("checked against a real allocation at 90% density:")
    dense90 = (rng.random((rows, cols)) < 0.9) * 1.0
    csr90 = sparse.csr_matrix(dense90)
    print(f"  dense {mb(dense90.nbytes):7,.2f} MB  vs  CSR "
          f"{mb(csr_bytes(csr90)):7,.2f} MB  -> CSR is "
          f"{csr_bytes(csr90) / dense90.nbytes:.2f}x the size  (MEASURED)")
    print("Below the crossover, use the recipe. Above it, the book's own recipe")
    print("is a pessimisation, and the book never says so.")

    print()
    print("=" * 70)
    print("5. FAILURE CASE THE BOOK'S 3x2 TOY NEVER HITS")
    print("=" * 70)
    dupes = sparse.csr_matrix(([1, 1, 1], ([0, 0, 0], [5, 5, 5])), shape=(10, 10))
    print("three separate 'user 0 watched movie 5' events, built as COO -> CSR:")
    print("  nnz reported :", dupes.nnz)
    print("  value stored :", dupes[0, 5])
    print("scipy silently SUMS duplicate coordinates. For a watch-count matrix")
    print("that is correct. For a ratings matrix it would turn every repeated")
    print("rating into a total. Same recipe, opposite meaning, no warning.")
