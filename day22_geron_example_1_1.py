"""Day 22 - Geron's Example 1-1, and the seven countries it leaves out.

Source: Hands-On Machine Learning with Scikit-Learn and TensorFlow (Aurelien Geron),
Chapter 1, Example 1-1 "Training and running a linear model using Scikit-Learn".
Data and notebook: github.com/ageron/handson-ml

TWO FINDINGS, and the second one is not an accusation.

1. Example 1-1 AS PRINTED CANNOT RUN. The listing calls prepare_country_stats() and
   never defines it. Copy the example out of the book exactly and you get
   NameError: name 'prepare_country_stats' is not defined. The function exists only in
   the companion notebook. Every other line runs; this one stops it dead.

2. That missing function silently drops 7 of the 36 countries:
       Brazil, Mexico, Chile, Czech Republic, Norway, Switzerland, Luxembourg
   which are the four poorest and the three richest in the merged data. Removing both
   ends of an income distribution is the single most effective way to straighten a line.

       kept 29 countries : R2 = 0.7344, Cyprus = 5.96242338   <- the book's printed output
       all 36 countries  : R2 = 0.4041, Cyprus = 6.28653637

   The slope halves. The fit loses almost half its explanatory power.

GERON IS NOT HIDING THIS. His notebook keeps the removed rows in a variable called
missing_data precisely so he can plot them later, and Chapter 1 returns to them as the
worked illustration of sampling bias and nonrepresentative training data. The filtering
is the setup for a lesson, not a mistake.

The problem is what a reader copies. Example 1-1 is the first runnable code in the book
and the piece most people lift. Taken alone it presents a relationship as roughly twice
as strong as the full data supports, and nothing in the printed listing says so.

Verified on Python 3.13.5, scikit-learn 1.8.0, pandas 2.3.1.
"""

import os

import numpy as np
import pandas as pd
import sklearn.linear_model

BASE = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat"
CACHE = "lifesat"
CYPRUS_GDP = 22587


def load():
    """Fetch Geron's two CSVs, cached so later runs stay offline."""
    os.makedirs(CACHE, exist_ok=True)
    paths = {}
    for name in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
        p = os.path.join(CACHE, name)
        if not os.path.exists(p):
            print(f"downloading {name}")
            import urllib.request
            urllib.request.urlretrieve(f"{BASE}/{name}", p)
        paths[name] = p
    oecd = pd.read_csv(paths["oecd_bli_2015.csv"], thousands=",")
    gdp = pd.read_csv(paths["gdp_per_capita.csv"], thousands=",", delimiter="\t",
                      encoding="latin1", na_values="n/a")
    return oecd, gdp


def merge_all(oecd_bli, gdp_per_capita):
    """Everything prepare_country_stats does EXCEPT the row removal."""
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp = gdp_per_capita.rename(columns={"2015": "GDP per capita"}).set_index("Country")
    full = pd.merge(left=oecd_bli, right=gdp, left_index=True, right_index=True)
    full.sort_values(by="GDP per capita", inplace=True)
    return full[["GDP per capita", "Life satisfaction"]]


def fit(df):
    """Example 1-1's model, on whatever rows it is handed."""
    X = np.c_[df["GDP per capita"]]
    y = np.c_[df["Life satisfaction"]]
    model = sklearn.linear_model.LinearRegression().fit(X, y)
    return {
        "n": len(df),
        "slope": model.coef_[0][0],
        "intercept": model.intercept_[0],
        "r2": model.score(X, y),
        "cyprus": model.predict([[CYPRUS_GDP]])[0][0],
    }


if __name__ == "__main__":
    print("=" * 72)
    print("1. EXAMPLE 1-1 EXACTLY AS PRINTED IN THE BOOK")
    print("=" * 72)
    oecd, gdp = load()
    print(f"oecd_bli {oecd.shape}   gdp_per_capita {gdp.shape}   both load fine")
    try:
        country_stats = prepare_country_stats(oecd, gdp)   # noqa: F821
        print("ran")
    except NameError as e:
        print(f"-> NameError: {e}")
        print("The listing calls this function and never defines it. The example as")
        print("printed does not run. The definition is only in the companion notebook.")

    print()
    print("=" * 72)
    print("2. WHAT THE MISSING FUNCTION QUIETLY DOES")
    print("=" * 72)
    full = merge_all(oecd, gdp)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]          # verbatim from Geron's notebook
    keep_indices = list(set(range(36)) - set(remove_indices))
    sample = full.iloc[keep_indices]
    dropped = full.iloc[remove_indices]

    print(f"countries after the merge : {len(full)}")
    print(f"countries the book keeps  : {len(sample)}")
    print(f"silently dropped          : {len(dropped)}")
    print()
    print(dropped.round(1).to_string())
    print()
    print("Those are the 4 poorest and the 3 richest countries in the data.")
    print("Cutting both tails is the most effective way to straighten a line.")

    print()
    print("=" * 72)
    print("3. THE SAME MODEL, THE SAME CODE, DIFFERENT ROWS")
    print("=" * 72)
    book, everything = fit(sample), fit(full)
    print(f"{'':22s}{'n':>4}{'slope':>13}{'intercept':>11}{'R2':>9}{'Cyprus':>13}")
    for label, r in (("book (7 dropped)", book), ("all 36 countries", everything)):
        print(f"{label:22s}{r['n']:>4}{r['slope']:>13.3e}{r['intercept']:>11.4f}"
              f"{r['r2']:>9.4f}{r['cyprus']:>13.8f}")
    print()
    print(f"The book prints [[ 5.96242338 ]]. Reproduced exactly: {book['cyprus']:.8f}")
    print(f"On the full data the same code predicts        : {everything['cyprus']:.8f}")
    print(f"R2 falls from {book['r2']:.4f} to {everything['r2']:.4f}. "
          f"The slope drops {100 * (1 - everything['slope'] / book['slope']):.0f}%.")

    print()
    print("=" * 72)
    print("4. THE FAIR READING")
    print("=" * 72)
    print("Geron removes these rows on purpose. His notebook stores them as")
    print("`missing_data` so he can plot them, and Chapter 1 comes back to them as the")
    print("worked example of sampling bias. The filtering is a teaching setup.")
    print()
    print("What it costs is this: Example 1-1 is the first runnable code in the book and")
    print("the piece most people copy. On its own it shows a relationship about twice as")
    print("strong as the full data supports, and the printed listing gives no sign that")
    print("seven countries were removed to get there.")
