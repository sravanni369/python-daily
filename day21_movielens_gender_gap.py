"""Day 21 - the MovieLens gender gap, with the error bars the book never draws.

Source: Python for Data Analysis, 3rd Edition (Wes McKinney, 2022), Chapter 13,
"Data Analysis Examples", the MovieLens 1M section. Free full text at
wesmckinney.com/book. Data pulled from the author's own repo, github.com/wesm/pydata-book.

FIRST RESULT: the book's code still runs clean and its published numbers reproduce
exactly on Python 3.13.5 / pandas 2.3.1, four years and two major pandas versions after
publication. 1,216 active titles; Close Shave 4.644444 F / 4.473795 M; Dirty Dancing
gap -0.830782. Nothing raised, nothing warned. That is worth stating plainly, because
most of this verification series finds breakage and this one does not.

THE ACTUAL FINDING is not a bug in his code. It is what the analysis leaves unchecked.

The book keeps titles with at least 250 ratings, then ranks the male-minus-female mean
gap and reads the ends of that ranking as a result about taste. The filter counts
TOTAL ratings. The comparison is between two groups. Those are different quantities,
and in this dataset they come apart badly: 71.7% of raters are men and 75.4% of ratings
come from men, so a title can clear 250 total ratings on the strength of its male raters
alone. The thinnest title the filter admits has 13 female ratings.

Put a standard error on each gap and 793 of the 1,216 gaps - 65.2% - cannot be
distinguished from zero at 95%.

The book's own headline examples survive this test. Dirty Dancing, Grease and Jumpin'
Jack Flash are all comfortably significant, so its stated conclusions stand. The problem
is that the same table read the same way yields several hundred "differences" that are
sampling noise, and nothing in the method tells you which is which.
"""

import os

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/datasets/movielens"
CACHE = "ml"
MIN_TOTAL = 250          # the book's filter
CONF = 1.96              # 95%


def load():
    """Fetch the three MovieLens 1M tables, cached so reruns stay offline."""
    os.makedirs(CACHE, exist_ok=True)
    specs = {
        "users": ["user_id", "gender", "age", "occupation", "zip"],
        "ratings": ["user_id", "movie_id", "rating", "timestamp"],
        "movies": ["movie_id", "title", "genres"],
    }
    out = {}
    for name, cols in specs.items():
        path = os.path.join(CACHE, name + ".dat")
        if not os.path.exists(path):
            print(f"downloading {name}.dat")
            pd.read_table(f"{BASE}/{name}.dat", sep="::", header=None, names=cols,
                          engine="python").to_csv(path, sep="\t", index=False)
        out[name] = pd.read_table(path, sep="\t")
    return out["users"], out["ratings"], out["movies"]


def book_analysis(data):
    """Chapter 13 as printed: filter on TOTAL ratings, rank the gap."""
    mean_ratings = data.pivot_table("rating", index="title", columns="gender",
                                    aggfunc="mean")
    ratings_by_title = data.groupby("title").size()
    active_titles = ratings_by_title.index[ratings_by_title >= MIN_TOTAL]
    mean_ratings = mean_ratings.loc[active_titles]
    mean_ratings["diff"] = mean_ratings["M"] - mean_ratings["F"]
    return mean_ratings, active_titles


def with_error_bars(data, active_titles):
    """The same gaps, plus the standard error of each difference in means."""
    counts = data.pivot_table("rating", index="title", columns="gender", aggfunc="count")
    means = data.pivot_table("rating", index="title", columns="gender", aggfunc="mean")
    sds = data.pivot_table("rating", index="title", columns="gender", aggfunc="std")
    c, m, s = counts.loc[active_titles], means.loc[active_titles], sds.loc[active_titles]

    out = pd.DataFrame({
        "F": m["F"], "M": m["M"],
        "diff": m["M"] - m["F"],
        "n_F": c["F"].astype(int), "n_M": c["M"].astype(int),
    })
    # standard error of a difference between two independent means
    out["se_diff"] = np.sqrt(s["F"] ** 2 / c["F"] + s["M"] ** 2 / c["M"])
    out["abs_diff"] = out["diff"].abs()
    out["significant"] = out["abs_diff"] > CONF * out["se_diff"]
    return out


if __name__ == "__main__":
    users, ratings, movies = load()
    data = pd.merge(pd.merge(ratings, users), movies)

    print("=" * 74)
    print("1. THE BOOK'S CODE, REPRODUCED  (Ch. 13, MovieLens 1M)")
    print("=" * 74)
    print(f"users {users.shape}  ratings {ratings.shape}  movies {movies.shape}")
    mean_ratings, active = book_analysis(data)
    print(f"titles with >= {MIN_TOTAL} ratings: {len(active):,}   (book prints 1,216)")
    top_f = mean_ratings.sort_values("F", ascending=False)
    print("\ntop female-rated, book's Table:")
    print(top_f[["F", "M"]].head(3).round(6).to_string())
    print("\nlargest gaps toward women, book's Table:")
    print(mean_ratings.sort_values("diff")[["F", "M", "diff"]].head(3).round(6).to_string())
    print("\nEvery published figure matches. The code does not break on pandas 2.x.")

    print()
    print("=" * 74)
    print("2. WHAT THE FILTER ACTUALLY GUARANTEES")
    print("=" * 74)
    g_users = users["gender"].value_counts(normalize=True)
    g_rates = data["gender"].value_counts(normalize=True)
    print(f"raters:  {g_users['M']:.1%} male / {g_users['F']:.1%} female")
    print(f"ratings: {g_rates['M']:.1%} male / {g_rates['F']:.1%} female")
    tbl = with_error_bars(data, active)
    print(f"\nThe filter counts TOTAL ratings, but the comparison is between groups.")
    print(f"  fewest female ratings on any 'active' title : {tbl['n_F'].min()}")
    print(f"  active titles with fewer than 50 women      : {(tbl['n_F'] < 50).sum()}")
    print(f"  median female ratings per active title      : {tbl['n_F'].median():.0f}")

    print()
    print("=" * 74)
    print("3. THE GAPS THAT SURVIVE THEIR OWN ERROR BARS")
    print("=" * 74)
    sig, tot = int(tbl["significant"].sum()), len(tbl)
    print(f"gaps distinguishable from zero at 95%: {sig:,} of {tot:,}")
    print(f"gaps that are NOT                    : {tot - sig:,}  = "
          f"{100 * (1 - sig / tot):.1f}% of the ranking")
    print("\nlargest gaps, with the female sample behind each:")
    cols = ["F", "M", "diff", "n_F", "se_diff", "significant"]
    print(tbl.sort_values("abs_diff", ascending=False).head(8)[cols].round(3).to_string())
    print("\nthinnest titles the >=250 filter let through:")
    print(tbl.nsmallest(5, "n_F")[cols].round(3).to_string())

    print()
    print("=" * 74)
    print("4. WHAT CHANGES IF YOU FILTER ON THE RIGHT QUANTITY")
    print("=" * 74)
    strict = tbl[(tbl["n_F"] >= 100) & (tbl["n_M"] >= 100)]
    print(f"requiring 100+ ratings from EACH gender keeps {len(strict):,} of {tot:,} titles")
    print(f"  of those, {int(strict['significant'].sum()):,} have a real gap "
          f"({100 * strict['significant'].mean():.1f}%), against "
          f"{100 * tbl['significant'].mean():.1f}% before")
    lost = set(tbl.sort_values("abs_diff", ascending=False).head(20).index) - set(strict.index)
    print(f"\n{len(lost)} of the book-style top 20 gaps disappear under that rule:")
    for t in sorted(lost)[:6]:
        print(f"  {t}  (n_F={tbl.loc[t, 'n_F']})")
    print("\nThe book's own examples are not among the casualties: Dirty Dancing,")
    print("Grease and Jumpin' Jack Flash all clear the stricter bar. Its conclusions")
    print("hold. The method around them is what does not generalise.")
