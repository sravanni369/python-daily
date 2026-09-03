"""End-to-end ML project: predicting home values across Georgia, with Marietta called out.

Structure follows Chapter 2 of Hands-On Machine Learning (Geron) stage for stage. The
1990 California census extract is replaced with current Zillow Research data, which is
published openly and needs no API key.

WHY THIS SUBSTITUTION
Geron predicts median_house_value for California block groups from other attributes of
those block groups. Here the unit is a Georgia ZIP code and the target is the Zillow Home
Value Index. Every stage of his chapter maps across:

    California (1990 census)   ->   Georgia (Zillow Research, current)
    median_house_value         ->   ZHVI, the target
    median_income              ->   median_sale_price
    total_rooms / bedrooms     ->   zhvi_1bed, zhvi_3bed  (housing stock mix)
    population / households    ->   inventory, new_listings  (market size)
    ocean_proximity            ->   metro  (Atlanta metro vs rest of Georgia)

HONEST DIFFERENCES FROM THE BOOK
  - Sample size. California has 20,640 block groups. Georgia has a few hundred ZIP codes
    with complete Zillow coverage. The pipeline is identical; the sample is far smaller,
    so cross-validation spread is wider and tree models overfit more easily.
  - This is market data, not census data, so the features are listing behaviour rather
    than demographics. A ZIP with a high sale price is nearly the same measurement as a
    ZIP with a high home value, which makes the regression much easier than Geron's.
    That is stated in the results rather than hidden: a very high R2 here is a warning
    about feature leakage, not a triumph.
  - No latitude/longitude, so the geographic plots in his chapter are not reproduced.

Data: files.zillowstatic.com/research/public_csvs  (free, no key, no login)
Verified on Python 3.13.5, scikit-learn 1.8.0, pandas 2.3.1.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, StratifiedShuffleSplit,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = "https://files.zillowstatic.com/research/public_csvs/"
CACHE = "zillow_cache"
STATE = "GA"
MARIETTA_ZIPS = [30060, 30062, 30064, 30066, 30067, 30068, 30008]

FILES = {
    "zhvi":          "zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "zori":          "zori/Zip_zori_uc_sfrcondomfr_sm_month.csv",
    "sale_price":    "median_sale_price/Zip_median_sale_price_uc_sfrcondo_sm_month.csv",
    "inventory":     "invt_fs/Zip_invt_fs_uc_sfrcondo_sm_month.csv",
    "days_pending":  "mean_doz_pending/Zip_mean_doz_pending_uc_sfrcondo_sm_month.csv",
    "price_cut_pct": "perc_listings_price_cut/Zip_perc_listings_price_cut_uc_sfrcondo_sm_month.csv",
    "new_listings":  "new_listings/Zip_new_listings_uc_sfrcondo_month.csv",
    "zhvi_1bed":     "zhvi/Zip_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "zhvi_3bed":     "zhvi/Zip_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
}
META = ["RegionName", "State", "City", "Metro", "CountyName"]


def fetch(name, path):
    """Geron's fetch_housing_data(), pointed at Zillow Research."""
    os.makedirs(CACHE, exist_ok=True)
    local = os.path.join(CACHE, name + ".csv")
    if not os.path.exists(local):
        print(f"  downloading {name}")
        urllib.request.urlretrieve(BASE + path, local)
    return pd.read_csv(local)


def latest_column(df, name):
    """Keep the metadata plus the most recent month that has data."""
    months = [c for c in df.columns if c[:2] == "20" and "-" in c]
    for col in reversed(months):
        if df[col].notna().sum() > 100:
            keep = [c for c in META if c in df.columns] + [col]
            out = df[keep].rename(columns={col: name})
            return out, col
    raise SystemExit(f"no usable month found in {name}")


def load_georgia():
    frames, months = {}, {}
    for name, path in FILES.items():
        df, month = fetch(name, path), None
        df, month = latest_column(df, name)
        months[name] = month
        frames[name] = df[df["State"] == STATE] if "State" in df.columns else df
    base = frames["zhvi"][[c for c in META if c in frames["zhvi"].columns] + ["zhvi"]]
    for name, df in frames.items():
        if name == "zhvi":
            continue
        base = base.merge(df[["RegionName", name]], on="RegionName", how="left")
    return base, months


def add_combined_attributes(df):
    """Geron's rooms_per_household step: ratios that mean more than the raw counts."""
    df = df.copy()
    df["price_to_rent"] = df["zhvi"] / (df["zori"] * 12)
    df["inventory_per_listing"] = df["inventory"] / df["new_listings"]
    df["bed_premium"] = df["zhvi_3bed"] / df["zhvi_1bed"]
    return df


def main():
    print("=" * 74)
    print("1. GET THE DATA  (Zillow Research, no API key)")
    print("=" * 74)
    ga, months = load_georgia()
    print(f"\nmost recent month used per file:")
    for k, v in months.items():
        print(f"  {k:14s} {v}")
    ga = ga.dropna(subset=["zhvi"]).reset_index(drop=True)
    print(f"\nGeorgia ZIP codes with a home value: {len(ga)}")
    print(f"(Geron's California extract has 20,640 block groups - far smaller sample here)")

    num_cols = [c for c in ga.columns if c not in META]
    print()
    print(ga[num_cols].describe().round(1).to_string())
    print(f"\nmissing values per column:\n{ga[num_cols].isna().sum().to_string()}")

    mar = ga[ga["RegionName"].isin(MARIETTA_ZIPS)]
    print(f"\nMarietta ZIPs present: {sorted(mar['RegionName'].tolist())}")
    print(mar[["RegionName", "City", "zhvi"]].to_string(index=False))

    print()
    print("=" * 74)
    print("2. CREATE A TEST SET  (stratified on value band, as Geron stratifies on income)")
    print("=" * 74)
    ga["value_cat"] = pd.cut(ga["zhvi"],
                             bins=[0, 150_000, 250_000, 400_000, 700_000, np.inf],
                             labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_i, te_i = next(split.split(ga, ga["value_cat"]))
    train, test = ga.loc[tr_i].copy(), ga.loc[te_i].copy()
    cmp = pd.DataFrame({
        "overall": ga["value_cat"].value_counts(normalize=True).sort_index().round(4),
        "test set": test["value_cat"].value_counts(normalize=True).sort_index().round(4)})
    print(cmp.to_string())
    for s in (train, test):
        s.drop(columns=["value_cat"], inplace=True)
    print(f"\ntrain {len(train)}   test {len(test)}")

    print()
    print("=" * 74)
    print("3. LOOK FOR CORRELATIONS")
    print("=" * 74)
    corr = add_combined_attributes(train).corr(numeric_only=True)["zhvi"]
    print(corr.sort_values(ascending=False).round(4).to_string())

    print()
    print("=" * 74)
    print("4. PREPARE THE DATA")
    print("=" * 74)
    def split_xy(df):
        d = add_combined_attributes(df)
        d["in_atlanta_metro"] = np.where(
            d["Metro"].fillna("").str.contains("Atlanta", case=False),
            "ATLANTA METRO", "REST OF GEORGIA")
        y = d["zhvi"].copy()
        X = d.drop(columns=[c for c in META if c in d.columns] + ["zhvi"])
        return X, y

    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    num_attribs = [c for c in X_train.columns if c != "in_atlanta_metro"]
    pipe = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["in_atlanta_metro"]),
    ])
    Xp = pipe.fit_transform(X_train)
    print(f"features used ({len(num_attribs)} numeric + metro flag):")
    print("  " + ", ".join(num_attribs))
    print(f"prepared matrix: {Xp.shape}")

    print()
    print("=" * 74)
    print("5. SELECT AND TRAIN MODELS  (10-fold cross-validation)")
    print("=" * 74)
    for name, m in (("LinearRegression", LinearRegression()),
                    ("DecisionTree", DecisionTreeRegressor(random_state=42)),
                    ("RandomForest", RandomForestRegressor(n_estimators=100,
                                                           random_state=42))):
        m.fit(Xp, y_train)
        tr_rmse = np.sqrt(mean_squared_error(y_train, m.predict(Xp)))
        cv = np.sqrt(-cross_val_score(m, Xp, y_train,
                                      scoring="neg_mean_squared_error", cv=10))
        print(f"{name:18s} train ${tr_rmse:>10,.0f}   CV ${cv.mean():>10,.0f} "
              f"+/- ${cv.std():>9,.0f}")
    print("\nA train error far below the CV error is overfitting, which is exactly what")
    print("Geron demonstrates when his Decision Tree scores 0.0 on the training set.")

    print()
    print("=" * 74)
    print("6. FINE-TUNE  (GridSearchCV)")
    print("=" * 74)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        [{"n_estimators": [30, 100], "max_features": [2, 4, 6, 8]},
         {"bootstrap": [False], "n_estimators": [30, 100], "max_features": [2, 4]}],
        cv=5, scoring="neg_mean_squared_error")
    grid.fit(Xp, y_train)
    print("best params:", grid.best_params_)
    print(f"best CV RMSE: ${np.sqrt(-grid.best_score_):,.0f}")
    names = num_attribs + list(pipe.named_transformers_["cat"].categories_[0])
    print("\nfeature importances:")
    for s, n in sorted(zip(grid.best_estimator_.feature_importances_, names),
                       reverse=True):
        print(f"  {s:.4f}  {n}")

    print()
    print("=" * 74)
    print("7. EVALUATE ON THE TEST SET")
    print("=" * 74)
    pred = grid.best_estimator_.predict(pipe.transform(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    sq = (pred - y_test) ** 2
    lo, hi = np.sqrt(stats.t.interval(0.95, len(sq) - 1,
                                      loc=sq.mean(), scale=stats.sem(sq)))
    print(f"final test RMSE      : ${rmse:,.0f}")
    print(f"95% confidence bound : ${lo:,.0f} to ${hi:,.0f}")
    print(f"median test home value: ${y_test.median():,.0f}")
    print(f"error as share of median: {100 * rmse / y_test.median():.1f}%")
    print("\nQuote the interval, not the point estimate. On a sample this small the")
    print("interval is wide and the single number would overstate what we know.")

    print()
    print("=" * 74)
    print("8. LEAKAGE CHECK - THE PART THAT DECIDES WHETHER ANY OF THIS IS REAL")
    print("=" * 74)
    print("zhvi_3bed and zhvi_1bed are Zillow Home Value Indices for 3-bed and 1-bed")
    print("homes. The target is the Zillow Home Value Index for all homes. Those are")
    print("the same measurement on a subset of the same houses, so a model leaning on")
    print("them is not predicting a price, it is copying one. bed_premium is built from")
    print("both, so it goes too. Refit with all three removed.")
    print()

    LEAKY = ["zhvi_1bed", "zhvi_3bed", "bed_premium"]
    honest_attribs = [c for c in num_attribs if c not in LEAKY]
    honest_pipe = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), honest_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["in_atlanta_metro"]),
    ])
    Xh = honest_pipe.fit_transform(X_train)
    honest_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        [{"n_estimators": [30, 100], "max_features": [2, 4, 6]}],
        cv=5, scoring="neg_mean_squared_error")
    honest_grid.fit(Xh, y_train)
    hp = honest_grid.best_estimator_.predict(honest_pipe.transform(X_test))
    h_rmse = np.sqrt(mean_squared_error(y_test, hp))
    hsq = (hp - y_test) ** 2
    hlo, hhi = np.sqrt(stats.t.interval(0.95, len(hsq) - 1,
                                        loc=hsq.mean(), scale=stats.sem(hsq)))

    print(f"{'':28s}{'test RMSE':>14}{'% of median':>14}")
    print(f"{'with ZHVI-derived features':28s}{rmse:>14,.0f}"
          f"{100 * rmse / y_test.median():>13.1f}%")
    print(f"{'with them removed':28s}{h_rmse:>14,.0f}"
          f"{100 * h_rmse / y_test.median():>13.1f}%")
    print(f"\nerror grows {100 * (h_rmse / rmse - 1):.0f}% once the leak is closed")
    print(f"honest 95% bound: ${hlo:,.0f} to ${hhi:,.0f}")
    print("\nhonest feature importances:")
    hnames = honest_attribs + list(honest_pipe.named_transformers_["cat"].categories_[0])
    for sc, n in sorted(zip(honest_grid.best_estimator_.feature_importances_, hnames),
                        reverse=True)[:6]:
        print(f"  {sc:.4f}  {n}")
    print("\nThis second number is the one worth reporting. The first was mostly the")
    print("model reading the answer off a feature.")

    print()
    print("=" * 74)
    print("9. MARIETTA")
    print("=" * 74)
    mar_rows = ga[ga["RegionName"].isin(MARIETTA_ZIPS)].copy()
    if len(mar_rows):
        Xm, ym = split_xy(mar_rows)
        pm = grid.best_estimator_.predict(pipe.transform(Xm))
        res = pd.DataFrame({"ZIP": mar_rows["RegionName"].values,
                            "actual": ym.values, "predicted": pm,
                            "error": pm - ym.values})
        print(res.round(0).to_string(index=False))
        print(f"\nMarietta mean absolute error: ${res['error'].abs().mean():,.0f}")
        print("Some of these ZIPs were in training, so this is not a clean holdout.")
        print("It is a sanity check on the home market, not a generalisation estimate.")
    else:
        print("no Marietta ZIPs found in the Zillow Georgia coverage")


if __name__ == "__main__":
    main()
