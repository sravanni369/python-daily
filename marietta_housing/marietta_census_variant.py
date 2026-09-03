"""End-to-end ML project: predicting median home value in Marietta / Cobb County, Georgia.

This follows Chapter 2 of Hands-On Machine Learning (Geron) step for step, but swaps the
1990 California census extract for live American Community Survey data covering Cobb
County, Georgia, where Marietta is the county seat.

WHY THIS SUBSTITUTION WORKS
Geron's California data is block-group level census data. So is this. The ACS publishes
the same shape of information for Cobb County block groups, so every stage of his
chapter maps across without inventing anything:

    California (1990)          ->   Cobb County, GA (ACS 5-year)
    median_house_value         ->   B25077_001E  median value, owner-occupied
    median_income              ->   B19013_001E  median household income
    population                 ->   B01003_001E  total population
    households                 ->   B11001_001E  total households
    total_rooms                ->   B25017_001E  aggregate rooms
    total_bedrooms             ->   B25041_001E  aggregate bedrooms
    housing_median_age         ->   B25035_001E  median year built, converted to age
    ocean_proximity            ->   tenure_mix   owner- vs renter-majority block group

WHAT IS HONESTLY DIFFERENT
  - Size. California has 20,640 block groups. Cobb County has roughly 500. The pipeline
    is identical; the sample is far smaller, so cross-validation spread will be wider and
    the RandomForest will overfit more easily. That is a real limitation, not a bug.
  - No latitude/longitude. The ACS API returns geography codes, not centroids, so the
    geographic scatter plots in Geron's chapter are not reproduced here.
  - The target is not capped at $500,001 the way the 1990 extract was, so the histogram
    quirk he discusses does not appear. Different data, different quirks.

SETUP: this needs a free Census API key, which takes about thirty seconds.
    1. Request one at https://api.census.gov/data/key_signup.html
    2. Then either set it in your shell:   set CENSUS_API_KEY=your_key_here
       or paste it into API_KEY below.
The Census API began rejecting keyless requests, so there is no way around this step.
"""

import os

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

API_KEY = os.environ.get("CENSUS_API_KEY", "")
YEAR = 2022
STATE, COUNTY = "13", "067"          # Georgia, Cobb County (Marietta is the county seat)
CACHE = "cobb_block_groups.csv"

VARS = {
    "B25077_001E": "median_house_value",
    "B19013_001E": "median_income",
    "B01003_001E": "population",
    "B11001_001E": "households",
    "B25017_001E": "total_rooms",
    "B25041_001E": "total_bedrooms",
    "B25035_001E": "median_year_built",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
}


# ----------------------------------------------------------------- get the data
def fetch_data():
    """Geron's fetch_housing_data(), pointed at the Census API instead of a .tgz."""
    if os.path.exists(CACHE):
        print(f"using cached {CACHE}")
        return pd.read_csv(CACHE)
    if not API_KEY:
        raise SystemExit(
            "No Census API key found.\n"
            "  1. Get one free at https://api.census.gov/data/key_signup.html\n"
            "  2. set CENSUS_API_KEY=your_key_here    (then rerun)\n"
            "The Census API rejects keyless requests, so this step cannot be skipped."
        )
    import urllib.request, json
    cols = ",".join(VARS)
    url = (f"https://api.census.gov/data/{YEAR}/acs/acs5?get=NAME,{cols}"
           f"&for=block%20group:*&in=state:{STATE}%20county:{COUNTY}%20tract:*"
           f"&key={API_KEY}")
    print("downloading Cobb County block groups from the Census API")
    raw = json.loads(urllib.request.urlopen(url, timeout=60).read())
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df.to_csv(CACHE, index=False)
    return df


def clean(df):
    """ACS returns strings and uses large negative numbers as null markers."""
    out = pd.DataFrame()
    for code, name in VARS.items():
        out[name] = pd.to_numeric(df[code], errors="coerce")
    out[out < 0] = np.nan                       # ACS null sentinels
    out["housing_median_age"] = YEAR - out.pop("median_year_built")
    # the categorical attribute, standing in for ocean_proximity
    out["tenure_mix"] = np.where(out["owner_occupied"] >= out["renter_occupied"],
                                 "OWNER MAJORITY", "RENTER MAJORITY")
    out = out.drop(columns=["owner_occupied", "renter_occupied"])
    # a block group with no home-value estimate cannot be a training row
    return out.dropna(subset=["median_house_value", "median_income"]).reset_index(drop=True)


# ------------------------------------------------- Geron's attribute combinations
def add_combined_attributes(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def main():
    print("=" * 74)
    print("1. GET THE DATA")
    print("=" * 74)
    housing = clean(fetch_data())
    print(f"Cobb County block groups usable: {len(housing)}")
    print(f"(Geron's California extract has 20,640 - this is a much smaller sample)")
    print()
    print(housing.head().to_string())
    print()
    print(housing.describe().round(1).to_string())
    print()
    print(housing["tenure_mix"].value_counts().to_string())

    print()
    print("=" * 74)
    print("2. CREATE A TEST SET  (stratified on income, as Geron does)")
    print("=" * 74)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0, 40_000, 70_000, 100_000, 150_000, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(housing, housing["income_cat"]))
    strat_train, strat_test = housing.loc[train_idx], housing.loc[test_idx]
    overall = housing["income_cat"].value_counts(normalize=True).sort_index()
    in_test = strat_test["income_cat"].value_counts(normalize=True).sort_index()
    print(pd.DataFrame({"overall": overall.round(4), "stratified test": in_test.round(4)})
          .to_string())
    for s in (strat_train, strat_test):
        s.drop("income_cat", axis=1, inplace=True)
    print(f"\ntrain {len(strat_train)}   test {len(strat_test)}")

    print()
    print("=" * 74)
    print("3. LOOK FOR CORRELATIONS")
    print("=" * 74)
    explore = add_combined_attributes(strat_train.copy())
    corr = explore.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
    print(corr.round(4).to_string())
    print("\nGeron finds median_income dominates in California. Compare that here.")

    print()
    print("=" * 74)
    print("4. PREPARE THE DATA")
    print("=" * 74)
    X_train = add_combined_attributes(strat_train.drop("median_house_value", axis=1))
    y_train = strat_train["median_house_value"].copy()
    num_attribs = [c for c in X_train.columns if c != "tenure_mix"]
    cat_attribs = ["tenure_mix"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ])
    X_prepared = full_pipeline.fit_transform(X_train)
    print(f"prepared matrix: {X_prepared.shape}  "
          f"({len(num_attribs)} numeric + one-hot tenure_mix)")

    print()
    print("=" * 74)
    print("5. SELECT AND TRAIN MODELS  (10-fold cross-validation)")
    print("=" * 74)
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_prepared, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_prepared)))
        cv = np.sqrt(-cross_val_score(model, X_prepared, y_train,
                                      scoring="neg_mean_squared_error", cv=10))
        print(f"{name:18s} train RMSE ${train_rmse:>10,.0f}   "
              f"CV RMSE ${cv.mean():>10,.0f} +/- ${cv.std():>8,.0f}")
    print("\nA train RMSE far below the CV RMSE is overfitting, exactly as Geron shows")
    print("with the Decision Tree scoring 0.0 on the training set in his chapter.")

    print()
    print("=" * 74)
    print("6. FINE-TUNE THE BEST MODEL  (GridSearchCV)")
    print("=" * 74)
    param_grid = [
        {"n_estimators": [10, 30, 100], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [10, 30], "max_features": [2, 4, 6]},
    ]
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                        scoring="neg_mean_squared_error", return_train_score=True)
    grid.fit(X_prepared, y_train)
    print("best params:", grid.best_params_)
    print(f"best CV RMSE: ${np.sqrt(-grid.best_score_):,.0f}")

    print("\nfeature importances:")
    importances = grid.best_estimator_.feature_importances_
    cat_names = list(full_pipeline.named_transformers_["cat"].categories_[0])
    for score, name in sorted(zip(importances, num_attribs + cat_names), reverse=True):
        print(f"  {score:.4f}  {name}")

    print()
    print("=" * 74)
    print("7. EVALUATE ON THE TEST SET")
    print("=" * 74)
    X_test = add_combined_attributes(strat_test.drop("median_house_value", axis=1))
    y_test = strat_test["median_house_value"].copy()
    final_pred = grid.best_estimator_.predict(full_pipeline.transform(X_test))
    final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    print(f"final test RMSE: ${final_rmse:,.0f}")

    sq_err = (final_pred - y_test) ** 2
    lo, hi = np.sqrt(stats.t.interval(0.95, len(sq_err) - 1,
                                      loc=sq_err.mean(), scale=stats.sem(sq_err)))
    print(f"95% confidence interval: ${lo:,.0f} to ${hi:,.0f}")
    print(f"median home value in the test set: ${y_test.median():,.0f}")
    print(f"error as a share of the median: {100 * final_rmse / y_test.median():.1f}%")
    print()
    print("Report the interval, not just the point estimate. On a sample this small the")
    print("interval is wide, and quoting the single number would overstate what we know.")


if __name__ == "__main__":
    main()
