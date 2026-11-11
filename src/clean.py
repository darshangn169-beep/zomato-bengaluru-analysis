"""Cleaning pipeline for the Kaggle 'Zomato Bangalore Restaurants' dataset (zomato.csv).

Usage:
    python src/clean.py data/zomato.csv data/zomato_clean.csv
"""
import sys
import numpy as np
import pandas as pd

RENAME = {
    "approx_cost(for two people)": "cost_for_two",
    "listed_in(type)": "listed_type",
    "listed_in(city)": "listed_city",
    "rate": "rating",
}
DROP = ["url", "phone", "menu_item", "reviews_list"]


def parse_rating(x):
    """'4.1/5' -> 4.1 ; 'NEW', '-', NaN -> NaN"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().split("/")[0]
    try:
        return float(s)
    except ValueError:  # 'NEW', '-'
        return np.nan


def parse_cost(x):
    """'1,200' -> 1200"""
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log = {"raw_rows": len(df)}
    df = df.rename(columns=RENAME).drop(columns=[c for c in DROP if c in df.columns])

    # 1. Duplicates (the raw file repeats the same restaurant across listing types)
    df = df.drop_duplicates(subset=["name", "address"]).copy()
    log["after_dedupe"] = len(df)

    # 2. Rating / cost / votes -> numeric
    log["rating_unparseable(NEW,-)"] = int(df["rating"].astype(str).str.strip().isin(["NEW", "-"]).sum())
    df["rating"] = df["rating"].map(parse_rating)
    df["cost_for_two"] = df["cost_for_two"].map(parse_cost)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

    # 3. Yes/No -> bool
    for c in ["online_order", "book_table"]:
        df[c] = df[c].str.strip().str.lower().eq("yes")

    # 4. Text columns
    for c in ["name", "location", "rest_type", "cuisines", "dish_liked"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["name"] = df["name"].str.title()

    # 5. Drop rows we cannot analyse (no area or no cuisine)
    df = df[(df["location"] != "") & (df["cuisines"] != "")]
    log["after_drop_missing_area_cuisine"] = len(df)

    # 6. Cuisine list + primary cuisine
    df["cuisine_list"] = df["cuisines"].str.split(",").map(lambda xs: [x.strip() for x in xs if x.strip()])
    df["primary_cuisine"] = df["cuisine_list"].str[0]

    # 7. Sanity: keep plausible cost values
    df.loc[(df["cost_for_two"] <= 0) | (df["cost_for_two"] > 10000), "cost_for_two"] = np.nan

    log["rating_missing"] = int(df["rating"].isna().sum())
    log["cost_missing"] = int(df["cost_for_two"].isna().sum())
    if verbose:
        print("--- cleaning log ---")
        for k, v in log.items():
            print(f"{k:35s} {v}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/zomato.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/zomato_clean.csv"
    out = clean(pd.read_csv(src))
    out.drop(columns=["cuisine_list"]).to_csv(dst, index=False)
    print(f"saved {len(out)} rows -> {dst}")
