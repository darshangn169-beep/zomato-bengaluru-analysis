"""EDA: answers business questions and saves charts to charts/.

Usage:
    python src/eda.py data/zomato_clean.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
OUT = Path("charts")
OUT.mkdir(exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=150)
    plt.close()


def load(path):
    df = pd.read_csv(path)
    df["cuisines"] = df["cuisines"].fillna("")
    df["cuisine_list"] = df["cuisines"].str.split(",").map(lambda xs: [x.strip() for x in xs])
    return df


def run(df):
    insights = {}

    # Q1. Where are restaurants concentrated?
    top_areas = df["location"].value_counts().head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_areas.values, y=top_areas.index, color="#E23744")
    plt.title("Top 15 areas by number of restaurants"); plt.xlabel("Restaurants"); plt.ylabel("")
    save("01_top_areas")
    insights["top_area"] = (top_areas.index[0], int(top_areas.iloc[0]))

    # Q2. Which cuisines dominate?
    cuisines = df.explode("cuisine_list")["cuisine_list"].value_counts().head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=cuisines.values, y=cuisines.index, color="#1F6FEB")
    plt.title("Top 15 cuisines"); plt.xlabel("Restaurants"); plt.ylabel("")
    save("02_top_cuisines")

    # Q3. Rating distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["rating"].dropna(), bins=25, color="#E23744")
    plt.title("Rating distribution"); plt.xlabel("Rating")
    save("03_rating_distribution")

    # Q4. Does higher cost mean higher rating?
    d = df.dropna(subset=["rating", "cost_for_two"]).copy()
    d["cost_band"] = pd.cut(d["cost_for_two"], [0, 300, 500, 800, 1200, 2000, 10000],
                            labels=["<300", "300-500", "500-800", "800-1200", "1200-2000", "2000+"])
    band = d.groupby("cost_band", observed=True)["rating"].agg(["mean", "count"])
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=band.index.astype(str).to_list(), y=band["mean"].to_numpy(), color="#2DA44E")
    plt.ylim(3, 4.5); plt.title("Average rating by cost-for-two band"); plt.xlabel("Cost for two (₹)"); plt.ylabel("Avg rating")
    save("04_rating_by_cost_band")
    insights["corr_cost_rating"] = round(d["cost_for_two"].corr(d["rating"]), 2)

    # Q5. Avg cost by area (areas with 50+ restaurants)
    area_stats = df.groupby("location").agg(n=("name", "size"), cost=("cost_for_two", "median"),
                                              rating=("rating", "mean")).query("n >= 50")
    top_cost = area_stats.sort_values("cost", ascending=False).head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_cost["cost"], y=top_cost.index, color="#8250DF")
    plt.title("Costliest areas (median cost for two, areas with 50+ restaurants)"); plt.xlabel("₹"); plt.ylabel("")
    save("05_costliest_areas")

    # Q6. Online ordering and table booking vs rating
    plt.figure(figsize=(8, 4.5))
    g = df.melt(id_vars="rating", value_vars=["online_order", "book_table"]).dropna()
    sns.barplot(data=g, x="variable", y="rating", hue="value", errorbar=None)
    plt.ylim(3, 4.5); plt.title("Avg rating: with vs without online order / table booking"); plt.xlabel("")
    save("06_online_booking_vs_rating")

    # Q7. Cuisine gaps: share of each top cuisine per top area (heatmap)
    areas = df["location"].value_counts().head(12).index
    cu = df.explode("cuisine_list")["cuisine_list"].value_counts().head(10).index
    ex = df.explode("cuisine_list").reset_index(drop=True)
    ex = ex[ex["location"].isin(areas) & ex["cuisine_list"].isin(cu)]
    share = pd.crosstab(ex["location"], ex["cuisine_list"], normalize="index") * 100
    plt.figure(figsize=(11, 6))
    sns.heatmap(share, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "% of cuisine mentions"})
    plt.title("Cuisine mix by area (where is each cuisine over/under-represented?)"); plt.ylabel(""); plt.xlabel("")
    save("07_cuisine_by_area_heatmap")

    # Q8. Overpriced-for-rating: expensive but below-median rating
    exp = df.dropna(subset=["rating", "cost_for_two"])
    exp = exp[(exp["votes"] >= 100)]
    hi = exp[exp["cost_for_two"] >= exp["cost_for_two"].quantile(0.8)]
    overpriced = hi[hi["rating"] < exp["rating"].median()].sort_values("cost_for_two", ascending=False)
    insights["overpriced_count"] = len(overpriced)
    overpriced.head(15)[["name", "location", "cost_for_two", "rating", "votes"]].to_csv(OUT / "overpriced_candidates.csv", index=False)

    # Q9. Hidden gems: high rating, low cost, decent votes
    gems = exp[(exp["rating"] >= 4.2) & (exp["cost_for_two"] <= 400)].sort_values("votes", ascending=False)
    gems.head(20)[["name", "location", "cuisines", "cost_for_two", "rating", "votes"]].to_csv(OUT / "hidden_gems.csv", index=False)

    # Print headline numbers for the README
    print("\n--- headline numbers ---")
    print("Most restaurants:", insights["top_area"])
    print("Cost vs rating correlation:", insights["corr_cost_rating"])
    print("Avg rating by cost band:\n", band.round(2))
    on = df.groupby("online_order")["rating"].mean().round(2).to_dict()
    bk = df.groupby("book_table")["rating"].mean().round(2).to_dict()
    print("Avg rating by online_order:", on, "| by book_table:", bk)
    print("Expensive-but-below-median-rating restaurants:", insights["overpriced_count"])
    print("Hidden gems found:", len(gems))
    print(f"charts saved to {OUT.resolve()}")


if __name__ == "__main__":
    run(load(sys.argv[1] if len(sys.argv) > 1 else "data/zomato_clean.csv"))
