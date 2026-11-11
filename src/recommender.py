"""Content-based 'restaurants like this one' recommender.

Each restaurant (name + area) becomes a text document made of cuisines, restaurant type
and dishes people liked. TF-IDF turns it into vectors; cosine similarity ranks neighbours.
Filters (area, budget, min rating) are applied after similarity so results stay relevant.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Recommender:
    def __init__(self, df: pd.DataFrame):
        d = df.copy()
        for c in ["name", "location", "cuisines", "rest_type", "dish_liked"]:
            d[c] = d[c].fillna("").astype(str)  # CSV round-trip turns "" into NaN
        # one row per (name, area): keep the most-voted listing
        d = d.sort_values("votes", ascending=False).drop_duplicates(["name", "location"]).reset_index(drop=True)
        d["key"] = d["name"] + " — " + d["location"]
        # cuisines weighted x3 so they dominate; dishes/type add flavour
        cuisines = d["cuisines"].str.replace(" ", "_").str.replace(",_", " ")
        d["soup"] = (
            (cuisines + " ") * 3
            + d["rest_type"].str.replace(" ", "_").str.replace(",_", " ") + " "
            + d["dish_liked"].str.lower().str.replace(",", " ")
        )
        self.df = d
        self.vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]+", sublinear_tf=True, min_df=2)
        self.X = self.vec.fit_transform(d["soup"])
        self.idx = pd.Series(d.index, index=d["key"])

    def keys(self):
        return self.df["key"].tolist()

    def recommend(self, key, n=10, area=None, max_cost=None, min_rating=None, min_votes=20):
        if key not in self.idx:
            raise KeyError(f"'{key}' not found")
        i = self.idx[key]
        sims = linear_kernel(self.X[i], self.X).ravel()
        out = self.df.assign(similarity=sims).drop(index=i)
        if area:
            out = out[out["location"] == area]
        if max_cost:
            out = out[out["cost_for_two"] <= max_cost]
        if min_rating:
            out = out[out["rating"] >= min_rating]
        out = out[out["votes"] >= min_votes]
        # rank by similarity, break ties with rating
        out = out.sort_values(["similarity", "rating"], ascending=False).head(n)
        return out[["name", "location", "cuisines", "rest_type", "rating", "cost_for_two", "votes", "similarity"]]
