import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from recommender import Recommender  # noqa: E402

st.set_page_config(page_title="Bengaluru Restaurant Recommender", page_icon="🍽️", layout="wide")

DATA = Path(__file__).parent / "data" / "zomato_clean.csv"


@st.cache_data
def load():
    return pd.read_csv(DATA)


@st.cache_resource
def get_model(df):
    return Recommender(df)


if not DATA.exists():
    st.error("Run `python src/clean.py data/zomato.csv data/zomato_clean.csv` first.")
    st.stop()

df = load()
rec = get_model(df)

st.title("🍽️ Restaurants like this one — Bengaluru")
st.caption("Content-based recommender (TF-IDF on cuisines, restaurant type, dishes liked). "
           "Data: Kaggle Zomato Bangalore dataset (~2019) — not current.")

tab1, tab2 = st.tabs(["Recommender", "EDA charts"])

with tab1:
    c1, c2 = st.columns([2, 1])
    key = c1.selectbox("Pick a restaurant you like", rec.keys(), index=0)
    n = c2.slider("How many results", 5, 20, 10)

    f1, f2, f3 = st.columns(3)
    areas = ["Any area"] + sorted(df["location"].unique())
    area = f1.selectbox("Area", areas)
    max_cost = f2.slider("Max cost for two (₹)", 100, 5000, 1500, step=100)
    min_rating = f3.slider("Min rating", 3.0, 4.8, 3.5, step=0.1)

    try:
        out = rec.recommend(key, n=n, area=None if area == "Any area" else area,
                            max_cost=max_cost, min_rating=min_rating)
        if out.empty:
            st.warning("No matches — loosen the filters.")
        else:
            out = out.rename(columns={"cost_for_two": "cost for two (₹)", "location": "area"})
            out["similarity"] = (out["similarity"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(out, width="stretch", hide_index=True)
    except KeyError as e:
        st.error(str(e))

with tab2:
    charts = sorted((Path(__file__).parent / "charts").glob("*.png"))
    if not charts:
        st.info("Run `python src/eda.py data/zomato_clean.csv` to generate charts.")
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, p in zip(cols, charts[i:i + 2]):
            col.image(str(p))
