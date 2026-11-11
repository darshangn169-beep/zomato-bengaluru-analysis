# Zomato Bengaluru Restaurant Analysis + "Restaurants Like This One" Recommender

EDA on cuisines, ratings and cost by area, plus a content-based recommender with a Streamlit app.

> **Data caveat:** Kaggle *Zomato Bangalore Restaurants* dataset, a ~2019 snapshot. Findings describe Bengaluru
> at that time, not today.

## Business questions
1. Which areas have the most restaurants, and which cuisines dominate each?
2. Where is a cuisine **over-supplied** or **under-served** compared with other areas? (cuisine × area heatmap)
3. Does a higher cost for two mean a better rating?
4. Do online ordering / table booking go with better ratings?
5. Which expensive restaurants are rated below median (overpriced), and which cheap ones are hidden gems?

## Key insights  <!-- fill with REAL numbers after running on zomato.csv -->
1. _Area with most restaurants: ___ (n = ___)_
2. _Correlation between cost and rating: ___ → meaning ____
3. _Cuisine gap: ___ is under-served in ___ → opportunity for ___ _
4. _Online order vs rating: ___ vs ____
5. _N expensive restaurants rated below median; examples in `charts/overpriced_candidates.csv`_

## Data cleaning (what was messy)
| Problem | Fix |
|---|---|
| `rate` as `"4.1/5"`, `"NEW"`, `"-"` | parsed to float; NEW/- → NaN (not 0) |
| `approx_cost` as `"1,200"` strings | stripped commas → numeric |
| Duplicate restaurants across listing types | dedupe on name + address |
| Cuisines in one comma-separated string | split to list, `primary_cuisine` added |
| Yes/No columns | converted to booleans |
| Missing area/cuisine | rows dropped (counts printed in the cleaning log) |

## Recommender
- Each restaurant = text of cuisines (weighted ×3) + restaurant type + dishes liked
- TF-IDF vectors → cosine similarity
- Filters: area, max budget, min rating, min votes
- **Limitations:** no user data (pure content-based), chains with many branches share similar vectors, 2019 data

## Run it
```bash
pip install -r requirements.txt
# download zomato.csv from Kaggle into data/
python src/clean.py data/zomato.csv data/zomato_clean.csv
python src/eda.py data/zomato_clean.csv     # charts -> charts/
streamlit run app.py
```
`data/make_sample.py` makes a synthetic file to test the pipeline (results from it are meaningless).

## Project layout
`src/clean.py` · `src/eda.py` · `src/recommender.py` · `app.py` · `notebooks/zomato_analysis.ipynb`

## Next ideas
Power BI / Tableau dashboard from `zomato_clean.csv`; sentiment on review text; hybrid recommender.
