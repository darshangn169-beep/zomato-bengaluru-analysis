"""Generates a SYNTHETIC sample (data/zomato_sample.csv) with the same messy schema as the
Kaggle file, so the pipeline can be tested without the real dataset.
Numbers from this file are meaningless - use the real zomato.csv for results."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
areas = ["Koramangala 5th Block", "Indiranagar", "BTM", "HSR", "Whitefield", "Jayanagar", "Marathahalli",
         "Electronic City", "Church Street", "MG Road", "Banashankari", "Malleshwaram"]
cuisines = ["North Indian", "South Indian", "Chinese", "Biryani", "Cafe", "Italian", "Fast Food", "Desserts",
            "Continental", "Street Food", "Mughlai", "Pizza", "Burger", "Asian", "Seafood"]
types = ["Casual Dining", "Quick Bites", "Cafe", "Delivery", "Bar", "Dessert Parlor", "Fine Dining"]
dishes = ["Biryani", "Paneer Tikka", "Pasta", "Pizza", "Dosa", "Burger", "Noodles", "Cheesecake", "Coffee",
          "Kebab", "Thali", "Fried Rice", "Brownie", "Wings", "Idli"]
rows = []
for i in range(3000):
    k = rng.integers(1, 4)
    cu = list(rng.choice(cuisines, k, replace=False))
    rate = rng.choice(["NEW", "-", None, f"{np.clip(rng.normal(3.7, .5), 2, 4.9):.1f}/5"], p=[.05, .03, .1, .82])
    cost = int(rng.choice([150, 250, 300, 400, 500, 600, 800, 1000, 1500, 2500]))
    rows.append({
        "url": "x", "address": f"addr {i % 2500}", "name": f"restaurant {i % 1800}",
        "online_order": rng.choice(["Yes", "No"]), "book_table": rng.choice(["Yes", "No"], p=[.2, .8]),
        "rate": rate, "votes": int(rng.integers(0, 2000)), "phone": "000",
        "location": rng.choice(areas), "rest_type": ", ".join(rng.choice(types, rng.integers(1, 3), replace=False)),
        "dish_liked": ", ".join(rng.choice(dishes, rng.integers(0, 4), replace=False)),
        "cuisines": ", ".join(cu),
        "approx_cost(for two people)": f"{cost:,}" if rng.random() > .03 else None,
        "reviews_list": "[]", "menu_item": "[]",
        "listed_in(type)": "Delivery", "listed_in(city)": "BTM",
    })
pd.DataFrame(rows).to_csv("data/zomato_sample.csv", index=False)
print("wrote data/zomato_sample.csv")
